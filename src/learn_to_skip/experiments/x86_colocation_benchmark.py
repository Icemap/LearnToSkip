"""x86 co-location validation experiment.

Compares separate vs co-located projected vector storage on x86.
Key metric: per-candidate skip overhead (ns).

Run: .venv/bin/python run_x86_colocation.py
"""
import time
import json
import numpy as np

# Use system hnswlib
import hnswlib

SEED = 42
M = 16
EF_C = 200
PROJ_DIM = 16
THRESHOLD = 0.7

# Universal tree params
TREE_PARAMS = {
    "rank_thresholds": np.array([545.455, 205.390, 921.405]),
    "adist_thresholds": np.array([13.823, 15.035, 10.380]),
    "ins_t0": 2146.184,
    "leaf_probas": np.array([0.133092, 0.241549, 0.345719, 0.537761,
                             0.559158, 0.763484, 0.738662, 0.870165]),
}


def log(msg):
    print(msg, flush=True)


def load_sift1m():
    import h5py
    path = "data/sift-128-euclidean.hdf5"
    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
    return train, test, neighbors


def compute_recall(index, queries, gt, k, ef_search):
    index.set_ef(ef_search)
    labels, _ = index.knn_query(queries, k=k)
    gt_k = gt[:, :k]
    hits = sum(len(set(labels[i]) & set(gt_k[i])) for i in range(len(queries)))
    return hits / (len(queries) * k)


def run_vanilla(data):
    """Vanilla single-threaded baseline."""
    n, dim = data.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    t0 = time.time()
    idx.add_items(data, np.arange(n), num_threads=1)
    build_time = time.time() - t0
    metrics = idx.get_construction_metrics()
    return idx, build_time, metrics


def run_skip_separate(data, data_proj, threshold):
    """Skip with SEPARATE projected data array (old mode)."""
    n, dim = data.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    # DO NOT enable_projection_storage => separate mode
    idx.create_skip_functor(threshold=threshold, proj_dim=PROJ_DIM)
    idx.set_skip_projected_data(data_proj)
    idx.set_skip_tree_params(**TREE_PARAMS)
    idx.activate_skip()
    t0 = time.time()
    idx.add_items_with_skip(data, data_proj)
    build_time = time.time() - t0
    metrics = idx.get_construction_metrics()
    idx.deactivate_skip()
    idx.clear_skip_functor()
    return idx, build_time, metrics


def run_skip_colocated(data, data_proj, threshold):
    """Skip with CO-LOCATED projected data (new mode)."""
    n, dim = data.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    idx.enable_projection_storage(PROJ_DIM)  # <-- co-locate
    idx.create_skip_functor(threshold=threshold, proj_dim=PROJ_DIM)
    idx.set_skip_projected_data(data_proj)
    idx.set_skip_tree_params(**TREE_PARAMS)
    idx.activate_skip()
    t0 = time.time()
    idx.add_items_with_skip(data, data_proj)
    build_time = time.time() - t0
    metrics = idx.get_construction_metrics()
    idx.deactivate_skip()
    idx.clear_skip_functor()
    return idx, build_time, metrics


def analyze_overhead(vanilla_time, vanilla_metrics, skip_time, skip_metrics, label):
    """Compute per-candidate overhead in ns."""
    v_dist = vanilla_metrics["distance_computations"]
    s_dist = skip_metrics["distance_computations"]
    s_skip = skip_metrics["skipped_computations"]
    s_total = s_dist + s_skip
    skip_rate = s_skip / s_total if s_total > 0 else 0

    # Per-candidate full distance cost (from vanilla)
    dist_cost_ns = (vanilla_time / v_dist) * 1e9

    # Per-candidate total cost in skip mode
    total_cost_ns = (skip_time / s_total) * 1e9

    # Overhead = total_cost - (1 - skip_rate) * dist_cost
    # i.e., the extra time per candidate beyond what distance-only would cost
    overhead_ns = total_cost_ns - (1 - skip_rate) * dist_cost_ns

    speedup = vanilla_time / skip_time if skip_time > 0 else 0

    log(f"\n  [{label}]")
    log(f"  build_time={skip_time:.1f}s, speedup={speedup:.2f}x")
    log(f"  skip_rate={skip_rate:.2%}")
    log(f"  dist_comp={s_dist}, skipped={s_skip}")
    log(f"  per-candidate dist cost: {dist_cost_ns:.1f}ns")
    log(f"  per-candidate total cost: {total_cost_ns:.1f}ns")
    log(f"  per-candidate OVERHEAD: {overhead_ns:.1f}ns")

    return {
        "build_time": skip_time,
        "speedup": speedup,
        "skip_rate": skip_rate,
        "distance_computations": s_dist,
        "skipped_computations": s_skip,
        "per_candidate_dist_ns": dist_cost_ns,
        "per_candidate_total_ns": total_cost_ns,
        "per_candidate_overhead_ns": overhead_ns,
    }


def main():
    log("Loading SIFT1M...")
    data, queries, gt = load_sift1m()
    n, dim = data.shape
    log(f"n={n}, dim={dim}")

    # Random projection
    rng = np.random.RandomState(SEED)
    proj_matrix = rng.randn(dim, PROJ_DIM).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)

    results = {}

    # 1. Vanilla
    log("\n=== Vanilla (1T) ===")
    v_idx, v_time, v_metrics = run_vanilla(data)
    v_dist = v_metrics["distance_computations"]
    dist_cost_ns = (v_time / v_dist) * 1e9
    log(f"  build_time={v_time:.1f}s")
    log(f"  dist_computations={v_dist}")
    log(f"  per-candidate dist cost: {dist_cost_ns:.1f}ns")
    v_recall = compute_recall(v_idx, queries, gt, 10, 200)
    log(f"  R@10(ef200)={v_recall:.4f}")
    results["vanilla"] = {
        "build_time": v_time,
        "distance_computations": v_dist,
        "per_candidate_dist_ns": dist_cost_ns,
        "recall_10_ef200": v_recall,
    }

    # 2. Skip with SEPARATE projected data (tau=0.7)
    log("\n=== Skip SEPARATE (τ=0.7) ===")
    s_idx, s_time, s_metrics = run_skip_separate(data, data_proj, THRESHOLD)
    sep_results = analyze_overhead(v_time, v_metrics, s_time, s_metrics, "SEPARATE")
    sep_results["recall_10_ef200"] = compute_recall(s_idx, queries, gt, 10, 200)
    log(f"  R@10(ef200)={sep_results['recall_10_ef200']:.4f}")
    results["skip_separate_0.7"] = sep_results

    # 3. Skip with CO-LOCATED projected data (tau=0.7)
    log("\n=== Skip CO-LOCATED (τ=0.7) ===")
    c_idx, c_time, c_metrics = run_skip_colocated(data, data_proj, THRESHOLD)
    col_results = analyze_overhead(v_time, v_metrics, c_time, c_metrics, "CO-LOCATED")
    col_results["recall_10_ef200"] = compute_recall(c_idx, queries, gt, 10, 200)
    log(f"  R@10(ef200)={col_results['recall_10_ef200']:.4f}")
    results["skip_colocated_0.7"] = col_results

    # 4. Also test tau=0.5 and tau=0.8 co-located
    for tau in [0.5, 0.8]:
        log(f"\n=== Skip CO-LOCATED (τ={tau}) ===")
        t_idx, t_time, t_metrics = run_skip_colocated(data, data_proj, tau)
        t_results = analyze_overhead(v_time, v_metrics, t_time, t_metrics, f"CO-LOCATED τ={tau}")
        t_results["recall_10_ef200"] = compute_recall(t_idx, queries, gt, 10, 200)
        log(f"  R@10(ef200)={t_results['recall_10_ef200']:.4f}")
        results[f"skip_colocated_{tau}"] = t_results

    # Save
    with open("results/x86_colocation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to results/x86_colocation_results.json")

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY: Per-candidate overhead (ns)")
    log("=" * 60)
    log(f"  Full distance cost:  {results['vanilla']['per_candidate_dist_ns']:.1f}ns")
    log(f"  SEPARATE overhead:   {results['skip_separate_0.7']['per_candidate_overhead_ns']:.1f}ns")
    log(f"  CO-LOCATED overhead: {results['skip_colocated_0.7']['per_candidate_overhead_ns']:.1f}ns")
    log(f"  Speedup separate:    {results['skip_separate_0.7']['speedup']:.2f}x")
    log(f"  Speedup co-located:  {results['skip_colocated_0.7']['speedup']:.2f}x")


if __name__ == "__main__":
    main()
