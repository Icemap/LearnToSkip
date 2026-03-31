"""New experiments for paper revision addressing reviewer feedback.

Runs on Mac (M1 Pro) for:
1. Heuristic baselines (rank-only, distance-only)
2. Projected dimension d' sweep
3. Direct waste measurement (actual edge count)
4. Graph structure analysis (degree distributions)
5. Multi-threaded skip experiments

Results saved to results/heuristic_baselines_multithread.json
"""
import json
import time
import sys
from pathlib import Path

import numpy as np
import hnswlib

from learn_to_skip.datasets import get_dataset
from learn_to_skip.builders.cpp_learned_skip import CppLearnedSkipBuilder, DEFAULT_TREE_PARAMS
from learn_to_skip.config import RESULTS_DIR, SEED

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

M = 16
EF_C = 200
K_VALUES = [1, 10, 100]


def log(msg):
    print(msg, flush=True)


def compute_recall(index, queries, gt, k, ef_search):
    index.set_ef(ef_search)
    labels, _ = index.knn_query(queries, k=k)
    gt_k = gt[:, :k]
    hits = sum(len(set(labels[i]) & set(gt_k[i])) for i in range(len(queries)))
    return hits / (len(queries) * k)


def build_vanilla(data, M, ef_c, metric, num_threads=1):
    n, dim = data.shape
    space = "l2" if metric == "l2" else "cosine"
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, M=M, ef_construction=ef_c, random_seed=SEED)
    index.set_num_threads(num_threads)
    t0 = time.time()
    index.add_items(data, np.arange(n), num_threads=num_threads)
    build_time = time.time() - t0
    return index, build_time


def run_heuristic_baselines(ds_name="sift1m"):
    """Experiment 1: Non-learned heuristic baselines."""
    log(f"\n{'='*60}")
    log(f"  Heuristic Baselines on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    space = "l2" if meta.metric == "l2" else "cosine"
    n, dim = data.shape

    # Vanilla baseline
    log("  Building vanilla baseline...")
    vanilla_idx, vanilla_time = build_vanilla(data, M, EF_C, meta.metric)
    vanilla_r10 = compute_recall(vanilla_idx, queries, gt, 10, 200)
    log(f"  Vanilla: {vanilla_time:.1f}s, R@10={vanilla_r10:.4f}")
    vanilla_metrics = vanilla_idx.get_construction_metrics()

    # Random projection for dist-only
    rng = np.random.RandomState(SEED)
    proj_matrix = rng.randn(dim, 16).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)

    results = {"vanilla_time": vanilla_time, "vanilla_r10": vanilla_r10}

    # Rank-only baselines
    log("  Rank-only baselines:")
    rank_results = {}
    for cutoff in [100, 200, 400, 600, 800]:
        log(f"    cutoff={cutoff}...")
        idx = hnswlib.Index(space=space, dim=dim)
        idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
        idx.create_rank_only_functor(rank_cutoff=cutoff)
        idx.activate_rank_only_skip()
        t0 = time.time()
        idx.add_items_with_rank_skip(data, np.arange(n))
        bt = time.time() - t0
        m = idx.get_construction_metrics()
        idx.deactivate_any_skip()
        r10 = compute_recall(idx, queries, gt, 10, 200)
        skip_rate = m["skipped_computations"] / max(1, m["distance_computations"] + m["skipped_computations"])
        rank_results[str(cutoff)] = {
            "time": bt, "speedup": vanilla_time / bt,
            "skip_rate": skip_rate, "recall_10": r10,
            "dist_comps": m["distance_computations"], "skipped": m["skipped_computations"]
        }
        log(f"      {bt:.1f}s, speedup={vanilla_time/bt:.2f}x, skip={skip_rate:.3f}, R@10={r10:.4f}")
    results["rank_only"] = rank_results

    # Distance-only baselines
    log("  Distance-only baselines:")
    # Compute percentile thresholds from projected distances
    sample_dists = []
    rng2 = np.random.RandomState(42)
    for _ in range(1000):
        i, j = rng2.randint(0, n, 2)
        d = np.sum((data_proj[i] - data_proj[j]) ** 2)
        sample_dists.append(d)
    sample_dists = np.array(sample_dists)
    percentiles = {
        "p50": float(np.percentile(sample_dists, 50)),
        "p75": float(np.percentile(sample_dists, 75)),
        "p90": float(np.percentile(sample_dists, 90)),
        "p95": float(np.percentile(sample_dists, 95)),
    }
    log(f"    Projected distance percentiles: {percentiles}")

    dist_results = {}
    for pname, cutoff in percentiles.items():
        log(f"    {pname} (cutoff={cutoff:.2f})...")
        idx = hnswlib.Index(space=space, dim=dim)
        idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
        idx.create_dist_only_functor(dist_cutoff=cutoff, proj_dim=16)
        idx.set_dist_only_projected_data(data_proj)
        idx.activate_dist_only_skip()
        t0 = time.time()
        idx.add_items_with_dist_skip(data, data_proj, np.arange(n))
        bt = time.time() - t0
        m = idx.get_construction_metrics()
        idx.deactivate_any_skip()
        r10 = compute_recall(idx, queries, gt, 10, 200)
        skip_rate = m["skipped_computations"] / max(1, m["distance_computations"] + m["skipped_computations"])
        dist_results[pname] = {
            "cutoff": cutoff, "time": bt, "speedup": vanilla_time / bt,
            "skip_rate": skip_rate, "recall_10": r10,
        }
        log(f"      {bt:.1f}s, speedup={vanilla_time/bt:.2f}x, skip={skip_rate:.3f}, R@10={r10:.4f}")
    results["dist_only"] = dist_results

    return results


def run_proj_dim_sweep(ds_name="sift1m"):
    """Experiment 2: Projected dimension d' sweep."""
    log(f"\n{'='*60}")
    log(f"  Projected Dimension Sweep on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    n, dim = data.shape

    # Vanilla baseline
    log("  Building vanilla baseline...")
    vanilla_idx, vanilla_time = build_vanilla(data, M, EF_C, meta.metric)

    results = {"vanilla_time": vanilla_time}

    for proj_dim in [8, 16, 32]:
        log(f"  d'={proj_dim}:")
        for tau in [0.5, 0.7, 0.8]:
            builder = CppLearnedSkipBuilder(threshold=tau, proj_dim=proj_dim, mode="universal", seed=SEED)
            result = builder.build(data, M=M, ef_construction=EF_C, metric=meta.metric)
            r10 = compute_recall(result.index, queries, gt, 10, 200)
            speedup = vanilla_time / result.build_time_seconds
            log(f"    tau={tau}: {result.build_time_seconds:.1f}s, speedup={speedup:.2f}x, "
                f"skip={result.skip_rate:.3f}, R@10={r10:.4f}")
            key = f"d{proj_dim}_tau{tau}"
            results[key] = {
                "proj_dim": proj_dim, "tau": tau,
                "time": result.build_time_seconds, "speedup": speedup,
                "skip_rate": result.skip_rate, "recall_10": r10,
            }

    return results


def run_direct_waste_measurement(ds_name="sift1m"):
    """Experiment 3: Direct waste measurement using actual edge counts."""
    log(f"\n{'='*60}")
    log(f"  Direct Waste Measurement on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    n, dim = data.shape
    space = "l2" if meta.metric == "l2" else "cosine"

    # Build vanilla with instrumentation
    log("  Building instrumented vanilla index...")
    idx = hnswlib.Index(space=space, dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    idx.add_items(data, np.arange(n))

    total_evals = idx.get_construction_metrics()["distance_computations"]
    actual_edges = idx.get_edge_count_level0()
    degrees = idx.get_degree_distribution()
    waste_ratio = 1.0 - (actual_edges / total_evals) if total_evals > 0 else 0.0

    log(f"  Total evaluations: {total_evals:,}")
    log(f"  Actual edges (level 0): {actual_edges:,}")
    log(f"  Estimated (2nM): {n * 2 * M:,}")
    log(f"  Waste ratio (measured): {waste_ratio:.6f}")
    log(f"  Degrees: min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.2f}, std={degrees.std():.2f}")

    return {
        "total_evaluations": int(total_evals),
        "actual_edges_level0": int(actual_edges),
        "estimated_2nM": n * 2 * M,
        "waste_ratio_measured": waste_ratio,
        "degree_mean": float(degrees.mean()),
        "degree_std": float(degrees.std()),
        "degree_min": int(degrees.min()),
        "degree_max": int(degrees.max()),
    }


def run_graph_analysis(ds_name="sift1m"):
    """Experiment 4: Graph structure analysis — degree distributions at various tau."""
    log(f"\n{'='*60}")
    log(f"  Graph Structure Analysis on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    n, dim = data.shape
    space = "l2" if meta.metric == "l2" else "cosine"

    results = {}

    # Vanilla
    log("  Building vanilla...")
    idx, bt = build_vanilla(data, M, EF_C, meta.metric)
    deg = idx.get_degree_distribution()
    r10 = compute_recall(idx, queries, gt, 10, 200)
    results["vanilla"] = {
        "degree_mean": float(deg.mean()), "degree_std": float(deg.std()),
        "degree_min": int(deg.min()), "degree_max": int(deg.max()),
        "edges": int(idx.get_edge_count_level0()), "recall_10": r10,
        "degree_histogram": np.bincount(deg).tolist(),
    }
    log(f"    Edges: {idx.get_edge_count_level0()}, Mean degree: {deg.mean():.2f}, R@10={r10:.4f}")

    # Skip at various tau
    for tau in [0.5, 0.7, 0.8]:
        log(f"  Building skip tau={tau}...")
        builder = CppLearnedSkipBuilder(threshold=tau, proj_dim=16, mode="universal", seed=SEED)
        result = builder.build(data, M=M, ef_construction=EF_C, metric=meta.metric)
        deg = result.index.get_degree_distribution()
        r10 = compute_recall(result.index, queries, gt, 10, 200)
        results[f"tau_{tau}"] = {
            "degree_mean": float(deg.mean()), "degree_std": float(deg.std()),
            "degree_min": int(deg.min()), "degree_max": int(deg.max()),
            "edges": int(result.index.get_edge_count_level0()), "recall_10": r10,
            "degree_histogram": np.bincount(deg).tolist(),
        }
        log(f"    Edges: {result.index.get_edge_count_level0()}, Mean degree: {deg.mean():.2f}, R@10={r10:.4f}")

    return results


def run_multithread_experiments(ds_name="sift1m"):
    """Experiment 5: Multi-threaded skip construction."""
    log(f"\n{'='*60}")
    log(f"  Multi-Threaded Skip on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    n, dim = data.shape
    space = "l2" if meta.metric == "l2" else "cosine"

    # Prepare projection
    rng = np.random.RandomState(SEED)
    proj_matrix = rng.randn(dim, 16).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)

    results = {}

    # Vanilla baselines at different thread counts
    for nt in [1, 2, 4, 8, 10]:
        log(f"  Vanilla {nt}T...")
        idx, bt = build_vanilla(data, M, EF_C, meta.metric, num_threads=nt)
        r10 = compute_recall(idx, queries, gt, 10, 200)
        results[f"vanilla_{nt}t"] = {"time": bt, "recall_10": r10}
        log(f"    {bt:.1f}s, R@10={r10:.4f}")

    vanilla_1t = results["vanilla_1t"]["time"]

    # Multi-threaded skip at various tau and thread counts
    for tau in [0.5, 0.7, 0.8]:
        for nt in [1, 2, 4, 8, 10]:
            log(f"  Skip tau={tau}, {nt}T...")
            idx = hnswlib.Index(space=space, dim=dim)
            idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
            idx.create_skip_functor(threshold=tau, proj_dim=16)
            idx.set_skip_projected_data(data_proj)
            idx.set_skip_tree_params(
                rank_thresholds=DEFAULT_TREE_PARAMS["rank_thresholds"],
                adist_thresholds=DEFAULT_TREE_PARAMS["adist_thresholds"],
                ins_t0=DEFAULT_TREE_PARAMS["ins_t0"],
                leaf_probas=DEFAULT_TREE_PARAMS["leaf_probas"],
            )
            idx.activate_skip()
            idx.set_num_threads(nt)

            t0 = time.time()
            if nt == 1:
                idx.add_items_with_skip(data, data_proj, np.arange(n))
            else:
                idx.add_items_with_skip_mt(data, data_proj, np.arange(n), num_threads=nt)
            bt = time.time() - t0
            m = idx.get_construction_metrics()
            skip_rate = m["skipped_computations"] / max(1, m["distance_computations"] + m["skipped_computations"])
            idx.deactivate_skip()
            r10 = compute_recall(idx, queries, gt, 10, 200)
            key = f"skip_tau{tau}_{nt}t"
            results[key] = {
                "time": bt, "speedup_vs_1t": vanilla_1t / bt,
                "skip_rate": skip_rate, "recall_10": r10,
            }
            log(f"    {bt:.1f}s, speedup={vanilla_1t/bt:.2f}x, skip={skip_rate:.3f}, R@10={r10:.4f}")

    return results


def main():
    experiments = sys.argv[1:] if len(sys.argv) > 1 else [
        "heuristic", "proj_dim", "waste", "graph", "multithread"
    ]

    all_results = {}

    for exp in experiments:
        try:
            if exp == "heuristic":
                all_results["heuristic_baselines"] = run_heuristic_baselines("sift1m")
            elif exp == "proj_dim":
                all_results["proj_dim_sweep"] = run_proj_dim_sweep("sift1m")
            elif exp == "waste":
                all_results["waste_sift1m"] = run_direct_waste_measurement("sift1m")
                all_results["waste_deep1m"] = run_direct_waste_measurement("deep1m")
            elif exp == "graph":
                all_results["graph_analysis"] = run_graph_analysis("sift1m")
            elif exp == "multithread":
                all_results["multithread"] = run_multithread_experiments("sift1m")
            else:
                log(f"Unknown experiment: {exp}")
        except Exception as e:
            log(f"ERROR in {exp}: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp] = {"error": str(e)}

    output_path = RESULTS_DIR / "heuristic_baselines_multithread.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
