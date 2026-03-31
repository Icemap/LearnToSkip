"""GIST1M (d=960, L2) benchmark for high-dimensional evaluation.

Addresses reviewer concern: "How does LearnToSkip perform on higher-dimensional embeddings?"
GIST1M has d=960, much larger than SIFT (128) or Deep (96).
"""
import time
import json
import os
import numpy as np
from pathlib import Path

import hnswlib

SEED = 42
M = 16
EF_C = 200
PROJ_DIM = 16

TREE_PARAMS = {
    "rank_thresholds": np.array([545.455, 205.390, 921.405]),
    "adist_thresholds": np.array([13.823, 15.035, 10.380]),
    "ins_t0": 2146.184,
    "leaf_probas": np.array([0.133092, 0.241549, 0.345719, 0.537761,
                             0.559158, 0.763484, 0.738662, 0.870165]),
}


def log(msg):
    print(msg, flush=True)


def load_gist1m():
    from learn_to_skip.datasets.gist import Gist1MDataset
    ds = Gist1MDataset()
    train = ds.load_train()
    test = ds.load_query()
    neighbors = ds.load_groundtruth()
    return train, test, neighbors


def compute_recall(index, queries, gt, k, ef_search):
    index.set_ef(ef_search)
    labels, _ = index.knn_query(queries, k=k)
    gt_k = gt[:, :k]
    hits = sum(len(set(labels[i]) & set(gt_k[i])) for i in range(len(queries)))
    return hits / (len(queries) * k)


def run_vanilla(data):
    n, dim = data.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    t0 = time.time()
    idx.add_items(data, np.arange(n), num_threads=1)
    build_time = time.time() - t0
    metrics = idx.get_construction_metrics()
    return idx, build_time, metrics


def run_skip(data, data_proj, tau):
    n, dim = data.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    idx.enable_projection_storage(PROJ_DIM)
    idx.create_skip_functor(threshold=tau, proj_dim=PROJ_DIM)
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


def run_combined(data, data_proj, alpha, tau):
    n, dim = data.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    idx.enable_projection_storage(PROJ_DIM)
    idx.create_combined_functor(alpha=alpha, tree_threshold=tau, proj_dim=PROJ_DIM)
    idx.set_combined_projected_data(data_proj)
    idx.set_combined_tree_params(**TREE_PARAMS)
    idx.activate_combined_skip()
    t0 = time.time()
    idx.add_items_with_combined_skip(data, data_proj)
    build_time = time.time() - t0
    metrics = idx.get_construction_metrics()
    combined_metrics = idx.get_combined_metrics()
    return idx, build_time, metrics, combined_metrics


def run_reduced_efc(data, efc):
    n, dim = data.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=efc, random_seed=SEED)
    t0 = time.time()
    idx.add_items(data, np.arange(n), num_threads=1)
    build_time = time.time() - t0
    metrics = idx.get_construction_metrics()
    return idx, build_time, metrics


def main():
    log("Loading GIST1M (d=960)...")
    data, queries, gt = load_gist1m()
    n, dim = data.shape
    log(f"n={n}, dim={dim}")

    rng = np.random.RandomState(SEED)
    proj_matrix = rng.randn(dim, PROJ_DIM).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)
    log(f"Projection: {dim} -> {PROJ_DIM}")

    results = {}
    results_path = Path("results/gist1m_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    # 1. Vanilla
    log("\n=== Vanilla ===")
    v_idx, v_time, v_metrics = run_vanilla(data)
    v_recall = compute_recall(v_idx, queries, gt, 10, 200)
    log(f"  time={v_time:.1f}s, dist={v_metrics['distance_computations']}, R@10={v_recall:.4f}")
    results["vanilla"] = {
        "build_time": v_time,
        "distance_computations": v_metrics["distance_computations"],
        "recall_10_ef200": v_recall,
    }

    # 2. LearnToSkip at several τ
    for tau in [0.5, 0.6, 0.7, 0.8]:
        key = f"skip_tau{tau}"
        log(f"\n=== Skip τ={tau} ===")
        s_idx, s_time, s_metrics = run_skip(data, data_proj, tau)
        total = s_metrics["distance_computations"] + s_metrics["skipped_computations"]
        skip_rate = s_metrics["skipped_computations"] / total
        recall = compute_recall(s_idx, queries, gt, 10, 200)
        speedup = v_time / s_time
        log(f"  time={s_time:.1f}s, skip={skip_rate:.2%}, R@10={recall:.4f}, speedup={speedup:.2f}x")
        results[key] = {
            "build_time": s_time, "speedup": speedup,
            "skip_rate": skip_rate, "recall_10_ef200": recall,
        }

    # 3. Combined (best config from SIFT1M)
    for alpha, tau in [(3.0, 0.8), (5.0, 0.8)]:
        key = f"combined_a{alpha}_t{tau}"
        log(f"\n=== Combined α={alpha}, τ={tau} ===")
        c_idx, c_time, c_metrics, cm = run_combined(data, data_proj, alpha, tau)
        total = c_metrics["distance_computations"] + c_metrics["skipped_computations"]
        skip_rate = c_metrics["skipped_computations"] / total
        recall = compute_recall(c_idx, queries, gt, 10, 200)
        speedup = v_time / c_time
        log(f"  time={c_time:.1f}s, skip={skip_rate:.2%}, R@10={recall:.4f}, speedup={speedup:.2f}x")
        results[key] = {
            "build_time": c_time, "speedup": speedup,
            "skip_rate": skip_rate, "recall_10_ef200": recall,
        }

    # 4. Reduced ef_c baselines
    for efc in [100, 50]:
        key = f"reduced_efc{efc}"
        log(f"\n=== Reduced ef_c={efc} ===")
        r_idx, r_time, r_metrics = run_reduced_efc(data, efc)
        recall = compute_recall(r_idx, queries, gt, 10, 200)
        speedup = v_time / r_time
        log(f"  time={r_time:.1f}s, R@10={recall:.4f}, speedup={speedup:.2f}x")
        results[key] = {
            "build_time": r_time, "speedup": speedup,
            "recall_10_ef200": recall,
        }

    # Save
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {results_path}")

    # Summary
    log("\n" + "=" * 60)
    log(f"{'Method':<25} {'Skip%':>6} {'R@10':>7} {'Speedup':>8}")
    log("-" * 60)
    for key, val in results.items():
        skip = val.get("skip_rate", None)
        speedup = val.get("speedup", 1.0)
        if skip is not None:
            log(f"{key:<25} {skip:>5.1%} {val['recall_10_ef200']:>7.4f} {speedup:>7.2f}x")
        else:
            log(f"{key:<25} {'---':>6} {val['recall_10_ef200']:>7.4f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
