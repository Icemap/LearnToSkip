"""Combined baseline experiment: α-pruning + learned tree.

Tests whether compounding α-pruning with the learned tree yields
better skip rates than either alone, at matched recall levels.

Runs on SIFT1M with co-located projected data.
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


def load_sift1m():
    from learn_to_skip.datasets.sift import Sift1MDataset
    ds = Sift1MDataset()
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


def run_tree_only(data, data_proj, tau):
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


def run_alpha_only(data, data_proj, alpha):
    n, dim = data.shape
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    idx.enable_projection_storage(PROJ_DIM)
    idx.create_alpha_pruning_functor(alpha=alpha, proj_dim=PROJ_DIM)
    idx.set_alpha_pruning_projected_data(data_proj)
    idx.activate_alpha_pruning_skip()
    t0 = time.time()
    idx.add_items_with_alpha_skip(data, data_proj)
    build_time = time.time() - t0
    metrics = idx.get_construction_metrics()
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


def main():
    log("Loading SIFT1M...")
    data, queries, gt = load_sift1m()
    n, dim = data.shape
    log(f"n={n}, dim={dim}")

    rng = np.random.RandomState(SEED)
    proj_matrix = rng.randn(dim, PROJ_DIM).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)

    results = {}

    # Load existing results
    results_path = Path("results/combined_baseline_results.json")
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

    # 2. Tree-only at multiple τ
    for tau in [0.5, 0.6, 0.7, 0.8]:
        key = f"tree_tau{tau}"
        log(f"\n=== Tree τ={tau} ===")
        t_idx, t_time, t_metrics = run_tree_only(data, data_proj, tau)
        total = t_metrics["distance_computations"] + t_metrics["skipped_computations"]
        skip_rate = t_metrics["skipped_computations"] / total
        recall = compute_recall(t_idx, queries, gt, 10, 200)
        speedup = v_time / t_time
        log(f"  time={t_time:.1f}s, skip={skip_rate:.2%}, R@10={recall:.4f}, speedup={speedup:.2f}x")
        results[key] = {
            "build_time": t_time, "speedup": speedup,
            "skip_rate": skip_rate, "recall_10_ef200": recall,
            "distance_computations": t_metrics["distance_computations"],
            "skipped_computations": t_metrics["skipped_computations"],
        }

    # 3. α-pruning only at multiple α
    for alpha in [2.0, 3.0, 5.0]:
        key = f"alpha_{alpha}"
        log(f"\n=== α-pruning α={alpha} ===")
        a_idx, a_time, a_metrics = run_alpha_only(data, data_proj, alpha)
        total = a_metrics["distance_computations"] + a_metrics["skipped_computations"]
        skip_rate = a_metrics["skipped_computations"] / total
        recall = compute_recall(a_idx, queries, gt, 10, 200)
        speedup = v_time / a_time
        log(f"  time={a_time:.1f}s, skip={skip_rate:.2%}, R@10={recall:.4f}, speedup={speedup:.2f}x")
        results[key] = {
            "build_time": a_time, "speedup": speedup,
            "skip_rate": skip_rate, "recall_10_ef200": recall,
            "distance_computations": a_metrics["distance_computations"],
            "skipped_computations": a_metrics["skipped_computations"],
        }

    # 4. Combined at several (α, τ) pairs
    for alpha, tau in [(3.0, 0.7), (3.0, 0.8), (5.0, 0.7), (5.0, 0.8), (2.0, 0.7)]:
        key = f"combined_a{alpha}_t{tau}"
        log(f"\n=== Combined α={alpha}, τ={tau} ===")
        c_idx, c_time, c_metrics, cm = run_combined(data, data_proj, alpha, tau)
        total = c_metrics["distance_computations"] + c_metrics["skipped_computations"]
        skip_rate = c_metrics["skipped_computations"] / total
        recall = compute_recall(c_idx, queries, gt, 10, 200)
        speedup = v_time / c_time
        alpha_frac = cm["alpha_skip_count"] / c_metrics["skipped_computations"] if c_metrics["skipped_computations"] > 0 else 0
        log(f"  time={c_time:.1f}s, skip={skip_rate:.2%}, R@10={recall:.4f}, speedup={speedup:.2f}x")
        log(f"  α-skips={cm['alpha_skip_count']}, tree-skips={cm['tree_skip_count']}, α-frac={alpha_frac:.1%}")
        results[key] = {
            "build_time": c_time, "speedup": speedup,
            "skip_rate": skip_rate, "recall_10_ef200": recall,
            "distance_computations": c_metrics["distance_computations"],
            "skipped_computations": c_metrics["skipped_computations"],
            "alpha_skip_count": cm["alpha_skip_count"],
            "tree_skip_count": cm["tree_skip_count"],
            "alpha_fraction": alpha_frac,
        }

    # Save
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {results_path}")

    # Summary table
    log("\n" + "=" * 70)
    log(f"{'Method':<25} {'Skip%':>6} {'R@10':>7} {'Speedup':>8}")
    log("-" * 70)
    for key, val in results.items():
        if key == "vanilla":
            log(f"{'Vanilla':<25} {'—':>6} {val['recall_10_ef200']:>7.4f} {'1.00x':>8}")
        else:
            log(f"{key:<25} {val['skip_rate']:>5.1%} {val['recall_10_ef200']:>7.4f} {val['speedup']:>7.2f}x")


if __name__ == "__main__":
    main()
