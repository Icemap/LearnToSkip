"""α-pruning baseline experiment (OptHNSW comparison).

Compares learned tree skip vs α-pruning at matched operating points on SIFT1M.
Also tests co-located vs separate projected data storage.

Results saved to results/alpha_pruning_results.json (merge-on-write).
"""
import json
import time
import sys
from pathlib import Path

import numpy as np
import hnswlib

from learn_to_skip.datasets import get_dataset
from learn_to_skip.builders.cpp_learned_skip import DEFAULT_TREE_PARAMS
from learn_to_skip.config import RESULTS_DIR, SEED

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "alpha_pruning_results.json"

M = 16
EF_C = 200
PROJ_DIM = 16
ALPHA_VALUES = [1.5, 2.0, 3.0, 5.0, 8.0]
TREE_THRESHOLDS = [0.5, 0.7, 0.8, 0.9]
K_VALUES = [1, 10, 100]
EF_SEARCH_VALUES = [50, 100, 200]


def log(msg):
    print(msg, flush=True)


def compute_recall(index, queries, gt, k, ef_search):
    index.set_ef(ef_search)
    labels, _ = index.knn_query(queries, k=k)
    gt_k = gt[:, :k]
    hits = 0
    for i in range(len(queries)):
        hits += len(set(labels[i]) & set(gt_k[i]))
    return hits / (len(queries) * k)


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    existing = load_results()
    existing.update(results)
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def run_vanilla(data, queries, gt, metric):
    """Vanilla single-threaded baseline."""
    n, dim = data.shape
    idx = hnswlib.Index(space=metric, dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    t0 = time.time()
    idx.add_items(data, np.arange(n), num_threads=1)
    build_time = time.time() - t0

    recalls = {}
    for ef in EF_SEARCH_VALUES:
        for k in K_VALUES:
            recalls[f"R@{k}_ef{ef}"] = compute_recall(idx, queries, gt, k, ef)

    return {
        "build_time": build_time,
        "recalls": recalls,
    }


def run_alpha_pruning(data, queries, gt, metric, alpha, data_proj):
    """α-pruning baseline at given alpha, co-located mode."""
    n, dim = data.shape
    idx = hnswlib.Index(space=metric, dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    idx.enable_projection_storage(PROJ_DIM)
    idx.create_alpha_pruning_functor(alpha=alpha, proj_dim=PROJ_DIM)
    idx.activate_alpha_pruning_skip()

    t0 = time.time()
    idx.add_items_with_alpha_skip(data, data_proj)
    build_time = time.time() - t0

    metrics = idx.get_construction_metrics()
    total = metrics["distance_computations"] + metrics["skipped_computations"]
    skip_rate = metrics["skipped_computations"] / total if total > 0 else 0

    idx.deactivate_any_skip()

    recalls = {}
    for ef in EF_SEARCH_VALUES:
        for k in K_VALUES:
            recalls[f"R@{k}_ef{ef}"] = compute_recall(idx, queries, gt, k, ef)

    return {
        "alpha": alpha,
        "build_time": build_time,
        "distance_computations": metrics["distance_computations"],
        "skipped_computations": metrics["skipped_computations"],
        "skip_rate": skip_rate,
        "recalls": recalls,
    }


def run_learned_tree(data, queries, gt, metric, threshold, data_proj):
    """Learned tree skip (co-located), at given threshold."""
    n, dim = data.shape
    idx = hnswlib.Index(space=metric, dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    idx.enable_projection_storage(PROJ_DIM)

    idx.create_skip_functor(threshold=threshold, proj_dim=PROJ_DIM)
    idx.set_skip_projected_data(data_proj)
    idx.set_skip_tree_params(
        rank_thresholds=DEFAULT_TREE_PARAMS["rank_thresholds"],
        adist_thresholds=DEFAULT_TREE_PARAMS["adist_thresholds"],
        ins_t0=DEFAULT_TREE_PARAMS["ins_t0"],
        leaf_probas=DEFAULT_TREE_PARAMS["leaf_probas"],
    )
    idx.activate_skip()

    t0 = time.time()
    idx.add_items_with_skip(data, data_proj)
    build_time = time.time() - t0

    metrics = idx.get_construction_metrics()
    total = metrics["distance_computations"] + metrics["skipped_computations"]
    skip_rate = metrics["skipped_computations"] / total if total > 0 else 0

    idx.deactivate_skip()
    idx.clear_skip_functor()

    recalls = {}
    for ef in EF_SEARCH_VALUES:
        for k in K_VALUES:
            recalls[f"R@{k}_ef{ef}"] = compute_recall(idx, queries, gt, k, ef)

    return {
        "threshold": threshold,
        "build_time": build_time,
        "distance_computations": metrics["distance_computations"],
        "skipped_computations": metrics["skipped_computations"],
        "skip_rate": skip_rate,
        "recalls": recalls,
    }


def run_dataset(dataset_name):
    log(f"\n{'='*60}")
    log(f"Dataset: {dataset_name}")
    log(f"{'='*60}")

    ds = get_dataset(dataset_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    metric = meta.metric
    n, dim = data.shape
    log(f"n={n}, dim={dim}, metric={metric}")

    # Random projection
    rng = np.random.RandomState(SEED)
    proj_matrix = rng.randn(dim, PROJ_DIM).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)

    results = {}

    # 1. Vanilla baseline
    log("\n--- Vanilla (1-thread) ---")
    vanilla = run_vanilla(data, queries, gt, metric)
    log(f"  build_time={vanilla['build_time']:.2f}s, R@10_ef100={vanilla['recalls'].get('R@10_ef100', 0):.4f}")
    results["vanilla_1t"] = vanilla

    # 2. α-pruning at multiple values
    for alpha in ALPHA_VALUES:
        key = f"alpha_{alpha}"
        log(f"\n--- α-pruning (α={alpha}) ---")
        r = run_alpha_pruning(data, queries, gt, metric, alpha, data_proj)
        speedup = vanilla["build_time"] / r["build_time"] if r["build_time"] > 0 else 0
        log(f"  build_time={r['build_time']:.2f}s ({speedup:.2f}x), skip={r['skip_rate']:.2%}, R@10_ef100={r['recalls'].get('R@10_ef100', 0):.4f}")
        results[key] = r

    # 3. Learned tree at multiple thresholds
    for tau in TREE_THRESHOLDS:
        key = f"learned_tree_tau{tau}"
        log(f"\n--- Learned tree (τ={tau}) ---")
        r = run_learned_tree(data, queries, gt, metric, tau, data_proj)
        speedup = vanilla["build_time"] / r["build_time"] if r["build_time"] > 0 else 0
        log(f"  build_time={r['build_time']:.2f}s ({speedup:.2f}x), skip={r['skip_rate']:.2%}, R@10_ef100={r['recalls'].get('R@10_ef100', 0):.4f}")
        results[key] = r

    save_results({dataset_name: results})
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "sift1m"
    run_dataset(dataset)
