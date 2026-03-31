"""Round 8 experiments addressing remaining reviewer feedback.

1. Visited-marking controlled experiment (Q1): mark-before vs mark-after
2. M/ef_c grid sensitivity (Q4): M in {8,16,32}, ef_c in {100,200,400}
3. GloVe inner-product experiment (Q6): metric generality

All run on Mac M1 Pro. Sequential to avoid contention.
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


def log(msg):
    print(msg, flush=True)


def compute_recall(index, queries, gt, k, ef_search):
    index.set_ef(ef_search)
    labels, _ = index.knn_query(queries, k=k)
    gt_k = gt[:, :k]
    hits = sum(len(set(labels[i]) & set(gt_k[i])) for i in range(len(queries)))
    return hits / (len(queries) * k)


def run_visited_marking_experiment(ds_name="sift1m"):
    """Experiment 1: Controlled visited-marking comparison.

    Compare:
    - Default (mark-before): visited set before skip check (current behavior)
    - Deferred (mark-after): only mark visited for non-skipped candidates

    Deferred allows re-encounter of skipped candidates via alternate paths.
    """
    log(f"\n{'='*60}")
    log(f"  Visited Marking Experiment on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    n, dim = data.shape
    space = "l2" if meta.metric == "l2" else "cosine"

    M, EF_C = 16, 200

    # Projection
    rng = np.random.RandomState(SEED)
    proj_matrix = rng.randn(dim, 16).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)

    results = {}

    for tau in [0.5, 0.7, 0.8]:
        for mode, defer in [("mark_before", False), ("mark_after", True)]:
            log(f"  tau={tau}, {mode}...")
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
            idx.set_defer_visited_marking(defer)

            t0 = time.time()
            idx.add_items_with_skip(data, data_proj, np.arange(n))
            bt = time.time() - t0

            m = idx.get_construction_metrics()
            total = m["distance_computations"] + m["skipped_computations"]
            skip_rate = m["skipped_computations"] / max(1, total)
            dist_comps = m["distance_computations"]

            idx.deactivate_skip()
            r10 = compute_recall(idx, queries, gt, 10, 200)
            r1 = compute_recall(idx, queries, gt, 1, 200)
            r100 = compute_recall(idx, queries, gt, 100, 200)

            # Connectivity
            cc = idx.get_connected_components()
            reach = idx.sample_greedy_reachability(num_samples=500, seed=42)
            edges = idx.get_edge_count_level0()

            key = f"tau{tau}_{mode}"
            results[key] = {
                "tau": tau, "mode": mode, "defer": defer,
                "build_time": bt, "skip_rate": skip_rate,
                "dist_comps": dist_comps,
                "recall_1": r1, "recall_10": r10, "recall_100": r100,
                "edges": edges,
                "num_components": cc["num_components"],
                "avg_path_length": reach["avg_path_length"],
            }
            log(f"    {bt:.1f}s, skip={skip_rate:.3f}, dist_comps={dist_comps:,}, "
                f"R@1={r1:.4f}, R@10={r10:.4f}, R@100={r100:.4f}, "
                f"edges={edges:,}, CC={cc['num_components']}")

    return results


def run_grid_sensitivity(ds_name="sift1m"):
    """Experiment 2: M/ef_c grid sensitivity.

    Run vanilla and skip at tau=0.7 for M in {8,16,32}, ef_c in {100,200,400}.
    """
    log(f"\n{'='*60}")
    log(f"  M/ef_c Grid Sensitivity on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    n, dim = data.shape
    space = "l2" if meta.metric == "l2" else "cosine"

    TAU = 0.7

    results = {}

    for M in [8, 16, 32]:
        for ef_c in [100, 200, 400]:
            log(f"  M={M}, ef_c={ef_c}:")

            # Vanilla
            idx_v = hnswlib.Index(space=space, dim=dim)
            idx_v.init_index(max_elements=n, M=M, ef_construction=ef_c, random_seed=SEED)
            t0 = time.time()
            idx_v.add_items(data, np.arange(n))
            vt = time.time() - t0
            vr10 = compute_recall(idx_v, queries, gt, 10, 200)

            # Skip
            builder = CppLearnedSkipBuilder(threshold=TAU, proj_dim=16, mode="universal", seed=SEED)
            result = builder.build(data, M=M, ef_construction=ef_c, metric=meta.metric)
            sr10 = compute_recall(result.index, queries, gt, 10, 200)
            speedup = vt / result.build_time_seconds

            key = f"M{M}_efc{ef_c}"
            results[key] = {
                "M": M, "ef_c": ef_c,
                "vanilla_time": vt, "vanilla_r10": vr10,
                "skip_time": result.build_time_seconds,
                "skip_r10": sr10, "skip_rate": result.skip_rate,
                "speedup": speedup,
                "recall_delta": sr10 - vr10,
            }
            log(f"    Vanilla: {vt:.1f}s, R@10={vr10:.4f}")
            log(f"    Skip:    {result.build_time_seconds:.1f}s, {speedup:.2f}x, "
                f"skip={result.skip_rate:.3f}, R@10={sr10:.4f} (delta={sr10-vr10:+.4f})")

    return results


def run_glove_experiment():
    """Experiment 3: GloVe-200 (angular/IP metric) to test metric generality."""
    log(f"\n{'='*60}")
    log(f"  GloVe-200 Experiment (angular/cosine)")
    log(f"{'='*60}")

    ds = get_dataset("glove200")
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    n, dim = data.shape
    space = "cosine"

    M, EF_C = 16, 200

    results = {}

    # Vanilla
    log("  Vanilla baseline...")
    idx_v = hnswlib.Index(space=space, dim=dim)
    idx_v.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)
    t0 = time.time()
    idx_v.add_items(data, np.arange(n))
    vt = time.time() - t0
    vr10 = compute_recall(idx_v, queries, gt, 10, 200)
    vr1 = compute_recall(idx_v, queries, gt, 1, 200)
    log(f"    {vt:.1f}s, R@1={vr1:.4f}, R@10={vr10:.4f}")
    results["vanilla"] = {"time": vt, "recall_1": vr1, "recall_10": vr10}

    # Skip at various tau
    for tau in [0.5, 0.7, 0.8]:
        log(f"  Skip tau={tau}...")
        builder = CppLearnedSkipBuilder(threshold=tau, proj_dim=16, mode="universal", seed=SEED)
        result = builder.build(data, M=M, ef_construction=EF_C, metric=meta.metric)
        r10 = compute_recall(result.index, queries, gt, 10, 200)
        r1 = compute_recall(result.index, queries, gt, 1, 200)
        speedup = vt / result.build_time_seconds
        log(f"    {result.build_time_seconds:.1f}s, speedup={speedup:.2f}x, "
            f"skip={result.skip_rate:.3f}, R@1={r1:.4f}, R@10={r10:.4f}")
        results[f"tau_{tau}"] = {
            "tau": tau, "time": result.build_time_seconds,
            "speedup": speedup, "skip_rate": result.skip_rate,
            "recall_1": r1, "recall_10": r10,
        }

    return results


def main():
    experiments = sys.argv[1:] if len(sys.argv) > 1 else [
        "visited", "grid", "glove"
    ]

    all_results = {}

    for exp in experiments:
        try:
            if exp == "visited":
                all_results["visited_marking"] = run_visited_marking_experiment("sift1m")
            elif exp == "grid":
                all_results["grid_sensitivity"] = run_grid_sensitivity("sift1m")
            elif exp == "glove":
                all_results["glove200"] = run_glove_experiment()
            else:
                log(f"Unknown experiment: {exp}")
        except Exception as e:
            log(f"ERROR in {exp}: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp] = {"error": str(e)}

    output_path = RESULTS_DIR / "visited_grid_glove.json"

    # Merge with existing results instead of overwriting
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        log(f"Loaded {len(existing)} existing experiment(s) from {output_path}")

    for key, value in all_results.items():
        if key in existing:
            log(f"WARNING: '{key}' already exists in {output_path}, skipping (use --force to overwrite)")
            if "--force" not in sys.argv:
                continue
        existing[key] = value

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    log(f"\nResults saved to {output_path} ({len(existing)} experiment(s))")


if __name__ == "__main__":
    main()
