"""Comprehensive million-scale experiments for paper revision.

Runs C++ hnswlib fork benchmarks on SIFT1M, Deep1M, etc.
Collects: wall-clock time, skip rate, recall@1/10/100, ef_search sweep,
vanilla multi-threaded baseline, reduced ef_c baseline.

Results saved to results/{dataset}_results.json (one file per dataset).
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
    """Print with flush so background processes show output immediately."""
    print(msg, flush=True)

M = 16
EF_C = 200
THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 0.9]
K_VALUES = [1, 10, 100]
EF_SEARCH_VALUES = [10, 50, 100, 200, 400]
N_TRIALS = 1  # Set to 3 for final paper numbers
DATASET_NAMES = ["sift1m", "gist1m", "glove200", "deep1m"]


def compute_recall(index, queries, gt, k, ef_search):
    """Compute recall@k with given ef_search."""
    index.set_ef(ef_search)
    labels, _ = index.knn_query(queries, k=k)
    gt_k = gt[:, :k]
    hits = 0
    for i in range(len(queries)):
        hits += len(set(labels[i]) & set(gt_k[i]))
    return hits / (len(queries) * k)


def build_vanilla(data, M, ef_c, metric, num_threads=1):
    """Build vanilla HNSW index, return (index, build_time)."""
    n, dim = data.shape
    space = "l2" if metric == "l2" else "cosine"
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, M=M, ef_construction=ef_c, random_seed=SEED)
    # IMPORTANT: set_num_threads BEFORE add_items for fair comparison
    index.set_num_threads(num_threads)
    t0 = time.time()
    index.add_items(data, np.arange(n), num_threads=num_threads)
    build_time = time.time() - t0
    return index, build_time


def build_skip(data, M, ef_c, metric, threshold, mode="universal"):
    """Build skip index, return (index, build_time, metrics)."""
    builder = CppLearnedSkipBuilder(
        threshold=threshold, proj_dim=16, mode=mode, seed=SEED
    )
    result = builder.build(data, M=M, ef_construction=ef_c, metric=metric)
    return result.index, result.build_time_seconds, {
        "distance_computations": result.distance_computations,
        "skipped_computations": result.skipped_computations,
        "skip_rate": result.skip_rate,
    }


def count_waste_ratio(data, M, ef_c, metric):
    """Build vanilla index with instrumentation to count waste.
    Use construction metrics from C++ (skip with tau=1.0 => no skip, but counts distances)."""
    n, dim = data.shape
    # Build with skip active but tau=0.99 so almost nothing skipped
    # Use the metrics counters to get total distance computations
    rng = np.random.RandomState(SEED)
    proj_matrix = rng.randn(dim, 16).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)

    space = "l2" if metric == "l2" else "cosine"
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, M=M, ef_construction=ef_c, random_seed=SEED)
    index.create_skip_functor(threshold=0.999, proj_dim=16)
    index.set_skip_projected_data(data_proj)
    index.set_skip_tree_params(
        rank_thresholds=DEFAULT_TREE_PARAMS["rank_thresholds"],
        adist_thresholds=DEFAULT_TREE_PARAMS["adist_thresholds"],
        ins_t0=DEFAULT_TREE_PARAMS["ins_t0"],
        leaf_probas=DEFAULT_TREE_PARAMS["leaf_probas"],
    )
    index.activate_skip()
    index.add_items_with_skip(data, data_proj)
    metrics = index.get_construction_metrics()
    index.deactivate_skip()
    index.clear_skip_functor()

    total_evals = metrics["distance_computations"] + metrics["skipped_computations"]
    # Approximate retained = n * 2 * M (each point gets ~2M edges)
    retained_approx = n * 2 * M
    waste_ratio = 1.0 - (retained_approx / total_evals) if total_evals > 0 else 0.0
    return {
        "total_evaluations": total_evals,
        "distance_computations": metrics["distance_computations"],
        "retained_approx": retained_approx,
        "waste_ratio_approx": waste_ratio,
    }


def run_dataset_experiments(ds_name):
    """Run all experiments for one dataset."""
    log(f"\n{'='*60}")
    log(f"  Dataset: {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    log(f"  Loading {meta.n_train} vectors, dim={meta.dim}, metric={meta.metric}")

    data = ds.load_train()
    queries = ds.load_query()
    log(f"  Data shape: {data.shape}, Queries: {queries.shape}")

    # Compute ground truth
    log("  Computing ground truth (k=100)...")
    gt = ds.load_groundtruth(k=100)
    log(f"  Ground truth shape: {gt.shape}")

    results = {
        "dataset": ds_name,
        "n": meta.n_train,
        "dim": meta.dim,
        "metric": meta.metric,
        "M": M,
        "ef_construction": EF_C,
    }

    # 1. Waste ratio
    log("  [1/5] Measuring waste ratio...")
    waste = count_waste_ratio(data, M, EF_C, meta.metric)
    results["waste"] = waste
    log(f"    Waste ratio: {waste['waste_ratio_approx']:.4f}")

    # 2. Vanilla baselines (1-thread and multi-thread)
    log("  [2/5] Building vanilla baselines...")
    vanilla_times_1t = []
    vanilla_times_mt = []
    vanilla_index = None
    for trial in range(N_TRIALS):
        idx, t = build_vanilla(data, M, EF_C, meta.metric, num_threads=1)
        vanilla_times_1t.append(t)
        if trial == 0:
            vanilla_index = idx
        log(f"    Vanilla 1T trial {trial+1}: {t:.2f}s")

    for trial in range(N_TRIALS):
        idx, t = build_vanilla(data, M, EF_C, meta.metric, num_threads=10)
        vanilla_times_mt.append(t)
        log(f"    Vanilla 10T trial {trial+1}: {t:.2f}s")

    vanilla_1t = np.median(vanilla_times_1t)
    vanilla_mt = np.median(vanilla_times_mt)

    # Vanilla recall sweep
    vanilla_recalls = {}
    for ef_s in EF_SEARCH_VALUES:
        for k in K_VALUES:
            r = compute_recall(vanilla_index, queries, gt, k, ef_s)
            vanilla_recalls[f"recall@{k}_ef{ef_s}"] = r

    results["vanilla"] = {
        "time_1t": vanilla_1t,
        "time_mt": vanilla_mt,
        "times_1t": vanilla_times_1t,
        "times_mt": vanilla_times_mt,
        "recalls": vanilla_recalls,
    }
    log(f"    Vanilla 1T median: {vanilla_1t:.2f}s, 10T median: {vanilla_mt:.2f}s")
    log(f"    Vanilla recall@10 (ef=200): {vanilla_recalls.get('recall@10_ef200', 'N/A'):.4f}")

    # 3. LearnToSkip (universal mode) at various thresholds
    log("  [3/5] Building LearnToSkip (universal) at various thresholds...")
    skip_results = {}
    for tau in THRESHOLDS:
        log(f"    tau={tau}:")
        times = []
        skip_index = None
        skip_metrics = None
        for trial in range(N_TRIALS):
            idx, t, m = build_skip(data, M, EF_C, meta.metric, threshold=tau)
            times.append(t)
            if trial == 0:
                skip_index = idx
                skip_metrics = m
            log(f"      trial {trial+1}: {t:.2f}s, skip_rate={m['skip_rate']:.4f}")

        med_time = np.median(times)

        # Recall sweep
        recalls = {}
        for ef_s in EF_SEARCH_VALUES:
            for k in K_VALUES:
                r = compute_recall(skip_index, queries, gt, k, ef_s)
                recalls[f"recall@{k}_ef{ef_s}"] = r

        skip_results[str(tau)] = {
            "time": med_time,
            "times": times,
            "skip_rate": skip_metrics["skip_rate"],
            "distance_computations": skip_metrics["distance_computations"],
            "skipped_computations": skip_metrics["skipped_computations"],
            "speedup_1t": vanilla_1t / med_time,
            "speedup_mt": vanilla_mt / med_time,
            "recalls": recalls,
        }
        log(f"      median: {med_time:.2f}s, speedup_1t: {vanilla_1t/med_time:.2f}x, "
              f"recall@10(ef200): {recalls.get('recall@10_ef200', 'N/A'):.4f}")

    results["skip_universal"] = skip_results

    # 4. LearnToSkip (online mode, 20% training)
    log("  [4/5] Building LearnToSkip (online, p=20%)...")
    online_results = {}
    for tau in [0.5, 0.7, 0.8]:
        log(f"    tau={tau}:")
        times = []
        online_index = None
        online_metrics = None
        for trial in range(N_TRIALS):
            idx, t, m = build_skip(data, M, EF_C, meta.metric, threshold=tau, mode="online")
            times.append(t)
            if trial == 0:
                online_index = idx
                online_metrics = m
            log(f"      trial {trial+1}: {t:.2f}s")

        med_time = np.median(times)
        recalls = {}
        for ef_s in [100, 200, 400]:
            for k in K_VALUES:
                r = compute_recall(online_index, queries, gt, k, ef_s)
                recalls[f"recall@{k}_ef{ef_s}"] = r

        online_results[str(tau)] = {
            "time": med_time,
            "times": times,
            "skip_rate": online_metrics["skip_rate"],
            "speedup_1t": vanilla_1t / med_time,
            "recalls": recalls,
        }
        log(f"      median: {med_time:.2f}s, speedup: {vanilla_1t/med_time:.2f}x")

    results["skip_online"] = online_results

    # 5. Baseline: reduced ef_construction to match recall
    log("  [5/5] Building reduced ef_c baselines...")
    reduced_ef_results = {}
    for ef_c_red in [50, 100, 150]:
        times = []
        red_index = None
        for trial in range(N_TRIALS):
            idx, t = build_vanilla(data, M, ef_c_red, meta.metric, num_threads=1)
            times.append(t)
            if trial == 0:
                red_index = idx
            log(f"    ef_c={ef_c_red} trial {trial+1}: {t:.2f}s")

        med_time = np.median(times)
        recalls = {}
        for ef_s in [100, 200, 400]:
            for k in K_VALUES:
                r = compute_recall(red_index, queries, gt, k, ef_s)
                recalls[f"recall@{k}_ef{ef_s}"] = r

        reduced_ef_results[str(ef_c_red)] = {
            "time": med_time,
            "times": times,
            "speedup_1t": vanilla_1t / med_time,
            "recalls": recalls,
        }
        log(f"    ef_c={ef_c_red}: {med_time:.2f}s, speedup: {vanilla_1t/med_time:.2f}x, "
              f"recall@10(ef200): {recalls.get('recall@10_ef200', 'N/A'):.4f}")

    results["reduced_ef_c"] = reduced_ef_results

    return results


def main():
    # Check which datasets to run (allow CLI filtering)
    datasets_to_run = sys.argv[1:] if len(sys.argv) > 1 else DATASET_NAMES

    for ds_name in datasets_to_run:
        if ds_name not in DATASET_NAMES:
            log(f"Unknown dataset: {ds_name}, skipping")
            continue

        output_path = RESULTS_DIR / f"{ds_name}_results.json"
        try:
            result = run_dataset_experiments(ds_name)
        except Exception as e:
            log(f"  ERROR on {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            result = {"error": str(e)}

        # Save per-dataset results (never overwrites other datasets)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"\n  Results saved to {output_path}")

    log(f"\nAll experiments complete.")


if __name__ == "__main__":
    main()
