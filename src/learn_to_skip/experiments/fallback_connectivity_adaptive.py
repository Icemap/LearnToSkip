"""Round 7 experiments addressing remaining reviewer feedback.

1. Fallback validation with overhead quantification (Reviewer Q1)
2. Connectivity diagnostics (Reviewer Q6)
3. Adaptive threshold calibration (Reviewer Q5)

All experiments run on SIFT1M (Mac M1 Pro). Run sequentially to avoid contention.
Results saved to results/fallback_connectivity_adaptive.json
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


def run_fallback_validation(ds_name="sift1m"):
    """Experiment 1: Fallback validation with overhead quantification.

    Simulates the auto-fallback mechanism:
    - Build in batches of 50K
    - Monitor skip rate per batch via construction metrics
    - When skip rate drops below threshold, retrain from recent traces
    - Measure: monitoring overhead, retrain overhead, total wall-clock
    """
    log(f"\n{'='*60}")
    log(f"  Fallback Validation on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    n, dim = data.shape
    space = "l2" if meta.metric == "l2" else "cosine"

    # Projection setup
    rng = np.random.RandomState(SEED)
    proj_matrix = rng.randn(dim, 16).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)

    BATCH_SIZE = 50_000
    TAU = 0.8
    MIN_SKIP_RATE = 0.01  # Trigger retrain if skip rate < 1%
    RETRAIN_SAMPLE = 10_000  # Retrain on last 10K insertions

    # Vanilla baseline for comparison
    log("  Building vanilla baseline...")
    vanilla_idx, vanilla_time = build_vanilla(data, M, EF_C, meta.metric)
    vanilla_r10 = compute_recall(vanilla_idx, queries, gt, 10, 200)
    log(f"  Vanilla: {vanilla_time:.1f}s, R@10={vanilla_r10:.4f}")

    # Build with fallback monitoring
    log(f"\n  Building with fallback monitoring (batch={BATCH_SIZE}, tau={TAU})...")
    idx = hnswlib.Index(space=space, dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=EF_C, random_seed=SEED)

    # Initialize skip functor with universal tree
    idx.create_skip_functor(threshold=TAU, proj_dim=16)
    idx.set_skip_projected_data(data_proj)
    idx.set_skip_tree_params(
        rank_thresholds=DEFAULT_TREE_PARAMS["rank_thresholds"],
        adist_thresholds=DEFAULT_TREE_PARAMS["adist_thresholds"],
        ins_t0=DEFAULT_TREE_PARAMS["ins_t0"],
        leaf_probas=DEFAULT_TREE_PARAMS["leaf_probas"],
    )
    idx.activate_skip()

    total_build_time = 0.0
    total_monitor_time = 0.0
    total_retrain_time = 0.0
    n_retrains = 0
    batch_stats = []

    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n)
        batch_data = data[start:end]
        batch_proj = data_proj[start:end]
        batch_ids = np.arange(start, end)

        # Reset metrics before this batch
        idx.reset_construction_metrics()

        # Build this batch
        t0 = time.time()
        idx.add_items_with_skip(batch_data, batch_proj, ids=batch_ids)
        batch_build_time = time.time() - t0
        total_build_time += batch_build_time

        # Monitor: check skip rate for this batch
        t_monitor = time.time()
        metrics = idx.get_construction_metrics()
        total_evals = metrics["distance_computations"] + metrics["skipped_computations"]
        batch_skip_rate = metrics["skipped_computations"] / max(1, total_evals)
        monitor_overhead = time.time() - t_monitor
        total_monitor_time += monitor_overhead

        log(f"    Batch {batch_idx+1}/{n_batches} [{start}:{end}]: "
            f"build={batch_build_time:.1f}s, skip_rate={batch_skip_rate:.3f}, "
            f"monitor={monitor_overhead*1000:.2f}ms")

        batch_info = {
            "batch": batch_idx,
            "range": [start, end],
            "build_time": batch_build_time,
            "skip_rate": batch_skip_rate,
            "monitor_time_ms": monitor_overhead * 1000,
            "retrained": False,
        }

        # Check if retrain needed
        if batch_skip_rate < MIN_SKIP_RATE and batch_idx < n_batches - 1:
            log(f"    ** Skip rate {batch_skip_rate:.3f} < {MIN_SKIP_RATE}, triggering retrain...")
            t_retrain = time.time()

            # Retrain from recent insertions
            retrain_start = max(0, end - RETRAIN_SAMPLE)
            retrain_data = data[retrain_start:end]
            retrain_proj = data_proj[retrain_start:end]

            try:
                from learn_to_skip.trace.tracer import HNSWTracer
                from sklearn.tree import DecisionTreeClassifier

                # Build small index to get traces
                tracer = HNSWTracer(dim=dim, M=M, ef_construction=EF_C)
                tracer.build(retrain_data)
                df = tracer.get_trace_df()

                if len(df) >= 100:
                    features = ["candidate_rank_in_beam", "approx_dist", "inserted_count"]
                    X = df[features].values
                    y = df["is_wasted"].values
                    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED, class_weight="balanced")
                    tree.fit(X, y)

                    # Extract params and update functor
                    builder = CppLearnedSkipBuilder(threshold=TAU, proj_dim=16)
                    new_params = builder._extract_denormalized_params(tree, features)
                    if new_params is not None:
                        idx.set_skip_tree_params(**new_params)
                        log(f"    ** Retrain successful, new tree params applied")
                    else:
                        log(f"    ** Retrain: tree extraction failed, keeping old params")
                else:
                    log(f"    ** Retrain: insufficient trace data ({len(df)} rows)")
            except Exception as e:
                log(f"    ** Retrain failed: {e}")

            retrain_overhead = time.time() - t_retrain
            total_retrain_time += retrain_overhead
            n_retrains += 1
            batch_info["retrained"] = True
            batch_info["retrain_time"] = retrain_overhead
            log(f"    ** Retrain overhead: {retrain_overhead:.2f}s")

        batch_stats.append(batch_info)

    # Final recall
    idx.deactivate_skip()
    r10 = compute_recall(idx, queries, gt, 10, 200)

    total_wall_clock = total_build_time + total_monitor_time + total_retrain_time
    log(f"\n  Summary:")
    log(f"    Build time:    {total_build_time:.1f}s")
    log(f"    Monitor time:  {total_monitor_time*1000:.2f}ms ({total_monitor_time/total_wall_clock*100:.3f}% of total)")
    log(f"    Retrain time:  {total_retrain_time:.2f}s ({n_retrains} retrains)")
    log(f"    Total:         {total_wall_clock:.1f}s")
    log(f"    Vanilla:       {vanilla_time:.1f}s")
    log(f"    Speedup:       {vanilla_time/total_wall_clock:.2f}x")
    log(f"    R@10:          {r10:.4f} (vanilla: {vanilla_r10:.4f})")

    return {
        "vanilla_time": vanilla_time,
        "vanilla_r10": vanilla_r10,
        "build_time": total_build_time,
        "monitor_time_ms": total_monitor_time * 1000,
        "retrain_time": total_retrain_time,
        "num_retrains": n_retrains,
        "total_wall_clock": total_wall_clock,
        "speedup": vanilla_time / total_wall_clock,
        "recall_10": r10,
        "tau": TAU,
        "batch_size": BATCH_SIZE,
        "batch_stats": batch_stats,
    }


def run_connectivity_diagnostics(ds_name="sift1m"):
    """Experiment 2: Connectivity diagnostics — connected components + greedy reachability.

    For vanilla and skip at tau={0.5, 0.7, 0.8}:
    - Number of connected components at level 0
    - Size of largest connected component
    - Average greedy search path length (navigation diameter proxy)
    - Max greedy search path length
    """
    log(f"\n{'='*60}")
    log(f"  Connectivity Diagnostics on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    n, dim = data.shape

    results = {}

    # Vanilla
    log("  Building vanilla...")
    idx, bt = build_vanilla(data, M, EF_C, meta.metric)
    r10 = compute_recall(idx, queries, gt, 10, 200)

    log("  Computing connectivity...")
    t0 = time.time()
    cc = idx.get_connected_components()
    cc_time = time.time() - t0

    log("  Sampling greedy reachability (1000 samples)...")
    t0 = time.time()
    reach = idx.sample_greedy_reachability(num_samples=1000, seed=42)
    reach_time = time.time() - t0

    results["vanilla"] = {
        "build_time": bt,
        "recall_10": r10,
        "num_components": cc["num_components"],
        "largest_component": cc["largest_component_size"],
        "largest_component_pct": cc["largest_component_size"] / n * 100,
        "top_component_sizes": cc["top_component_sizes"],
        "avg_path_length": reach["avg_path_length"],
        "max_path_length": reach["max_path_length"],
        "cc_analysis_time": cc_time,
        "reach_analysis_time": reach_time,
    }
    log(f"    Components: {cc['num_components']}, Largest: {cc['largest_component_size']} ({cc['largest_component_size']/n*100:.1f}%)")
    log(f"    Greedy path: avg={reach['avg_path_length']:.2f}, max={reach['max_path_length']:.0f}")
    log(f"    Analysis time: CC={cc_time:.2f}s, Reach={reach_time:.2f}s")

    # Skip at various tau
    for tau in [0.5, 0.7, 0.8]:
        log(f"  Building skip tau={tau}...")
        builder = CppLearnedSkipBuilder(threshold=tau, proj_dim=16, mode="universal", seed=SEED)
        result = builder.build(data, M=M, ef_construction=EF_C, metric=meta.metric)
        r10 = compute_recall(result.index, queries, gt, 10, 200)

        log(f"  Computing connectivity for tau={tau}...")
        cc = result.index.get_connected_components()
        reach = result.index.sample_greedy_reachability(num_samples=1000, seed=42)

        results[f"tau_{tau}"] = {
            "build_time": result.build_time_seconds,
            "recall_10": r10,
            "num_components": cc["num_components"],
            "largest_component": cc["largest_component_size"],
            "largest_component_pct": cc["largest_component_size"] / n * 100,
            "top_component_sizes": cc["top_component_sizes"],
            "avg_path_length": reach["avg_path_length"],
            "max_path_length": reach["max_path_length"],
        }
        log(f"    Components: {cc['num_components']}, Largest: {cc['largest_component_size']} ({cc['largest_component_size']/n*100:.1f}%)")
        log(f"    Greedy path: avg={reach['avg_path_length']:.2f}, max={reach['max_path_length']:.0f}")

    return results


def run_adaptive_threshold(ds_name="sift1m"):
    """Experiment 3: Adaptive threshold calibration.

    Instead of fixed tau, calibrate tau to target a specific skip rate.
    Uses a binary search over tau in [0.3, 0.95] on a small probe batch
    to find the tau that achieves a target skip rate.

    Compares:
    - Fixed tau=0.8 (may over/under-skip on different datasets)
    - Adaptive tau targeting 60% skip rate (balanced speed/quality)
    - Adaptive tau targeting 80% skip rate (aggressive)
    """
    log(f"\n{'='*60}")
    log(f"  Adaptive Threshold Calibration on {ds_name}")
    log(f"{'='*60}")

    ds = get_dataset(ds_name)
    meta = ds.metadata()
    data = ds.load_train()
    queries = ds.load_query()
    gt = ds.load_groundtruth(k=100)
    n, dim = data.shape
    space = "l2" if meta.metric == "l2" else "cosine"

    # Projection
    rng_proj = np.random.RandomState(SEED)
    proj_matrix = rng_proj.randn(dim, 16).astype(np.float32)
    proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
    data_proj = (data @ proj_matrix).astype(np.float32)

    # Vanilla baseline
    log("  Building vanilla baseline...")
    vanilla_idx, vanilla_time = build_vanilla(data, M, EF_C, meta.metric)
    vanilla_r10 = compute_recall(vanilla_idx, queries, gt, 10, 200)
    log(f"  Vanilla: {vanilla_time:.1f}s, R@10={vanilla_r10:.4f}")

    results = {"vanilla_time": vanilla_time, "vanilla_r10": vanilla_r10}

    def probe_skip_rate(tau, probe_data, probe_proj, probe_ids):
        """Build a small probe to measure skip rate at given tau."""
        probe_n = len(probe_data)
        idx = hnswlib.Index(space=space, dim=dim)
        idx.init_index(max_elements=probe_n, M=M, ef_construction=EF_C, random_seed=SEED)
        idx.create_skip_functor(threshold=tau, proj_dim=16)
        idx.set_skip_projected_data(probe_proj)
        idx.set_skip_tree_params(
            rank_thresholds=DEFAULT_TREE_PARAMS["rank_thresholds"],
            adist_thresholds=DEFAULT_TREE_PARAMS["adist_thresholds"],
            ins_t0=DEFAULT_TREE_PARAMS["ins_t0"],
            leaf_probas=DEFAULT_TREE_PARAMS["leaf_probas"],
        )
        idx.activate_skip()
        idx.add_items_with_skip(probe_data, probe_proj, ids=probe_ids)
        m = idx.get_construction_metrics()
        total = m["distance_computations"] + m["skipped_computations"]
        return m["skipped_computations"] / max(1, total)

    def calibrate_tau(target_skip_rate, probe_data, probe_proj, probe_ids, tol=0.02):
        """Binary search for tau that achieves target skip rate."""
        lo, hi = 0.1, 0.95
        best_tau = 0.7
        best_diff = float('inf')
        steps = 0

        for _ in range(12):  # Max 12 iterations of binary search
            mid = (lo + hi) / 2
            sr = probe_skip_rate(mid, probe_data, probe_proj, probe_ids)
            diff = abs(sr - target_skip_rate)
            steps += 1

            if diff < best_diff:
                best_diff = diff
                best_tau = mid

            if abs(sr - target_skip_rate) < tol:
                break

            if sr > target_skip_rate:
                lo = mid  # tau too low (too aggressive), increase it
            else:
                hi = mid  # tau too high (too conservative), decrease it

        return best_tau, steps

    # Use first 50K as calibration probe
    PROBE_N = 50_000
    probe_data = data[:PROBE_N]
    probe_proj = data_proj[:PROBE_N]
    probe_ids = np.arange(PROBE_N)

    # Calibrate for different target skip rates
    for target_sr, label in [(0.60, "balanced"), (0.80, "aggressive")]:
        log(f"\n  Calibrating for {target_sr*100:.0f}% skip rate ({label})...")
        t_cal = time.time()
        cal_tau, cal_steps = calibrate_tau(target_sr, probe_data, probe_proj, probe_ids)
        cal_time = time.time() - t_cal
        log(f"    Calibrated tau={cal_tau:.4f} in {cal_steps} steps ({cal_time:.1f}s)")

        # Full build with calibrated tau
        log(f"    Full build with tau={cal_tau:.4f}...")
        builder = CppLearnedSkipBuilder(threshold=cal_tau, proj_dim=16, mode="universal", seed=SEED)
        result = builder.build(data, M=M, ef_construction=EF_C, metric=meta.metric)
        r10 = compute_recall(result.index, queries, gt, 10, 200)
        speedup = vanilla_time / result.build_time_seconds
        log(f"    Build: {result.build_time_seconds:.1f}s, speedup={speedup:.2f}x, "
            f"skip={result.skip_rate:.3f}, R@10={r10:.4f}")

        results[f"adaptive_{label}"] = {
            "target_skip_rate": target_sr,
            "calibrated_tau": cal_tau,
            "calibration_steps": cal_steps,
            "calibration_time": cal_time,
            "build_time": result.build_time_seconds,
            "speedup": speedup,
            "skip_rate": result.skip_rate,
            "recall_10": r10,
            "total_time_with_calibration": result.build_time_seconds + cal_time,
            "speedup_with_calibration": vanilla_time / (result.build_time_seconds + cal_time),
        }

    # Also test fixed tau=0.8 for comparison
    log(f"\n  Fixed tau=0.8 for comparison...")
    builder = CppLearnedSkipBuilder(threshold=0.8, proj_dim=16, mode="universal", seed=SEED)
    result = builder.build(data, M=M, ef_construction=EF_C, metric=meta.metric)
    r10 = compute_recall(result.index, queries, gt, 10, 200)
    speedup = vanilla_time / result.build_time_seconds
    log(f"    Build: {result.build_time_seconds:.1f}s, speedup={speedup:.2f}x, "
        f"skip={result.skip_rate:.3f}, R@10={r10:.4f}")
    results["fixed_tau_0.8"] = {
        "tau": 0.8,
        "build_time": result.build_time_seconds,
        "speedup": speedup,
        "skip_rate": result.skip_rate,
        "recall_10": r10,
    }

    # Also test on Deep1M where universal tree is known to struggle
    log(f"\n  Testing adaptive threshold on Deep1M...")
    try:
        ds2 = get_dataset("deep1m")
        meta2 = ds2.metadata()
        data2 = ds2.load_train()
        queries2 = ds2.load_query()
        gt2 = ds2.load_groundtruth(k=100)
        n2, dim2 = data2.shape

        rng2 = np.random.RandomState(SEED)
        proj2 = rng2.randn(dim2, 16).astype(np.float32)
        proj2 /= np.linalg.norm(proj2, axis=0, keepdims=True)
        data2_proj = (data2 @ proj2).astype(np.float32)
        space2 = "cosine"

        log("  Deep1M vanilla...")
        v_idx2, v_time2 = build_vanilla(data2, M, EF_C, meta2.metric)
        v_r10_2 = compute_recall(v_idx2, queries2, gt2, 10, 200)
        log(f"    Vanilla: {v_time2:.1f}s, R@10={v_r10_2:.4f}")

        # Fixed tau=0.8 on Deep1M
        log("  Deep1M fixed tau=0.8...")
        builder2 = CppLearnedSkipBuilder(threshold=0.8, proj_dim=16, mode="universal", seed=SEED)
        r2 = builder2.build(data2, M=M, ef_construction=EF_C, metric=meta2.metric)
        r10_2 = compute_recall(r2.index, queries2, gt2, 10, 200)
        log(f"    Fixed 0.8: {r2.build_time_seconds:.1f}s, skip={r2.skip_rate:.3f}, R@10={r10_2:.4f}")

        results["deep1m_fixed_0.8"] = {
            "build_time": r2.build_time_seconds,
            "speedup": v_time2 / r2.build_time_seconds,
            "skip_rate": r2.skip_rate,
            "recall_10": r10_2,
            "vanilla_time": v_time2,
            "vanilla_r10": v_r10_2,
        }

        # Adaptive on Deep1M
        log("  Deep1M adaptive calibration (target 60%)...")
        probe2 = data2[:PROBE_N]
        probe2_proj = data2_proj[:PROBE_N]
        probe2_ids = np.arange(PROBE_N)

        # Need to use Deep1M's space for probing
        def probe_skip_rate_deep(tau):
            idx = hnswlib.Index(space=space2, dim=dim2)
            idx.init_index(max_elements=PROBE_N, M=M, ef_construction=EF_C, random_seed=SEED)
            idx.create_skip_functor(threshold=tau, proj_dim=16)
            idx.set_skip_projected_data(probe2_proj)
            idx.set_skip_tree_params(
                rank_thresholds=DEFAULT_TREE_PARAMS["rank_thresholds"],
                adist_thresholds=DEFAULT_TREE_PARAMS["adist_thresholds"],
                ins_t0=DEFAULT_TREE_PARAMS["ins_t0"],
                leaf_probas=DEFAULT_TREE_PARAMS["leaf_probas"],
            )
            idx.activate_skip()
            idx.add_items_with_skip(probe2, probe2_proj, ids=probe2_ids)
            m = idx.get_construction_metrics()
            total = m["distance_computations"] + m["skipped_computations"]
            return m["skipped_computations"] / max(1, total)

        # Binary search
        lo, hi = 0.1, 0.95
        target = 0.60
        best_tau, best_diff = 0.7, float('inf')
        t_cal2 = time.time()
        for _ in range(12):
            mid = (lo + hi) / 2
            sr = probe_skip_rate_deep(mid)
            if abs(sr - target) < best_diff:
                best_diff = abs(sr - target)
                best_tau = mid
            if abs(sr - target) < 0.02:
                break
            if sr > target:
                lo = mid
            else:
                hi = mid
        cal_time2 = time.time() - t_cal2
        log(f"    Calibrated tau={best_tau:.4f} ({cal_time2:.1f}s)")

        builder2a = CppLearnedSkipBuilder(threshold=best_tau, proj_dim=16, mode="universal", seed=SEED)
        r2a = builder2a.build(data2, M=M, ef_construction=EF_C, metric=meta2.metric)
        r10_2a = compute_recall(r2a.index, queries2, gt2, 10, 200)
        log(f"    Adaptive: {r2a.build_time_seconds:.1f}s, skip={r2a.skip_rate:.3f}, R@10={r10_2a:.4f}")

        results["deep1m_adaptive"] = {
            "calibrated_tau": best_tau,
            "calibration_time": cal_time2,
            "build_time": r2a.build_time_seconds,
            "speedup": v_time2 / r2a.build_time_seconds,
            "skip_rate": r2a.skip_rate,
            "recall_10": r10_2a,
        }

    except Exception as e:
        log(f"  Deep1M error: {e}")
        import traceback
        traceback.print_exc()

    return results


def main():
    experiments = sys.argv[1:] if len(sys.argv) > 1 else [
        "fallback", "connectivity", "adaptive"
    ]

    all_results = {}

    for exp in experiments:
        try:
            if exp == "fallback":
                all_results["fallback_validation"] = run_fallback_validation("sift1m")
            elif exp == "connectivity":
                all_results["connectivity"] = run_connectivity_diagnostics("sift1m")
            elif exp == "adaptive":
                all_results["adaptive_threshold"] = run_adaptive_threshold("sift1m")
            else:
                log(f"Unknown experiment: {exp}")
        except Exception as e:
            log(f"ERROR in {exp}: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp] = {"error": str(e)}

    output_path = RESULTS_DIR / "fallback_connectivity_adaptive.json"

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
