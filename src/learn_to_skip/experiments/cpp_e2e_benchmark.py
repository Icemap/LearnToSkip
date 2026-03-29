"""E2E wall-clock benchmark: C++ vanilla vs C++ LearnToSkip (universal + online)."""
import time

import numpy as np
import pandas as pd
import hnswlib

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.builders.cpp_learned_skip import CppLearnedSkipBuilder
from learn_to_skip.config import SEED


class CppE2EBenchmark(BaseExperiment):
    name = "cpp_e2e_benchmark"

    def run(self, **kwargs):
        # Load SIFT10K
        train = np.load("data/raw/sift10k/train.npy").astype(np.float32)
        query = np.load("data/raw/sift10k/query.npy").astype(np.float32)
        gt = np.load("data/processed/sift10k/groundtruth_k100.npy")
        n, dim = train.shape
        print(f"SIFT10K: {n} points, {dim} dim, {query.shape[0]} queries")

        M = 16
        ef_construction = 200
        k = 10
        ef_search = 50

        results = []

        # === 1. C++ Vanilla ===
        print("\n=== C++ Vanilla ===")
        times_vanilla = []
        for trial in range(3):
            idx = hnswlib.Index(space="l2", dim=dim)
            idx.init_index(max_elements=n, M=M, ef_construction=ef_construction, random_seed=SEED)
            t0 = time.time()
            idx.add_items(train)
            elapsed = time.time() - t0
            times_vanilla.append(elapsed)
            print(f"  Trial {trial+1}: {elapsed:.3f}s")

        # Use last index for recall
        idx.set_ef(ef_search)
        labels_v, _ = idx.knn_query(query, k=k)
        metrics_v = idx.get_construction_metrics()
        recall_v = _compute_recall(labels_v, gt, k)
        avg_time_v = np.mean(times_vanilla)
        print(f"  Avg build time: {avg_time_v:.3f}s, Recall@{k}: {recall_v:.4f}")
        print(f"  dist_comps: {metrics_v['distance_computations']}")

        results.append({
            "method": "C++ Vanilla",
            "build_time_s": avg_time_v,
            "build_time_std": np.std(times_vanilla),
            "distance_computations": metrics_v["distance_computations"],
            "skipped_computations": 0,
            "skip_rate": 0.0,
            f"recall@{k}": recall_v,
            "speedup_vs_vanilla": 1.0,
        })

        # === 2. C++ LearnToSkip - Universal ===
        for tau in [0.5, 0.6, 0.7, 0.8]:
            print(f"\n=== C++ LearnToSkip Universal (τ={tau}) ===")
            times_skip = []
            for trial in range(3):
                builder = CppLearnedSkipBuilder(threshold=tau, proj_dim=16, mode="universal", seed=SEED)
                result = builder.build(train, M=M, ef_construction=ef_construction)
                times_skip.append(result.build_time_seconds)
                print(f"  Trial {trial+1}: {result.build_time_seconds:.3f}s, "
                      f"skip_rate={result.skip_rate:.3f}")

            # Use last result for recall
            result.index.set_ef(ef_search)
            labels_s, _ = result.index.knn_query(query, k=k)
            recall_s = _compute_recall(labels_s, gt, k)
            avg_time_s = np.mean(times_skip)
            speedup = avg_time_v / avg_time_s
            print(f"  Avg build time: {avg_time_s:.3f}s, Recall@{k}: {recall_s:.4f}, "
                  f"Speedup: {speedup:.2f}x")

            results.append({
                "method": f"C++ Universal τ={tau}",
                "build_time_s": avg_time_s,
                "build_time_std": np.std(times_skip),
                "distance_computations": result.distance_computations,
                "skipped_computations": result.skipped_computations,
                "skip_rate": result.skip_rate,
                f"recall@{k}": recall_s,
                "speedup_vs_vanilla": speedup,
            })

        # === 3. C++ LearnToSkip - Online ===
        for frac in [0.2, 0.3, 0.5]:
            print(f"\n=== C++ LearnToSkip Online (train_frac={frac}) ===")
            times_online = []
            for trial in range(3):
                builder = CppLearnedSkipBuilder(
                    threshold=0.7, proj_dim=16, mode="online",
                    train_fraction=frac, seed=SEED)
                result = builder.build(train, M=M, ef_construction=ef_construction)
                times_online.append(result.build_time_seconds)
                print(f"  Trial {trial+1}: {result.build_time_seconds:.3f}s, "
                      f"skip_rate={result.skip_rate:.3f}")

            result.index.set_ef(ef_search)
            labels_o, _ = result.index.knn_query(query, k=k)
            recall_o = _compute_recall(labels_o, gt, k)
            avg_time_o = np.mean(times_online)
            speedup_o = avg_time_v / avg_time_o
            print(f"  Avg build time: {avg_time_o:.3f}s, Recall@{k}: {recall_o:.4f}, "
                  f"Speedup: {speedup_o:.2f}x")

            results.append({
                "method": f"C++ Online frac={frac}",
                "build_time_s": avg_time_o,
                "build_time_std": np.std(times_online),
                "distance_computations": result.distance_computations,
                "skipped_computations": result.skipped_computations,
                "skip_rate": result.skip_rate,
                f"recall@{k}": recall_o,
                "speedup_vs_vanilla": speedup_o,
            })

        # Save results
        df = pd.DataFrame(results)
        out_path = self.output_dir / "cpp_e2e_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
        print("\n" + df.to_string(index=False))


def _compute_recall(predicted_labels, ground_truth, k):
    """Compute recall@k against ground truth."""
    gt_k = ground_truth[:, :k]
    recalls = []
    for i in range(len(predicted_labels)):
        pred = set(predicted_labels[i])
        true = set(gt_k[i])
        recalls.append(len(pred & true) / k)
    return np.mean(recalls)


if __name__ == "__main__":
    exp = CppE2EBenchmark()
    exp.run()
