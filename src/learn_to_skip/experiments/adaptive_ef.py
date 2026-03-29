"""Adaptive ef_construction experiment with Thompson Sampling (REQ-E8)."""
import time

import pandas as pd
import numpy as np
import hnswlib

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.experiments.recall import compute_recall
from learn_to_skip.datasets import get_dataset
from learn_to_skip.datasets.streaming import StreamingDataGenerator
from learn_to_skip.adaptive.thompson import ThompsonSamplingTuner
from learn_to_skip.pipeline import prepare_classifier
from learn_to_skip.features.extractor import FeatureSet
from learn_to_skip.config import (
    TRACES_DIR, DEFAULT_M, DEFAULT_EF_CONSTRUCTION,
    TS_EVAL_QUERIES, SEED,
)


def _build_streaming(
    data: np.ndarray,
    dim: int,
    metric: str,
    M: int,
    streamer: StreamingDataGenerator,
    use_ts: bool,
    skip_rate: float = 0.0,
    fixed_ef: int = 200,
) -> tuple[object, list[dict]]:
    """Build index from streaming data, returning (index, per-batch stats)."""
    from sklearn.neighbors import NearestNeighbors

    space = "l2" if metric == "l2" else "cosine"
    effective_ef = max(M + 1, int(fixed_ef * (1 - skip_rate * 0.8)))
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=len(data), M=M, ef_construction=effective_ef)

    ts = ThompsonSamplingTuner(seed=SEED) if use_ts else None
    batch_stats = []
    inserted = 0
    all_streamed_data = []

    for batch_idx, (batch_data, batch_labels) in enumerate(streamer.stream()):
        if use_ts:
            ef_c = ts.select_arm()
        else:
            ef_c = fixed_ef

        ids = np.arange(inserted, inserted + len(batch_data))
        t0 = time.time()
        index.add_items(batch_data, ids)
        batch_time = time.time() - t0
        inserted += len(batch_data)
        all_streamed_data.append(batch_data)

        # Evaluate recall
        recall = 0.0
        if inserted > TS_EVAL_QUERIES and inserted > 10:
            n_eval = min(TS_EVAL_QUERIES, len(batch_data))
            eval_queries = batch_data[:n_eval]

            all_data = np.vstack(all_streamed_data)
            nn = NearestNeighbors(
                n_neighbors=min(10, inserted),
                metric="euclidean" if metric == "l2" else "cosine",
                algorithm="brute",
            )
            nn.fit(all_data)
            _, gt_ids = nn.kneighbors(eval_queries)

            index.set_ef(200)
            k = min(10, inserted)
            pred_labels, _ = index.knn_query(eval_queries, k=k)
            recalls = [
                len(set(int(x) for x in pred_labels[i]) & set(int(x) for x in gt_ids[i])) / k
                for i in range(n_eval)
            ]
            recall = float(np.mean(recalls))

        max_time = len(batch_data) * 0.01
        normalized_time = min(batch_time / max(max_time, 0.001), 1.0)
        if use_ts and ts is not None:
            ts.update(ef_c, recall, normalized_time)

        batch_stats.append({
            "batch": batch_idx,
            "n_inserted": inserted,
            "ef_construction": ef_c,
            "batch_time": batch_time,
            "recall_at_10": recall,
        })

    return index, batch_stats


class AdaptiveEfExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "adaptive"

    def run(
        self,
        dataset: str = "sift10k",
        best_classifier: str = "xgboost",
        **kwargs,
    ) -> None:
        print(f"[AdaptiveEf] {dataset}...")
        ds = get_dataset(dataset)
        train = ds.load_train()
        metric = ds.metadata().metric
        dim = ds.metadata().dim

        streamer = StreamingDataGenerator(train, n_clusters=min(10, len(train) // 100))

        # Get skip rate from trained classifier with temporal split
        trace_path = TRACES_DIR / dataset / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
        skip_rate = 0.0
        if trace_path.exists():
            clf, extractor, X_train, y_train, X_test, y_test = prepare_classifier(
                trace_path=str(trace_path), clf_name=best_classifier,
            )
            proba = clf.predict_proba(X_test)
            skip_rate = float(np.mean(proba > 0.7))

        configs = [
            ("Vanilla+Fixed", False, 0.0, DEFAULT_EF_CONSTRUCTION),
            ("Vanilla+TS", True, 0.0, DEFAULT_EF_CONSTRUCTION),
            ("LearnToSkip+Fixed", False, skip_rate, DEFAULT_EF_CONSTRUCTION),
            ("LearnToSkip+TS", True, skip_rate, DEFAULT_EF_CONSTRUCTION),
        ]

        all_stats = []
        summary_rows = []

        for config_name, use_ts, sr, fixed_ef in configs:
            print(f"  Config: {config_name}")
            streamer_copy = StreamingDataGenerator(train, n_clusters=min(10, len(train) // 100))
            index, batch_stats = _build_streaming(
                train, dim, metric, DEFAULT_M, streamer_copy,
                use_ts=use_ts, skip_rate=sr, fixed_ef=fixed_ef,
            )

            for bs in batch_stats:
                bs["config"] = config_name
            all_stats.extend(batch_stats)

            total_time = sum(bs["batch_time"] for bs in batch_stats)
            final_recall = batch_stats[-1]["recall_at_10"] if batch_stats else 0.0

            summary_rows.append({
                "config": config_name,
                "total_build_time_sec": total_time,
                "final_recall_at_10": final_recall,
                "n_batches": len(batch_stats),
            })

        df_stats = pd.DataFrame(all_stats)
        df_stats.to_csv(self.output_dir / "fig7_data.csv", index=False)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(self.output_dir / "table7_joint.csv", index=False)
        print(f"[AdaptiveEf] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "table7_joint.csv").exists()
