"""Threshold sensitivity experiment (REQ-E5): Fig.5."""
import pandas as pd
import numpy as np

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.experiments.recall import compute_recall
from learn_to_skip.datasets import get_dataset
from learn_to_skip.pipeline import prepare_classifier
from learn_to_skip.features.extractor import FeatureSet
from learn_to_skip.builders import VanillaHNSWBuilder, RandomSkipBuilder, LearnedSkipBuilder
from learn_to_skip.config import (
    TRACES_DIR, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, THRESHOLD_VALUES,
)


class ThresholdSensitivityExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "threshold"

    def run(
        self,
        dataset: str = "sift10k",
        best_classifier: str = "xgboost",
        **kwargs,
    ) -> None:
        print(f"[Threshold] {dataset}...")
        ds = get_dataset(dataset)
        train = ds.load_train()
        query = ds.load_query()
        gt = ds.load_groundtruth(k=100)
        metric = ds.metadata().metric

        trace_path = TRACES_DIR / dataset / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
        if not trace_path.exists():
            print(f"  Trace not found, skipping")
            return

        clf, extractor, X_train, y_train, X_test, y_test = prepare_classifier(
            trace_path=str(trace_path), clf_name=best_classifier,
        )

        # Python-Vanilla baseline (fair comparison — same Python HNSW implementation)
        py_vanilla = RandomSkipBuilder(skip_prob=0.0)
        py_vanilla_result = py_vanilla.build(train, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)

        rows = []
        for tau in THRESHOLD_VALUES:
            print(f"  Threshold τ={tau}")
            builder = LearnedSkipBuilder(
                classifier=clf, extractor=extractor, threshold=tau,
            )
            result = builder.build(train, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)
            recall_10 = compute_recall(result.index, query, gt, ef_search=200, k=10)

            rows.append({
                "threshold": tau,
                "skip_rate": result.skip_rate,
                "dist_speedup": py_vanilla_result.distance_computations / max(result.distance_computations, 1),
                "recall_at_10": recall_10,
                "build_time_sec": result.build_time_seconds,
                "classifier_overhead_sec": result.classifier_overhead_seconds,
                "skipped_computations": result.skipped_computations,
                "distance_computations": result.distance_computations,
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "fig5_data.csv", index=False)
        print(f"[Threshold] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "fig5_data.csv").exists()
