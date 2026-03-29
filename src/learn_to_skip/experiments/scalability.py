"""Scalability experiment (REQ-E6): Fig.6."""
import pandas as pd
import numpy as np

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.datasets import get_dataset
from learn_to_skip.builders import VanillaHNSWBuilder, LearnedSkipBuilder
from learn_to_skip.pipeline import prepare_classifier
from learn_to_skip.features.extractor import FeatureSet
from learn_to_skip.config import (
    TRACES_DIR, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, SCALABILITY_SIZES,
)


class ScalabilityExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "scalability"

    def run(
        self,
        dataset: str = "sift10k",
        best_classifier: str = "xgboost",
        **kwargs,
    ) -> None:
        print(f"[Scalability] {dataset}...")
        ds = get_dataset(dataset)
        full_train = ds.load_train()
        metric = ds.metadata().metric
        n_total = len(full_train)

        # Train classifier from trace with temporal split
        trace_path = TRACES_DIR / dataset / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
        learned_builder = None
        if trace_path.exists():
            clf, extractor, *_ = prepare_classifier(
                trace_path=str(trace_path), clf_name=best_classifier,
            )
            learned_builder = LearnedSkipBuilder(
                classifier=clf, extractor=extractor, threshold=0.7,
            )

        rows = []
        sizes = [s for s in SCALABILITY_SIZES if s <= n_total]
        if n_total not in sizes:
            sizes.append(n_total)

        for size in sizes:
            print(f"  Size: {size}")
            data_subset = full_train[:size]

            # Vanilla
            vanilla = VanillaHNSWBuilder()
            v_result = vanilla.build(data_subset, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)
            rows.append({
                "dataset": dataset,
                "size": size,
                "method": "Vanilla-HNSW",
                "build_time_sec": v_result.build_time_seconds,
                "dist_computations": v_result.distance_computations,
            })

            # LearnToSkip
            if learned_builder is not None:
                l_result = learned_builder.build(data_subset, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)
                rows.append({
                    "dataset": dataset,
                    "size": size,
                    "method": f"LearnToSkip-{best_classifier}",
                    "build_time_sec": l_result.build_time_seconds,
                    "dist_computations": l_result.distance_computations,
                    "skipped_computations": l_result.skipped_computations,
                    "classifier_overhead_sec": l_result.classifier_overhead_seconds,
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "fig6_data.csv", index=False)
        print(f"[Scalability] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "fig6_data.csv").exists()
