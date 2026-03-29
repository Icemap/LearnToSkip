"""Cross-dataset generalization experiment (REQ-E7): Table 6."""
import pandas as pd
import numpy as np

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.experiments.recall import compute_recall
from learn_to_skip.datasets import get_dataset
from learn_to_skip.features.extractor import FeatureExtractor, FeatureSet
from learn_to_skip.classifiers import get_classifier
from learn_to_skip.builders import VanillaHNSWBuilder, RandomSkipBuilder, LearnedSkipBuilder
from learn_to_skip.tracer.hnsw_tracer import temporal_split_trace
from learn_to_skip.config import (
    TRACES_DIR, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, TRANSFER_DATASETS,
)


class GeneralizationExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "generalization"

    def run(
        self,
        datasets: list[str] | None = None,
        best_classifier: str = "xgboost",
        **kwargs,
    ) -> None:
        datasets = datasets or TRANSFER_DATASETS
        rows = []

        # Train classifiers on each dataset with temporal split
        trained_clfs = {}
        extractors = {}
        for ds_name in datasets:
            trace_path = TRACES_DIR / ds_name / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
            if not trace_path.exists():
                print(f"  No trace for {ds_name}, skipping as train source")
                continue
            trace_df = pd.read_parquet(trace_path)
            train_df, _ = temporal_split_trace(trace_df)
            extractor = FeatureExtractor(FeatureSet.FULL)
            X_train, y_train = extractor.fit_transform(train_df)
            clf = get_classifier(best_classifier)
            clf.train(X_train, y_train)
            trained_clfs[ds_name] = clf
            extractors[ds_name] = extractor

        # Test each (train_ds, test_ds) pair
        for train_ds in datasets:
            if train_ds not in trained_clfs:
                continue
            clf = trained_clfs[train_ds]
            extractor = extractors[train_ds]

            for test_ds in datasets:
                print(f"[Generalization] Train={train_ds}, Test={test_ds}")
                ds = get_dataset(test_ds)
                train_data = ds.load_train()
                query = ds.load_query()
                gt = ds.load_groundtruth(k=100)
                metric = ds.metadata().metric

                # Vanilla baseline (hnswlib for recall comparison)
                vanilla = VanillaHNSWBuilder()
                v_result = vanilla.build(train_data, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)
                v_recall = compute_recall(v_result.index, query, gt, ef_search=200, k=10)

                # Python-Vanilla baseline for fair dist_speedup
                py_vanilla = RandomSkipBuilder(skip_prob=0.0)
                py_vanilla_result = py_vanilla.build(train_data, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)

                # Build with transferred classifier
                builder = LearnedSkipBuilder(
                    classifier=clf, extractor=extractor, threshold=0.7,
                )
                l_result = builder.build(train_data, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)

                recall_10 = compute_recall(l_result.index, query, gt, ef_search=200, k=10)

                rows.append({
                    "train_dataset": train_ds,
                    "test_dataset": test_ds,
                    "dist_speedup": py_vanilla_result.distance_computations / max(l_result.distance_computations, 1),
                    "recall_at_10": recall_10,
                    "vanilla_recall": v_recall,
                    "recall_drop": v_recall - recall_10,
                    "skipped_computations": l_result.skipped_computations,
                    "classifier_overhead_sec": l_result.classifier_overhead_seconds,
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "table6_transfer.csv", index=False)
        print(f"[Generalization] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "table6_transfer.csv").exists()
