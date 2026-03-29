"""Online self-training experiment: train on first X% insertions, skip on remainder.

Validates that a classifier trained on early HNSW construction trace data
generalizes to later insertions, enabling single-pass construction.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.experiments.recall import compute_recall
from learn_to_skip.datasets import get_dataset
from learn_to_skip.features.extractor import FeatureExtractor, FeatureSet
from learn_to_skip.classifiers import get_classifier
from learn_to_skip.builders import RandomSkipBuilder, LearnedSkipBuilder
from learn_to_skip.pipeline import _subsample_stratified, MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES
from learn_to_skip.config import (
    TRACES_DIR, DEFAULT_M, DEFAULT_EF_CONSTRUCTION,
)


class OnlineTrainingExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "online_training"

    def run(
        self,
        dataset: str = "sift10k",
        best_classifier: str = "tree",
        **kwargs,
    ) -> None:
        print(f"[OnlineTraining] {dataset}...")
        ds = get_dataset(dataset)
        train = ds.load_train()
        query = ds.load_query()
        gt = ds.load_groundtruth(k=100)
        metric = ds.metadata().metric

        trace_path = TRACES_DIR / dataset / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
        if not trace_path.exists():
            print(f"  Trace not found, skipping")
            return

        df = pd.read_parquet(trace_path)
        max_insert = df["insert_id"].max()

        # Python-Vanilla baseline
        print("  Building Python-Vanilla baseline...")
        py_vanilla = RandomSkipBuilder(skip_prob=0.0)
        py_vanilla_result = py_vanilla.build(train, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)
        vanilla_recall = compute_recall(py_vanilla_result.index, query, gt, ef_search=200, k=10)

        rows = []
        # Test different train fractions: how much early data do we need?
        for train_frac in [0.10, 0.20, 0.30, 0.50, 0.70]:
            cutoff = int(max_insert * train_frac)
            train_df = df[df["insert_id"] <= cutoff]
            test_df = df[df["insert_id"] > cutoff]

            print(f"  train_frac={train_frac:.0%}: "
                  f"train={len(train_df):,} rows (insert_id<=={cutoff}), "
                  f"test={len(test_df):,} rows")

            # Subsample for tractable training
            train_sub = _subsample_stratified(train_df, MAX_TRAIN_SAMPLES)
            test_sub = _subsample_stratified(test_df, MAX_TEST_SAMPLES)

            # Fit extractor on train portion only
            extractor = FeatureExtractor(FeatureSet.FULL)
            X_train, y_train = extractor.fit_transform(train_sub)
            X_test, y_test = extractor.transform(test_sub)

            # Train classifier
            clf = get_classifier(best_classifier)
            clf.train(X_train, y_train)

            # Evaluate on held-out (later insertions)
            y_test_skip = 1 - y_test
            y_proba = clf.predict_proba(X_test)
            y_pred = clf.predict(X_test)
            auc = roc_auc_score(y_test_skip, y_proba)
            f1 = f1_score(y_test_skip, y_pred, zero_division=0)

            # Build full index with this classifier
            builder = LearnedSkipBuilder(
                classifier=clf, extractor=extractor, threshold=0.7,
            )
            result = builder.build(train, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)
            recall_10 = compute_recall(result.index, query, gt, ef_search=200, k=10)

            rows.append({
                "dataset": dataset,
                "train_fraction": train_frac,
                "train_cutoff_insert_id": cutoff,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "classifier": best_classifier,
                "auc_on_future": auc,
                "f1_on_future": f1,
                "skip_rate": result.skip_rate,
                "dist_speedup": py_vanilla_result.distance_computations / max(result.distance_computations, 1),
                "recall_at_10": recall_10,
                "vanilla_recall": vanilla_recall,
                "recall_drop": vanilla_recall - recall_10,
                "build_time_sec": result.build_time_seconds,
                "classifier_overhead_sec": result.classifier_overhead_seconds,
                "distance_computations": result.distance_computations,
                "skipped_computations": result.skipped_computations,
            })

            print(f"    AUC={auc:.3f}, F1={f1:.3f}, "
                  f"skip_rate={result.skip_rate:.1%}, "
                  f"speedup={rows[-1]['dist_speedup']:.2f}x, "
                  f"recall={recall_10:.3f}")

        result_df = pd.DataFrame(rows)
        result_df.to_csv(self.output_dir / "online_training.csv", index=False)
        print(f"[OnlineTraining] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "online_training.csv").exists()
