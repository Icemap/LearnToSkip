"""Classifier analysis experiment (REQ-E3): Table 4 + Fig.4 ROC."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.features.extractor import FeatureExtractor, FeatureSet
from learn_to_skip.classifiers import get_classifier, CLASSIFIER_REGISTRY
from learn_to_skip.tracer.hnsw_tracer import temporal_split_trace
from learn_to_skip.pipeline import MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES
from learn_to_skip.config import TRACES_DIR, DEFAULT_M, DEFAULT_EF_CONSTRUCTION


class ClassifierAnalysisExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "classifier"

    def run(self, datasets: list[str] | None = None, **kwargs) -> None:
        datasets = datasets or ["sift10k"]
        all_rows = []
        roc_data = []

        for ds_name in datasets:
            print(f"[ClassifierAnalysis] {ds_name}...")
            trace_path = TRACES_DIR / ds_name / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
            if not trace_path.exists():
                print(f"  Trace not found at {trace_path}, skipping")
                continue

            trace_df = pd.read_parquet(trace_path)
            train_df, test_df = temporal_split_trace(trace_df)

            # Subsample for tractable training
            from learn_to_skip.pipeline import _subsample_stratified
            train_df = _subsample_stratified(train_df, MAX_TRAIN_SAMPLES)
            test_df = _subsample_stratified(test_df, MAX_TEST_SAMPLES)

            extractor = FeatureExtractor(FeatureSet.FULL)
            X_train, y_train = extractor.fit_transform(train_df)
            X_test, y_test = extractor.transform(test_df)
            y_test_skip = 1 - y_test

            print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

            for clf_name, clf_cls in CLASSIFIER_REGISTRY.items():
                print(f"  Training {clf_name}...")
                clf = clf_cls()
                metrics = clf.evaluate_holdout(X_train, y_train, X_test, y_test)

                all_rows.append({
                    "dataset": ds_name,
                    "classifier": clf_name,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "auc": metrics.auc,
                    "train_time_sec": metrics.train_time_sec,
                    "model_size_bytes": metrics.model_size_bytes,
                    "inference_time_ns": metrics.inference_time_ns,
                })

                # ROC curve on held-out test set
                y_proba = clf.predict_proba(X_test)
                fpr, tpr, _ = roc_curve(y_test_skip, y_proba)
                roc_auc = auc(fpr, tpr)
                roc_data.append({
                    "dataset": ds_name,
                    "classifier": clf_name,
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": roc_auc,
                })

        # Save Table 4
        df = pd.DataFrame(all_rows)
        df.to_csv(self.output_dir / "table4.csv", index=False)

        # Save ROC data for plotting
        roc_df = pd.DataFrame(roc_data)
        roc_df.to_json(self.output_dir / "roc_data.json", orient="records")
        print(f"[ClassifierAnalysis] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "table4.csv").exists()
