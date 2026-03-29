"""Feature ablation experiment (REQ-E4): Table 5."""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.experiments.recall import compute_recall
from learn_to_skip.datasets import get_dataset
from learn_to_skip.features.extractor import FeatureExtractor, FeatureSet
from learn_to_skip.classifiers import get_classifier
from learn_to_skip.builders import VanillaHNSWBuilder, RandomSkipBuilder, LearnedSkipBuilder
from learn_to_skip.tracer.hnsw_tracer import temporal_split_trace
from learn_to_skip.pipeline import MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES
from learn_to_skip.config import (
    TRACES_DIR, DEFAULT_M, DEFAULT_EF_CONSTRUCTION,
)


class AblationExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "ablation"

    def run(self, datasets: list[str] | None = None, best_classifier: str = "xgboost", **kwargs) -> None:
        datasets = datasets or ["sift10k"]
        rows = []

        for ds_name in datasets:
            print(f"[Ablation] {ds_name}...")
            ds = get_dataset(ds_name)
            train = ds.load_train()
            query = ds.load_query()
            gt = ds.load_groundtruth(k=100)
            metric = ds.metadata().metric

            trace_path = TRACES_DIR / ds_name / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
            if not trace_path.exists():
                print(f"  Trace not found, skipping")
                continue
            trace_df = pd.read_parquet(trace_path)
            train_df, test_df = temporal_split_trace(trace_df)

            from learn_to_skip.pipeline import _subsample_stratified
            train_df = _subsample_stratified(train_df, MAX_TRAIN_SAMPLES)
            test_df = _subsample_stratified(test_df, MAX_TEST_SAMPLES)

            # Python-Vanilla baseline for fair dist_speedup comparison
            py_vanilla = RandomSkipBuilder(skip_prob=0.0)
            py_vanilla_result = py_vanilla.build(train, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)

            for fs in FeatureSet:
                print(f"  Feature set: {fs.value}")
                extractor = FeatureExtractor(fs)
                X_train, y_train = extractor.fit_transform(train_df)
                X_test, y_test = extractor.transform(test_df)

                clf = get_classifier(best_classifier)
                clf.train(X_train, y_train)

                # Classifier F1 on held-out test set
                y_test_skip = 1 - y_test
                y_pred = clf.predict(X_test)
                f1 = f1_score(y_test_skip, y_pred, zero_division=0)

                # Build with this classifier
                builder = LearnedSkipBuilder(
                    classifier=clf, extractor=extractor, threshold=0.7,
                )
                result = builder.build(train, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)

                # Recall
                recall_10 = compute_recall(result.index, query, gt, ef_search=200, k=10)

                rows.append({
                    "dataset": ds_name,
                    "feature_set": fs.value,
                    "n_features": extractor.n_features,
                    "classifier_f1": f1,
                    "dist_speedup": py_vanilla_result.distance_computations / max(result.distance_computations, 1),
                    "recall_at_10": recall_10,
                    "build_time_sec": result.build_time_seconds,
                    "classifier_overhead_sec": result.classifier_overhead_seconds,
                    "skipped_computations": result.skipped_computations,
                    "distance_computations": result.distance_computations,
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "table5.csv", index=False)
        print(f"[Ablation] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "table5.csv").exists()
