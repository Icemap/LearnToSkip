"""Universal pre-trained classifier experiment.

Tests whether a decision tree trained on one dataset's structural features
generalizes across different data distributions, dimensionalities, and scales.

If successful, we can ship a single pre-trained tree that users apply directly
with zero training cost.
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
from learn_to_skip.tracer.hnsw_tracer import HNSWTracer, temporal_split_trace
from learn_to_skip.pipeline import _subsample_stratified, MAX_TRAIN_SAMPLES, MAX_TEST_SAMPLES
from learn_to_skip.config import (
    TRACES_DIR, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, SEED,
)


def _generate_synthetic_data(name: str, n: int = 10000, seed: int = SEED):
    """Generate synthetic datasets with different distributions."""
    rng = np.random.RandomState(seed)
    if name == "uniform_128":
        return rng.uniform(0, 256, (n, 128)).astype(np.float32)
    elif name == "gaussian_128":
        return rng.randn(n, 128).astype(np.float32)
    elif name == "clustered_128":
        # 10 clusters
        centers = rng.randn(10, 128).astype(np.float32) * 50
        labels = rng.randint(0, 10, n)
        data = centers[labels] + rng.randn(n, 128).astype(np.float32) * 5
        return data
    elif name == "uniform_32":
        return rng.uniform(0, 256, (n, 32)).astype(np.float32)
    elif name == "uniform_256":
        return rng.uniform(0, 256, (n, 256)).astype(np.float32)
    elif name == "high_dim_512":
        return rng.randn(n, 512).astype(np.float32)
    else:
        raise ValueError(f"Unknown synthetic dataset: {name}")


class UniversalClassifierExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "universal_classifier"

    def run(
        self,
        dataset: str = "sift10k",
        best_classifier: str = "tree",
        **kwargs,
    ) -> None:
        print(f"[UniversalClassifier] Training on {dataset}...")

        # === Step 1: Train the "universal" classifier on source dataset ===
        trace_path = TRACES_DIR / dataset / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
        if not trace_path.exists():
            print(f"  Trace not found for {dataset}, cannot proceed")
            return

        df = pd.read_parquet(trace_path)
        train_df, test_df = temporal_split_trace(df)
        train_sub = _subsample_stratified(train_df, MAX_TRAIN_SAMPLES)

        extractor = FeatureExtractor(FeatureSet.FULL)
        X_train, y_train = extractor.fit_transform(train_sub)

        clf = get_classifier(best_classifier)
        clf.train(X_train, y_train)

        print(f"  Trained {best_classifier} on {dataset} ({len(train_sub):,} samples)")

        # === Step 2: Test on source dataset (sanity check) ===
        test_sub = _subsample_stratified(test_df, MAX_TEST_SAMPLES)
        X_test_src, y_test_src = extractor.transform(test_sub)
        y_skip_src = 1 - y_test_src
        auc_src = roc_auc_score(y_skip_src, clf.predict_proba(X_test_src))
        print(f"  Source AUC: {auc_src:.3f}")

        # === Step 3: Test on synthetic datasets with different properties ===
        synthetic_datasets = [
            "uniform_128",      # same dim, different distribution
            "gaussian_128",     # same dim, gaussian
            "clustered_128",    # same dim, clustered
            "uniform_32",       # lower dim
            "uniform_256",      # higher dim
            "high_dim_512",     # much higher dim
        ]

        rows = []

        # Source dataset result
        ds = get_dataset(dataset)
        train_data = ds.load_train()
        query = ds.load_query()
        gt = ds.load_groundtruth(k=100)
        metric = ds.metadata().metric

        py_vanilla = RandomSkipBuilder(skip_prob=0.0)
        pv_result = py_vanilla.build(train_data, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)
        pv_recall = compute_recall(pv_result.index, query, gt, ef_search=200, k=10)

        builder = LearnedSkipBuilder(classifier=clf, extractor=extractor, threshold=0.7)
        ls_result = builder.build(train_data, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, metric)
        ls_recall = compute_recall(ls_result.index, query, gt, ef_search=200, k=10)

        rows.append({
            "source_dataset": dataset,
            "target_dataset": dataset,
            "target_dim": train_data.shape[1],
            "target_n": train_data.shape[0],
            "target_distribution": "real",
            "auc_on_target_trace": auc_src,
            "skip_rate": ls_result.skip_rate,
            "dist_speedup": pv_result.distance_computations / max(ls_result.distance_computations, 1),
            "recall_at_10": ls_recall,
            "vanilla_recall": pv_recall,
            "recall_drop": pv_recall - ls_recall,
            "build_time_sec": ls_result.build_time_seconds,
        })

        # Synthetic datasets
        for syn_name in synthetic_datasets:
            print(f"  Testing on {syn_name}...")
            syn_data = _generate_synthetic_data(syn_name, n=5000)
            dim = syn_data.shape[1]

            # Generate queries and ground truth
            rng = np.random.RandomState(SEED + 1)
            n_query = min(100, len(syn_data) // 10)
            query_idx = rng.choice(len(syn_data), n_query, replace=False)
            syn_query = syn_data[query_idx]

            # Brute force ground truth
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=100, metric="euclidean", algorithm="brute")
            nn.fit(syn_data)
            _, syn_gt = nn.kneighbors(syn_query)

            # Build Python-Vanilla baseline
            pv = RandomSkipBuilder(skip_prob=0.0)
            pv_res = pv.build(syn_data, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, "l2")
            pv_recall = compute_recall(pv_res.index, syn_query, syn_gt, ef_search=200, k=10)

            # Build with transferred classifier
            # Note: extractor normalization is from source dataset,
            # this tests whether structural features transfer
            ls_builder = LearnedSkipBuilder(
                classifier=clf, extractor=extractor, threshold=0.7,
            )
            ls_res = ls_builder.build(syn_data, DEFAULT_M, DEFAULT_EF_CONSTRUCTION, "l2")
            ls_recall = compute_recall(ls_res.index, syn_query, syn_gt, ef_search=200, k=10)

            # Generate trace for the synthetic dataset to measure AUC
            tracer = HNSWTracer(dim=dim, M=DEFAULT_M, ef_construction=DEFAULT_EF_CONSTRUCTION)
            tracer.build(syn_data)
            syn_trace = tracer.get_trace_df()
            if len(syn_trace) > 0:
                _, syn_test = temporal_split_trace(syn_trace)
                syn_test_sub = _subsample_stratified(syn_test, MAX_TEST_SAMPLES)
                try:
                    X_syn, y_syn = extractor.transform(syn_test_sub)
                    y_syn_skip = 1 - y_syn
                    auc_syn = roc_auc_score(y_syn_skip, clf.predict_proba(X_syn))
                except Exception:
                    auc_syn = float("nan")
            else:
                auc_syn = float("nan")

            rows.append({
                "source_dataset": dataset,
                "target_dataset": syn_name,
                "target_dim": dim,
                "target_n": len(syn_data),
                "target_distribution": syn_name.split("_")[0],
                "auc_on_target_trace": auc_syn,
                "skip_rate": ls_res.skip_rate,
                "dist_speedup": pv_res.distance_computations / max(ls_res.distance_computations, 1),
                "recall_at_10": ls_recall,
                "vanilla_recall": pv_recall,
                "recall_drop": pv_recall - ls_recall,
                "build_time_sec": ls_res.build_time_seconds,
            })

            print(f"    AUC={auc_syn:.3f}, skip={ls_res.skip_rate:.1%}, "
                  f"speedup={rows[-1]['dist_speedup']:.2f}x, "
                  f"recall={ls_recall:.3f} (vanilla={pv_recall:.3f})")

        result_df = pd.DataFrame(rows)
        result_df.to_csv(self.output_dir / "universal_classifier.csv", index=False)
        print(f"[UniversalClassifier] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "universal_classifier.csv").exists()
