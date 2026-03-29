"""Build speed experiment (REQ-E1): Table 2 + Fig.2."""
import pandas as pd
import numpy as np

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.datasets import get_dataset
from learn_to_skip.builders import (
    VanillaHNSWBuilder, RandomSkipBuilder,
    DistanceThresholdBuilder, LearnedSkipBuilder,
)
from learn_to_skip.pipeline import prepare_classifier
from learn_to_skip.features.extractor import FeatureSet
from learn_to_skip.config import (
    TRACES_DIR, DEFAULT_M, DEFAULT_EF_CONSTRUCTION,
    N_REPEATS, M_VALUES, EF_CONSTRUCTION_VALUES, DEV_DATASETS,
)


class BuildSpeedExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "build_speed"

    def run(self, datasets: list[str] | None = None, **kwargs) -> None:
        datasets = datasets or ["sift10k"]
        rows = []

        for ds_name in datasets:
            print(f"[BuildSpeed] {ds_name}...")
            ds = get_dataset(ds_name)
            train = ds.load_train()
            metric = ds.metadata().metric

            # Train LearnToSkip classifiers from trace with temporal split
            trace_path = TRACES_DIR / ds_name / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
            learned_builders = {}
            if trace_path.exists():
                for clf_name in ["logistic", "tree", "xgboost"]:
                    clf, extractor, X_train, y_train, X_test, y_test = prepare_classifier(
                        trace_path=str(trace_path), clf_name=clf_name,
                    )
                    builder = LearnedSkipBuilder(
                        classifier=clf, extractor=extractor, threshold=0.7,
                    )
                    learned_builders[clf_name] = builder

            # All builders to test
            # Python-Vanilla (skip_prob=0) as fair baseline for Python-HNSW methods
            builders = {
                "Vanilla-HNSW": VanillaHNSWBuilder(),
                "Python-Vanilla": RandomSkipBuilder(skip_prob=0.0),
                "Random-Skip": RandomSkipBuilder(),
                "Dist-Threshold": DistanceThresholdBuilder(),
            }
            for clf_name, lb in learned_builders.items():
                builders[f"LearnToSkip-{clf_name}"] = lb

            # Use reduced grid for dev datasets (Python HNSW is slow)
            is_dev = ds_name in DEV_DATASETS
            m_vals = [DEFAULT_M] if is_dev else M_VALUES
            ef_vals = [DEFAULT_EF_CONSTRUCTION] if is_dev else EF_CONSTRUCTION_VALUES
            n_reps = 1 if is_dev else N_REPEATS

            for M in m_vals:
                for ef in ef_vals:
                    for method_name, builder in builders.items():
                        for run_id in range(n_reps):
                            print(f"  {ds_name}/{method_name}/M={M}/ef={ef}/run={run_id}")
                            result = builder.build(train, M=M, ef_construction=ef, metric=metric)
                            rows.append({
                                "dataset": ds_name,
                                "method": method_name,
                                "M": M,
                                "ef_construction": ef,
                                "run_id": run_id,
                                "build_time_sec": result.build_time_seconds,
                                "dist_computations": result.distance_computations,
                                "skipped_computations": result.skipped_computations,
                                "classifier_overhead_sec": result.classifier_overhead_seconds,
                                "n_classifier_calls": result.n_classifier_calls,
                            })

        # Save raw results
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "raw.csv", index=False)

        # Aggregate Table 2
        agg = df.groupby(["dataset", "method", "M", "ef_construction"]).agg(
            build_time_mean=("build_time_sec", "mean"),
            build_time_std=("build_time_sec", "std"),
            dist_comp_mean=("dist_computations", "mean"),
            skipped_mean=("skipped_computations", "mean"),
            clf_overhead_mean=("classifier_overhead_sec", "mean"),
        ).reset_index()

        # Compute dist_speedup relative to Python-Vanilla (fair comparison)
        py_vanilla_dists = agg[agg["method"] == "Python-Vanilla"][
            ["dataset", "M", "ef_construction", "dist_comp_mean"]
        ].rename(columns={"dist_comp_mean": "py_vanilla_dists"})
        agg = agg.merge(py_vanilla_dists, on=["dataset", "M", "ef_construction"])
        agg["dist_speedup"] = agg["py_vanilla_dists"] / agg["dist_comp_mean"]

        # Wall-clock speedup relative to hnswlib Vanilla
        vanilla_times = agg[agg["method"] == "Vanilla-HNSW"][
            ["dataset", "M", "ef_construction", "build_time_mean"]
        ].rename(columns={"build_time_mean": "vanilla_time"})
        agg = agg.merge(vanilla_times, on=["dataset", "M", "ef_construction"])
        agg["wall_speedup"] = agg["vanilla_time"] / agg["build_time_mean"]

        agg.to_csv(self.output_dir / "table2.csv", index=False)
        print(f"[BuildSpeed] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "raw.csv").exists()
