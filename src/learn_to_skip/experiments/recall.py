"""Recall experiment (REQ-E2): Table 3 + Fig.3 Pareto."""
import numpy as np
import pandas as pd

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
    EF_SEARCH_VALUES, K_VALUES,
)


def compute_recall(index, queries: np.ndarray, groundtruth: np.ndarray, ef_search: int, k: int) -> float:
    """Compute recall@k for a built HNSW index."""
    index.set_ef(ef_search)
    labels, _ = index.knn_query(queries, k=k)
    gt_k = groundtruth[:, :k]
    recalls = []
    for i in range(len(queries)):
        intersection = len(set(int(x) for x in labels[i]) & set(int(x) for x in gt_k[i]))
        recalls.append(intersection / k)
    return float(np.mean(recalls))


class RecallExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "recall"

    def run(self, datasets: list[str] | None = None, **kwargs) -> None:
        datasets = datasets or ["sift10k"]
        rows = []

        for ds_name in datasets:
            print(f"[Recall] {ds_name}...")
            ds = get_dataset(ds_name)
            train = ds.load_train()
            query = ds.load_query()
            gt = ds.load_groundtruth(k=100)
            metric = ds.metadata().metric

            # Build indices with different methods
            trace_path = TRACES_DIR / ds_name / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
            learned_builders = {}
            if trace_path.exists():
                for clf_name in ["logistic", "tree", "xgboost"]:
                    clf, extractor, *_ = prepare_classifier(
                        trace_path=str(trace_path), clf_name=clf_name,
                    )
                    builder = LearnedSkipBuilder(
                        classifier=clf, extractor=extractor, threshold=0.7,
                    )
                    learned_builders[clf_name] = builder

            builders = {
                "Vanilla-HNSW": VanillaHNSWBuilder(),
                "Python-Vanilla": RandomSkipBuilder(skip_prob=0.0),
                "Random-Skip": RandomSkipBuilder(),
                "Dist-Threshold": DistanceThresholdBuilder(),
            }
            for clf_name, lb in learned_builders.items():
                builders[f"LearnToSkip-{clf_name}"] = lb

            M = DEFAULT_M
            ef_c = DEFAULT_EF_CONSTRUCTION

            for method_name, builder in builders.items():
                print(f"  Building {method_name}...")
                result = builder.build(train, M=M, ef_construction=ef_c, metric=metric)

                for ef_s in EF_SEARCH_VALUES:
                    for k in K_VALUES:
                        if k > ef_s:
                            continue
                        recall = compute_recall(result.index, query, gt, ef_s, k)
                        rows.append({
                            "dataset": ds_name,
                            "method": method_name,
                            "M": M,
                            "ef_construction": ef_c,
                            "ef_search": ef_s,
                            "K": k,
                            "recall": recall,
                            "build_time_sec": result.build_time_seconds,
                            "dist_computations": result.distance_computations,
                            "skipped_computations": result.skipped_computations,
                            "classifier_overhead_sec": result.classifier_overhead_seconds,
                        })

        df = pd.DataFrame(rows)

        # Compute speedup based on distance computations (vs Python-Vanilla)
        py_vanilla_dists = df[df["method"] == "Python-Vanilla"].groupby("dataset")["dist_computations"].first()
        df["dist_speedup"] = 1.0
        for idx in df.index:
            ds = df.loc[idx, "dataset"]
            if ds in py_vanilla_dists.index and df.loc[idx, "dist_computations"] > 0:
                df.loc[idx, "dist_speedup"] = py_vanilla_dists[ds] / df.loc[idx, "dist_computations"]

        df.to_csv(self.output_dir / "raw.csv", index=False)

        # Table 3: fixed ef_search=200
        table3 = df[df["ef_search"] == 200].pivot_table(
            index=["dataset", "method"],
            columns="K",
            values="recall",
            aggfunc="mean",
        ).reset_index()
        table3.to_csv(self.output_dir / "table3.csv", index=False)

        # Pareto data (for Fig.3)
        pareto = df[(df["K"] == 10) & (df["ef_search"] == 200)][
            ["dataset", "method", "dist_speedup", "recall"]
        ].drop_duplicates()
        pareto.to_csv(self.output_dir / "pareto_data.csv", index=False)
        print(f"[Recall] Results saved to {self.output_dir}")

    def is_complete(self) -> bool:
        return (self.output_dir / "raw.csv").exists()
