"""Motivation experiment: waste ratio analysis (Fig.1 + Table 1)."""
import pandas as pd
import numpy as np

from learn_to_skip.experiments.base import BaseExperiment
from learn_to_skip.datasets import get_dataset
from learn_to_skip.tracer.hnsw_tracer import HNSWTracer
from learn_to_skip.config import DEFAULT_M, DEFAULT_EF_CONSTRUCTION, TRACES_DIR


class MotivationExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "motivation"

    def run(self, datasets: list[str] | None = None, **kwargs) -> None:
        datasets = datasets or ["sift10k"]
        all_stats = []

        for ds_name in datasets:
            print(f"[Motivation] Processing {ds_name}...")
            ds = get_dataset(ds_name)
            train = ds.load_train()

            # Check for existing trace with new schema
            trace_path = TRACES_DIR / ds_name / f"trace_{DEFAULT_EF_CONSTRUCTION}_{DEFAULT_M}.parquet"
            if trace_path.exists():
                trace_df = pd.read_parquet(trace_path)
                # Detect old schema and regenerate if needed
                if "candidate_rank_in_beam" not in trace_df.columns:
                    print(f"  Old trace schema detected, regenerating...")
                    trace_df = self._generate_trace(train, ds_name)
                else:
                    print(f"  Loading existing trace from {trace_path}")
            else:
                trace_df = self._generate_trace(train, ds_name)

            # Compute statistics
            total_evals = len(trace_df)
            retained = trace_df["label_retained"].sum()
            waste_ratio = 1.0 - retained / total_evals if total_evals > 0 else 0

            all_stats.append({
                "dataset": ds_name,
                "total_evaluations": total_evals,
                "retained": int(retained),
                "discarded": total_evals - int(retained),
                "waste_ratio": waste_ratio,
                "n_vectors": len(train),
                "M": DEFAULT_M,
                "ef_construction": DEFAULT_EF_CONSTRUCTION,
            })
            print(f"  {ds_name}: waste_ratio={waste_ratio:.3f}, total_evals={total_evals}")

        # Save Table 1
        df_stats = pd.DataFrame(all_stats)
        df_stats.to_csv(self.output_dir / "table1.csv", index=False)
        df_stats.to_csv(self.output_dir / "fig1_data.csv", index=False)
        print(f"[Motivation] Results saved to {self.output_dir}")

    def _generate_trace(self, train: np.ndarray, ds_name: str) -> pd.DataFrame:
        tracer = HNSWTracer(dim=train.shape[1], M=DEFAULT_M, ef_construction=DEFAULT_EF_CONSTRUCTION)
        tracer.build(train)
        tracer.save_trace(ds_name)
        return tracer.get_trace_df()

    def is_complete(self) -> bool:
        return (self.output_dir / "table1.csv").exists()
