"""Experiment runner with dependency management."""
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from learn_to_skip.config import RESULTS_DIR, SEED, MAIN_DATASETS
from learn_to_skip.experiments import EXPERIMENT_REGISTRY
from learn_to_skip.visualization.plots import (
    plot_waste_ratio_bar,
    plot_speedup_bar,
    plot_pareto_scatter,
    plot_roc_curves,
    plot_threshold_sensitivity,
    plot_scalability_line,
    plot_recall_over_time,
)

# Dependency DAG: child -> [parents]
DEPENDENCIES = {
    "motivation": [],
    "classifier": ["motivation"],
    "ablation": ["classifier"],
    "threshold": ["classifier"],
    "build_speed": ["motivation"],
    "recall": ["build_speed"],
    "scalability": ["motivation"],
    "generalization": ["motivation"],
    "adaptive": ["motivation"],
}

EXPERIMENT_ORDER = [
    "motivation", "classifier", "build_speed",
    "ablation", "threshold", "recall",
    "scalability", "generalization", "adaptive",
]


class ExperimentRunner:
    def __init__(self, datasets: list[str] | None = None, force: bool = False) -> None:
        self.datasets = datasets or MAIN_DATASETS
        self.force = force
        self._meta_dir = RESULTS_DIR / "meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    def _deps_met(self, experiment_name: str) -> bool:
        for dep in DEPENDENCIES.get(experiment_name, []):
            exp = EXPERIMENT_REGISTRY[dep]()
            if not exp.is_complete():
                return False
        return True

    def run_experiment(self, name: str) -> None:
        if name not in EXPERIMENT_REGISTRY:
            raise ValueError(f"Unknown experiment: {name}")
        exp = EXPERIMENT_REGISTRY[name]()

        if exp.is_complete() and not self.force:
            print(f"[Skip] {name} already complete. Use --force to re-run.")
            return

        if not self._deps_met(name):
            print(f"[Deps] Running dependencies for {name}...")
            for dep in DEPENDENCIES.get(name, []):
                self.run_experiment(dep)

        print(f"\n{'='*60}")
        print(f"Running experiment: {name}")
        print(f"{'='*60}")

        start = time.time()
        # Some experiments take dataset (singular), others take datasets (plural)
        if name in ("threshold", "scalability", "adaptive"):
            exp.run(dataset=self.datasets[0])
        else:
            exp.run(datasets=self.datasets)
        elapsed = time.time() - start

        self._log_run(name, elapsed)
        print(f"[Done] {name} in {elapsed:.1f}s")

    def run_all(self) -> None:
        for name in EXPERIMENT_ORDER:
            self.run_experiment(name)

    def status(self) -> dict[str, dict]:
        results = {}
        for name in EXPERIMENT_ORDER:
            exp = EXPERIMENT_REGISTRY[name]()
            deps = DEPENDENCIES.get(name, [])
            deps_ok = self._deps_met(name)
            results[name] = {
                "complete": exp.is_complete(),
                "deps": deps,
                "deps_met": deps_ok,
                "ready": deps_ok and not exp.is_complete(),
            }
        return results

    def plot_all(self) -> None:
        """Generate all figures from existing results."""
        print("\n=== Generating all plots ===")

        # Fig.1
        p = RESULTS_DIR / "motivation" / "fig1_data.csv"
        if p.exists():
            plot_waste_ratio_bar(pd.read_csv(p), str(RESULTS_DIR / "motivation"))

        # Fig.2
        p = RESULTS_DIR / "build_speed" / "table2.csv"
        if p.exists():
            plot_speedup_bar(pd.read_csv(p), str(RESULTS_DIR / "build_speed"))

        # Fig.3
        p = RESULTS_DIR / "recall" / "pareto_data.csv"
        if p.exists():
            plot_pareto_scatter(pd.read_csv(p), str(RESULTS_DIR / "recall"))

        # Fig.4
        p = RESULTS_DIR / "classifier" / "roc_data.json"
        if p.exists():
            plot_roc_curves(p, str(RESULTS_DIR / "classifier"))

        # Fig.5
        p = RESULTS_DIR / "threshold" / "fig5_data.csv"
        if p.exists():
            plot_threshold_sensitivity(pd.read_csv(p), str(RESULTS_DIR / "threshold"))

        # Fig.6
        p = RESULTS_DIR / "scalability" / "fig6_data.csv"
        if p.exists():
            plot_scalability_line(pd.read_csv(p), str(RESULTS_DIR / "scalability"))

        # Fig.7
        p = RESULTS_DIR / "adaptive" / "fig7_data.csv"
        if p.exists():
            plot_recall_over_time(pd.read_csv(p), str(RESULTS_DIR / "adaptive"))

        print("=== All plots generated ===")

    def _log_run(self, name: str, elapsed: float) -> None:
        log_path = self._meta_dir / "run_log.json"
        logs = []
        if log_path.exists():
            with open(log_path) as f:
                logs = json.load(f)
        logs.append({
            "experiment": name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "seed": SEED,
            "datasets": self.datasets,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": platform.processor(),
        })
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)
