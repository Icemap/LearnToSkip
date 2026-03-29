"""All plot functions for the paper (REQ-V1)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from learn_to_skip.visualization.style import (
    setup_style, METHOD_COLORS, CLASSIFIER_COLORS, CONFIG_COLORS,
)


def _save_fig(fig: plt.Figure, output_dir: str | Path, name: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png to {output_dir}")


def plot_waste_ratio_bar(data: pd.DataFrame, output_dir: str) -> None:
    """Fig.1: Candidate waste ratio per dataset."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(data["dataset"], data["waste_ratio"], color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Waste Ratio")
    ax.set_xlabel("Dataset")
    ax.set_title("Candidate Neighbor Waste Ratio During HNSW Construction")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, data["waste_ratio"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig1_waste_ratio")


def plot_speedup_bar(data: pd.DataFrame, output_dir: str) -> None:
    """Fig.2: Build speedup bar chart."""
    setup_style()
    # Filter to default params for cleaner plot
    methods = data["method"].unique()
    datasets = data["dataset"].unique()

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 2), 5))
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    speedup_col = "dist_speedup" if "dist_speedup" in data.columns else "wall_speedup" if "wall_speedup" in data.columns else "speedup"
    for i, method in enumerate(methods):
        subset = data[data["method"] == method]
        vals = []
        for ds in datasets:
            v = subset[subset["dataset"] == ds][speedup_col].mean()
            vals.append(v if not np.isnan(v) else 1.0)
        color = METHOD_COLORS.get(method, f"C{i}")
        ax.bar(x + i * width, vals, width, label=method, color=color, edgecolor="black", linewidth=0.3)

    ax.set_ylabel("Speedup (×)")
    ax.set_xlabel("Dataset")
    ax.set_title("HNSW Construction Speedup")
    ax.set_xticks(x + width * len(methods) / 2)
    ax.set_xticklabels(datasets)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig2_speedup")


def plot_pareto_scatter(data: pd.DataFrame, output_dir: str) -> None:
    """Fig.3: Recall vs Speedup Pareto scatter."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    speedup_col = "dist_speedup" if "dist_speedup" in data.columns else "speedup"
    for method in data["method"].unique():
        subset = data[data["method"] == method]
        color = METHOD_COLORS.get(method, None)
        ax.scatter(subset[speedup_col], subset["recall"], label=method, color=color, s=80, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Distance Computation Speedup (×)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall vs. Speedup Pareto Front")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig3_pareto")


def plot_roc_curves(data_path: str | Path, output_dir: str) -> None:
    """Fig.4: ROC curves per dataset."""
    setup_style()
    data_path = Path(data_path)
    with open(data_path) as f:
        roc_data = json.load(f)

    datasets = sorted(set(r["dataset"] for r in roc_data))
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), squeeze=False)

    for col, ds in enumerate(datasets):
        ax = axes[0, col]
        ds_data = [r for r in roc_data if r["dataset"] == ds]
        for r in ds_data:
            color = CLASSIFIER_COLORS.get(r["classifier"], None)
            ax.plot(r["fpr"], r["tpr"], label=f'{r["classifier"]} (AUC={r["auc"]:.3f})',
                    color=color, linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {ds}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, output_dir, "fig4_roc")


def plot_threshold_sensitivity(data: pd.DataFrame, output_dir: str) -> None:
    """Fig.5: Dual Y-axis — threshold vs speedup and recall."""
    setup_style()
    fig, ax1 = plt.subplots(figsize=(6, 4))

    speedup_col = "dist_speedup" if "dist_speedup" in data.columns else "speedup"
    ax1.set_xlabel("Threshold (τ)")
    ax1.set_ylabel("Dist. Computation Speedup (×)", color="#d62728")
    ax1.plot(data["threshold"], data[speedup_col], "o-", color="#d62728", label="Speedup", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="#d62728")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Recall@10", color="#1f77b4")
    ax2.plot(data["threshold"], data["recall_at_10"], "s--", color="#1f77b4", label="Recall@10", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    ax1.set_title("Threshold Sensitivity Analysis")
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig5_sensitivity")


def plot_scalability_line(data: pd.DataFrame, output_dir: str) -> None:
    """Fig.6: Build time vs data size."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    for method in data["method"].unique():
        subset = data[data["method"] == method].sort_values("size")
        color = METHOD_COLORS.get(method, None)
        ax.plot(subset["size"], subset["build_time_sec"], "o-", label=method, color=color, linewidth=2)

    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Build Time (seconds)")
    ax.set_title("Construction Time Scalability")
    ax.legend()
    ax.set_xscale("log")
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig6_scalability")


def plot_recall_over_time(data: pd.DataFrame, output_dir: str) -> None:
    """Fig.7: Recall over streaming batches for adaptive configs."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    for config in data["config"].unique():
        subset = data[data["config"] == config].sort_values("batch")
        color = CONFIG_COLORS.get(config, None)
        ax.plot(subset["n_inserted"], subset["recall_at_10"], label=config, color=color, linewidth=1.5)

    ax.set_xlabel("Number of Vectors Inserted")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall Over Streaming Insertion")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, output_dir, "fig7_recall_over_time")
