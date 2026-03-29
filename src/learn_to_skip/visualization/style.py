"""Global plot style and color scheme."""
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Method colors (consistent across all figures)
METHOD_COLORS = {
    "Vanilla-HNSW": "#1f77b4",
    "Random-Skip": "#ff7f0e",
    "Dist-Threshold": "#2ca02c",
    "LearnToSkip-logistic": "#d62728",
    "LearnToSkip-svm": "#9467bd",
    "LearnToSkip-tree": "#8c564b",
    "LearnToSkip-xgboost": "#e377c2",
}

# Classifier colors for ROC
CLASSIFIER_COLORS = {
    "logistic": "#d62728",
    "svm": "#9467bd",
    "tree": "#8c564b",
    "xgboost": "#e377c2",
}

# Config colors for adaptive experiment
CONFIG_COLORS = {
    "Vanilla+Fixed": "#1f77b4",
    "Vanilla+TS": "#ff7f0e",
    "LearnToSkip+Fixed": "#d62728",
    "LearnToSkip+TS": "#e377c2",
}


def setup_style() -> None:
    """Set up publication-quality plot style."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })
    # Try Times New Roman, fall back to serif
    try:
        matplotlib.font_manager.findfont("Times New Roman")
        plt.rcParams["font.family"] = "Times New Roman"
    except Exception:
        plt.rcParams["font.family"] = "serif"
