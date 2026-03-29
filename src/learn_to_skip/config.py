"""Centralized configuration for all experiments."""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRACES_DIR = DATA_DIR / "traces"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

SEED = 42

# HNSW construction parameters
M_VALUES = [16, 32, 48]
EF_CONSTRUCTION_VALUES = [100, 200, 400]
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200

# Search parameters
EF_SEARCH_VALUES = [10, 50, 100, 200, 400]
K_VALUES = [1, 10, 100]

# Threshold sweep
THRESHOLD_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# Scalability subsets
SCALABILITY_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]

# Thompson Sampling
TS_EF_ARMS = [50, 100, 200, 400, 800]
TS_ALPHA = 0.7
TS_BATCH_SIZE = 1000
TS_EVAL_QUERIES = 100

# Random skip baseline
RANDOM_SKIP_PROB = 0.3

# Experiment repetitions
N_REPEATS = 3

# Temporal train/test split
TRAIN_FRACTION = 0.7

# Datasets for main experiments
MAIN_DATASETS = ["sift1m", "gist1m", "glove200", "deep1m"]
DEV_DATASETS = ["sift10k"]
TRANSFER_DATASETS = ["sift1m", "gist1m", "glove200", "deep1m"]

# Classifiers
CLASSIFIER_NAMES = ["logistic", "svm", "tree", "xgboost"]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    seed: int = SEED
    m_values: list[int] = field(default_factory=lambda: M_VALUES)
    ef_construction_values: list[int] = field(default_factory=lambda: EF_CONSTRUCTION_VALUES)
    ef_search_values: list[int] = field(default_factory=lambda: EF_SEARCH_VALUES)
    k_values: list[int] = field(default_factory=lambda: K_VALUES)
    threshold_values: list[float] = field(default_factory=lambda: THRESHOLD_VALUES)
    n_repeats: int = N_REPEATS
    datasets: list[str] = field(default_factory=lambda: MAIN_DATASETS)
    classifiers: list[str] = field(default_factory=lambda: CLASSIFIER_NAMES)
