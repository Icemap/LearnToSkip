"""Feature extraction and engineering (REQ-C1)."""
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class FeatureSet(Enum):
    """Feature combinations for ablation study."""
    FULL = "full"
    NO_APPROX_DIST = "no_approx_dist"
    NO_GRAPH_FEATURES = "no_graph_features"
    NO_QUEUE_FEATURES = "no_queue_features"
    MINIMAL = "minimal"


# Column names in trace data
ALL_FEATURES = [
    "candidate_degree",
    "candidate_layer",
    "current_layer",
    "approx_dist",
    "candidate_rank_in_beam",
    "beam_size",
    "inserted_count",
]

FEATURE_SET_COLUMNS: dict[FeatureSet, list[str]] = {
    FeatureSet.FULL: ALL_FEATURES,
    FeatureSet.NO_APPROX_DIST: [f for f in ALL_FEATURES if f != "approx_dist"],
    FeatureSet.NO_GRAPH_FEATURES: [
        f for f in ALL_FEATURES
        if f not in ("candidate_degree", "candidate_layer", "current_layer")
    ],
    FeatureSet.NO_QUEUE_FEATURES: [
        f for f in ALL_FEATURES if f not in ("candidate_rank_in_beam", "beam_size")
    ],
    FeatureSet.MINIMAL: ["approx_dist", "candidate_rank_in_beam"],
}

LABEL_COLUMN = "label_retained"


class FeatureExtractor:
    """Extract and normalize features from trace data."""

    def __init__(self, feature_set: FeatureSet = FeatureSet.FULL) -> None:
        self.feature_set = feature_set
        self._columns = FEATURE_SET_COLUMNS[feature_set]
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self._columns)

    @property
    def n_features(self) -> int:
        return len(self._columns)

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        """Compute normalization parameters from training data."""
        X = df[self._columns].values.astype(np.float32)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-8] = 1.0  # avoid division by zero
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) with z-score normalized features."""
        assert self._mean is not None, "Call fit() first"
        X = df[self._columns].values.astype(np.float32)
        X = (X - self._mean) / self._std
        y = df[LABEL_COLUMN].values.astype(np.int32)
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def transform_features_only(self, df: pd.DataFrame) -> np.ndarray:
        """Transform without labels (for inference)."""
        assert self._mean is not None, "Call fit() first"
        X = df[self._columns].values.astype(np.float32)
        return (X - self._mean) / self._std

    def save(self, path: Path) -> None:
        np.savez(
            path,
            mean=self._mean,
            std=self._std,
            columns=np.array(self._columns),
            feature_set=self.feature_set.value,
        )

    def load(self, path: Path) -> "FeatureExtractor":
        data = np.load(path, allow_pickle=True)
        self._mean = data["mean"]
        self._std = data["std"]
        self._columns = list(data["columns"])
        self.feature_set = FeatureSet(str(data["feature_set"]))
        return self
