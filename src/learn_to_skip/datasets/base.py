"""Abstract base class for datasets."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from learn_to_skip.config import RAW_DIR, PROCESSED_DIR


@dataclass
class DatasetMetadata:
    name: str
    dim: int
    metric: str  # "l2" or "cosine"
    n_train: int
    n_query: int


class BaseDataset(ABC):
    """Abstract base for all vector datasets."""

    def __init__(self) -> None:
        self._raw_dir = RAW_DIR / self.name
        self._processed_dir = PROCESSED_DIR / self.name
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def download(self) -> None:
        """Download raw data files."""
        ...

    @abstractmethod
    def load_train(self) -> np.ndarray:
        """Return (n, dim) float32 array of base vectors."""
        ...

    @abstractmethod
    def load_query(self) -> np.ndarray:
        """Return (nq, dim) float32 array of query vectors."""
        ...

    @abstractmethod
    def metadata(self) -> DatasetMetadata: ...

    def load_groundtruth(self, k: int = 100) -> np.ndarray:
        """Return (nq, k) int32 ground-truth nearest neighbor IDs."""
        gt_path = self._processed_dir / f"groundtruth_k{k}.npy"
        if gt_path.exists():
            return np.load(gt_path)
        gt = self._compute_groundtruth(k)
        np.save(gt_path, gt)
        return gt

    def _compute_groundtruth(self, k: int) -> np.ndarray:
        """Brute-force KNN computation."""
        from sklearn.neighbors import NearestNeighbors

        train = self.load_train()
        query = self.load_query()
        meta = self.metadata()
        metric = "euclidean" if meta.metric == "l2" else "cosine"
        nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="brute")
        nn.fit(train)
        _, indices = nn.kneighbors(query)
        return indices.astype(np.int32)

    def ensure_available(self) -> None:
        """Download if not already present."""
        if not any(self._raw_dir.iterdir()):
            self.download()
