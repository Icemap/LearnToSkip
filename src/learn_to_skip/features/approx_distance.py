"""Approximate distance computation modules (REQ-D1.3)."""
from abc import ABC, abstractmethod

import numpy as np

from learn_to_skip.config import SEED


class BaseApproxDistance(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> None: ...

    @abstractmethod
    def estimate(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float: ...

    @abstractmethod
    def estimate_batch(self, vec_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
        """Estimate distances from vec_a to each row in vecs_b."""
        ...


class RandomProjectionDistance(BaseApproxDistance):
    """Project to low dimension then compute L2."""

    def __init__(self, target_dim: int = 8, seed: int = SEED) -> None:
        self.target_dim = target_dim
        self.seed = seed
        self._projection: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        rng = np.random.RandomState(self.seed)
        dim = data.shape[1]
        self._projection = rng.randn(dim, self.target_dim).astype(np.float32)
        self._projection /= np.linalg.norm(self._projection, axis=0, keepdims=True)

    def estimate(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        assert self._projection is not None
        a_proj = vec_a @ self._projection
        b_proj = vec_b @ self._projection
        return float(np.sum((a_proj - b_proj) ** 2))

    def estimate_batch(self, vec_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
        assert self._projection is not None
        a_proj = vec_a @ self._projection
        b_proj = vecs_b @ self._projection
        return np.sum((a_proj - b_proj) ** 2, axis=1)


class SimHashDistance(BaseApproxDistance):
    """1-bit SimHash then Hamming distance."""

    def __init__(self, n_bits: int = 64, seed: int = SEED) -> None:
        self.n_bits = n_bits
        self.seed = seed
        self._hyperplanes: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        rng = np.random.RandomState(self.seed)
        dim = data.shape[1]
        self._hyperplanes = rng.randn(dim, self.n_bits).astype(np.float32)

    def _hash(self, vecs: np.ndarray) -> np.ndarray:
        """Compute binary hash. Returns bool array of shape (..., n_bits)."""
        assert self._hyperplanes is not None
        return (vecs @ self._hyperplanes) > 0

    def estimate(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        h_a = self._hash(vec_a.reshape(1, -1))[0]
        h_b = self._hash(vec_b.reshape(1, -1))[0]
        hamming = np.sum(h_a != h_b)
        return float(hamming / self.n_bits)

    def estimate_batch(self, vec_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
        h_a = self._hash(vec_a.reshape(1, -1))  # (1, n_bits)
        h_b = self._hash(vecs_b)  # (m, n_bits)
        hamming = np.sum(h_a != h_b, axis=1)
        return hamming.astype(np.float32) / self.n_bits
