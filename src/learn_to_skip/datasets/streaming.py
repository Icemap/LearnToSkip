"""Streaming data generator with concept drift simulation (REQ-E8)."""
from typing import Iterator

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from learn_to_skip.config import SEED


class StreamingDataGenerator:
    """Simulate concept drift by partitioning data into clusters and streaming them sequentially."""

    def __init__(
        self,
        data: np.ndarray,
        n_clusters: int = 10,
        batch_size: int = 1000,
        seed: int = SEED,
    ) -> None:
        self.data = data
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.seed = seed
        self._cluster_labels: np.ndarray | None = None

    def _partition(self) -> None:
        """Cluster the data to simulate distinct distributions."""
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, random_state=self.seed, batch_size=1024
        )
        self._cluster_labels = kmeans.fit_predict(self.data)

    def stream(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (batch_vectors, batch_cluster_labels) in cluster-sequential order."""
        if self._cluster_labels is None:
            self._partition()
        assert self._cluster_labels is not None

        rng = np.random.RandomState(self.seed)
        # Iterate clusters in order (simulates distribution shift)
        for cluster_id in range(self.n_clusters):
            mask = self._cluster_labels == cluster_id
            cluster_data = self.data[mask]
            cluster_labels = self._cluster_labels[mask]
            # Shuffle within cluster
            perm = rng.permutation(len(cluster_data))
            cluster_data = cluster_data[perm]
            cluster_labels = cluster_labels[perm]
            # Yield batches
            for start in range(0, len(cluster_data), self.batch_size):
                end = min(start + self.batch_size, len(cluster_data))
                yield cluster_data[start:end], cluster_labels[start:end]

    @property
    def total_vectors(self) -> int:
        return len(self.data)
