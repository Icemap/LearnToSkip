"""Abstract base for index builders (REQ-E0)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BuiltIndex:
    index: object  # hnswlib.Index or compatible
    build_time_seconds: float
    distance_computations: int
    skipped_computations: int
    classifier_overhead_seconds: float = 0.0
    n_classifier_calls: int = 0

    @property
    def skip_rate(self) -> float:
        total = self.distance_computations + self.skipped_computations
        return self.skipped_computations / total if total > 0 else 0.0

    @property
    def speedup(self) -> float:
        """Placeholder; actual speedup computed by comparing to vanilla."""
        return 0.0


class BaseBuilder(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def build(
        self, data: np.ndarray, M: int, ef_construction: int, metric: str = "l2"
    ) -> BuiltIndex: ...
