"""Abstract base for experiments."""
from abc import ABC, abstractmethod
from pathlib import Path

from learn_to_skip.config import RESULTS_DIR


class BaseExperiment(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def output_dir(self) -> Path:
        d = RESULTS_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @abstractmethod
    def run(self, **kwargs) -> None: ...

    def is_complete(self) -> bool:
        """Check if outputs already exist."""
        return False
