"""Threshold strategy and sweep (REQ-C3)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from learn_to_skip.classifiers.base import BaseSkipClassifier
from learn_to_skip.config import THRESHOLD_VALUES


@dataclass
class ThresholdResult:
    threshold: float
    skip_rate: float
    precision: float
    recall: float


class ThresholdStrategy:
    """Apply a threshold to classifier probabilities."""

    def __init__(self, classifier: BaseSkipClassifier, threshold: float = 0.5) -> None:
        self.classifier = classifier
        self.threshold = threshold

    def should_skip(self, X: np.ndarray) -> np.ndarray:
        """Return bool array: True = skip."""
        proba = self.classifier.predict_proba(X)
        return proba > self.threshold


class ThresholdSweep:
    """Evaluate multiple thresholds on validation data."""

    def __init__(
        self,
        classifier: BaseSkipClassifier,
        thresholds: list[float] | None = None,
    ) -> None:
        self.classifier = classifier
        self.thresholds = thresholds or THRESHOLD_VALUES

    def sweep(self, X: np.ndarray, y: np.ndarray) -> list[ThresholdResult]:
        """Run sweep. y should be label_retained (1=retained, 0=not)."""
        y_skip = 1 - y  # skip = not retained
        proba = self.classifier.predict_proba(X)
        results = []
        for tau in self.thresholds:
            y_pred = (proba > tau).astype(int)
            tp = np.sum((y_pred == 1) & (y_skip == 1))
            fp = np.sum((y_pred == 1) & (y_skip == 0))
            fn = np.sum((y_pred == 0) & (y_skip == 1))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            skip_rate = np.mean(y_pred)
            results.append(ThresholdResult(
                threshold=tau,
                skip_rate=float(skip_rate),
                precision=float(precision),
                recall=float(recall),
            ))
        return results

    def sweep_to_df(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        results = self.sweep(X, y)
        return pd.DataFrame([
            {"threshold": r.threshold, "skip_rate": r.skip_rate,
             "precision": r.precision, "recall": r.recall}
            for r in results
        ])
