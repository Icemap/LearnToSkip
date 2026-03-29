"""Abstract base class for skip classifiers (REQ-C2)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from learn_to_skip.config import SEED


@dataclass
class ClassifierMetrics:
    precision: float
    recall: float
    f1: float
    auc: float
    train_time_sec: float
    model_size_bytes: int
    inference_time_ns: float  # per sample


class BaseSkipClassifier(ABC):
    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed
        self._train_time: float = 0.0
        self._model_size: int = 0

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return bool array: True = skip this candidate."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of skip."""
        ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    def inference_time_per_sample(self, X: np.ndarray | None = None) -> float:
        """Return average inference time in nanoseconds."""
        if X is None:
            X = np.random.randn(1000, 7).astype(np.float32)
        # Warmup
        self.predict(X[:10])
        start = time.perf_counter_ns()
        n_iters = 10
        for _ in range(n_iters):
            self.predict(X)
        elapsed = time.perf_counter_ns() - start
        return elapsed / (n_iters * len(X))

    def evaluate_cv(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> ClassifierMetrics:
        """Run cross-validation and return metrics."""
        # Train timing
        t0 = time.time()
        self.train(X, y)
        train_time = time.time() - t0

        # CV predictions
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        # y_skip: 1 = not retained (skip), 0 = retained
        y_skip = 1 - y  # Invert: label_retained -> label_skip
        y_pred_proba = cross_val_predict(
            self._get_sklearn_estimator(), X, y_skip, cv=skf, method="predict_proba"
        )[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Model size
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            self.save(f.name)
            model_size = os.path.getsize(f.name)
            os.unlink(f.name)

        inf_time = self.inference_time_per_sample(X[:1000] if len(X) > 1000 else X)

        return ClassifierMetrics(
            precision=float(precision_score(y_skip, y_pred, zero_division=0)),
            recall=float(recall_score(y_skip, y_pred, zero_division=0)),
            f1=float(f1_score(y_skip, y_pred, zero_division=0)),
            auc=float(roc_auc_score(y_skip, y_pred_proba)) if len(np.unique(y_skip)) > 1 else 0.0,
            train_time_sec=train_time,
            model_size_bytes=model_size,
            inference_time_ns=inf_time,
        )

    def evaluate_holdout(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ClassifierMetrics:
        """Train on train set, evaluate on held-out test set (temporal split)."""
        # y is label_retained (1=keep, 0=skip). Invert to label_skip.
        y_train_skip = 1 - y_train
        y_test_skip = 1 - y_test

        t0 = time.time()
        self.train(X_train, y_train)
        train_time = time.time() - t0

        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            self.save(f.name)
            model_size = os.path.getsize(f.name)
            os.unlink(f.name)

        inf_time = self.inference_time_per_sample(X_test[:1000] if len(X_test) > 1000 else X_test)

        return ClassifierMetrics(
            precision=float(precision_score(y_test_skip, y_pred, zero_division=0)),
            recall=float(recall_score(y_test_skip, y_pred, zero_division=0)),
            f1=float(f1_score(y_test_skip, y_pred, zero_division=0)),
            auc=float(roc_auc_score(y_test_skip, y_pred_proba)) if len(np.unique(y_test_skip)) > 1 else 0.0,
            train_time_sec=train_time,
            model_size_bytes=model_size,
            inference_time_ns=inf_time,
        )

    @abstractmethod
    def _get_sklearn_estimator(self):
        """Return a fresh sklearn estimator for CV."""
        ...
