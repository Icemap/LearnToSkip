"""Logistic Regression classifier."""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from learn_to_skip.classifiers.base import BaseSkipClassifier
from learn_to_skip.config import SEED


class LogisticRegressionClassifier(BaseSkipClassifier):
    def __init__(self, seed: int = SEED) -> None:
        super().__init__(seed)
        self._model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed
        )

    @property
    def name(self) -> str:
        return "logistic"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        y_skip = 1 - y  # retained -> skip
        self._model.fit(X, y_skip)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(bool)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)

    def _get_sklearn_estimator(self):
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=self.seed
        )
