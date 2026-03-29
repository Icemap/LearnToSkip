"""Linear SVM classifier."""
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

from learn_to_skip.classifiers.base import BaseSkipClassifier
from learn_to_skip.config import SEED


class LinearSVMClassifier(BaseSkipClassifier):
    def __init__(self, seed: int = SEED) -> None:
        super().__init__(seed)
        base_svc = SGDClassifier(
            loss="hinge", class_weight="balanced", max_iter=1000,
            random_state=seed, tol=1e-3,
        )
        self._model = CalibratedClassifierCV(base_svc, cv=2)

    @property
    def name(self) -> str:
        return "svm"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        y_skip = 1 - y
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
        base_svc = SGDClassifier(
            loss="hinge", class_weight="balanced", max_iter=1000,
            random_state=self.seed, tol=1e-3,
        )
        return CalibratedClassifierCV(base_svc, cv=2)
