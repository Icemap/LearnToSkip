"""Decision Tree classifier."""
import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.model_selection import GridSearchCV

from learn_to_skip.classifiers.base import BaseSkipClassifier
from learn_to_skip.config import SEED


class DecisionTreeClassifier(BaseSkipClassifier):
    def __init__(self, seed: int = SEED) -> None:
        super().__init__(seed)
        self._model = SklearnDT(
            class_weight="balanced", random_state=seed, max_depth=5
        )
        self._best_depth = 5

    @property
    def name(self) -> str:
        return "tree"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        y_skip = 1 - y
        # Grid search over max_depth — subsample for speed if large
        n = len(X)
        if n > 50_000:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(n, size=50_000, replace=False)
            X_grid, y_grid = X[idx], y_skip[idx]
        else:
            X_grid, y_grid = X, y_skip
        grid = GridSearchCV(
            SklearnDT(class_weight="balanced", random_state=self.seed),
            param_grid={"max_depth": [3, 5, 7]},
            cv=2,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_grid, y_grid)
        self._best_depth = grid.best_params_["max_depth"]
        # Refit on full data with best depth
        self._model = SklearnDT(
            class_weight="balanced", random_state=self.seed, max_depth=self._best_depth,
        )
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
        return SklearnDT(
            class_weight="balanced", random_state=self.seed, max_depth=self._best_depth
        )
