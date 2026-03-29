"""XGBoost classifier."""
import pickle

import numpy as np
import xgboost as xgb

from learn_to_skip.classifiers.base import BaseSkipClassifier
from learn_to_skip.config import SEED


class XGBoostClassifier(BaseSkipClassifier):
    def __init__(self, seed: int = SEED) -> None:
        super().__init__(seed)
        self._model = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=30,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

    @property
    def name(self) -> str:
        return "xgboost"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        y_skip = 1 - y
        # Handle class imbalance via scale_pos_weight
        n_pos = np.sum(y_skip == 1)
        n_neg = np.sum(y_skip == 0)
        scale = n_neg / max(n_pos, 1)
        self._model.set_params(scale_pos_weight=scale)
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
        return xgb.XGBClassifier(
            max_depth=3,
            n_estimators=30,
            random_state=self.seed,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
