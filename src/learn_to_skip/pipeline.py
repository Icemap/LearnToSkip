"""Shared pipeline utilities for trace → split → fit → train."""
import pandas as pd
import numpy as np

from learn_to_skip.tracer.hnsw_tracer import temporal_split_trace
from learn_to_skip.features.extractor import FeatureExtractor, FeatureSet
from learn_to_skip.features.approx_distance import RandomProjectionDistance
from learn_to_skip.classifiers import get_classifier
from learn_to_skip.classifiers.base import BaseSkipClassifier
from learn_to_skip.config import TRAIN_FRACTION


MAX_TRAIN_SAMPLES = 100_000
MAX_TEST_SAMPLES = 50_000


def _subsample_stratified(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    """Subsample DataFrame while preserving label_retained class balance."""
    if len(df) <= max_n:
        return df
    pos = df[df["label_retained"] == True]
    neg = df[df["label_retained"] == False]
    n_per_class = max_n // 2
    pos_sample = pos.sample(n=min(len(pos), n_per_class), random_state=42)
    neg_sample = neg.sample(n=min(len(neg), n_per_class), random_state=42)
    return pd.concat([pos_sample, neg_sample]).reset_index(drop=True)


def prepare_classifier(
    trace_path: str | None = None,
    trace_df: pd.DataFrame | None = None,
    clf_name: str = "xgboost",
    feature_set: FeatureSet = FeatureSet.FULL,
    train_frac: float = TRAIN_FRACTION,
    max_train: int = MAX_TRAIN_SAMPLES,
    max_test: int = MAX_TEST_SAMPLES,
) -> tuple[BaseSkipClassifier, FeatureExtractor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load trace → temporal split → fit extractor on train → train classifier.

    Subsamples large traces to keep training tractable while preserving
    temporal split integrity (subsample within each split, not across).

    Returns: (clf, extractor, X_train, y_train, X_test, y_test)
    """
    if trace_df is None:
        assert trace_path is not None
        trace_df = pd.read_parquet(trace_path)

    train_df, test_df = temporal_split_trace(trace_df, train_frac)

    # Subsample if too large (stratified by label to preserve class balance)
    train_df = _subsample_stratified(train_df, max_train)
    test_df = _subsample_stratified(test_df, max_test)

    extractor = FeatureExtractor(feature_set)
    X_train, y_train = extractor.fit_transform(train_df)
    X_test, y_test = extractor.transform(test_df)

    clf = get_classifier(clf_name)
    clf.train(X_train, y_train)

    return clf, extractor, X_train, y_train, X_test, y_test
