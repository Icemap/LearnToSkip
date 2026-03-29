"""Tests for classifiers."""
import numpy as np
import pytest

from learn_to_skip.classifiers import (
    get_classifier, CLASSIFIER_REGISTRY,
    LogisticRegressionClassifier, ThresholdSweep,
)


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    n = 300
    X = rng.randn(n, 7).astype(np.float32)
    y = rng.randint(0, 2, n)  # label_retained
    return X, y


@pytest.mark.parametrize("clf_name", list(CLASSIFIER_REGISTRY.keys()))
def test_classifier_train_predict(clf_name, sample_data):
    X, y = sample_data
    clf = get_classifier(clf_name)
    clf.train(X, y)
    pred = clf.predict(X)
    assert pred.shape == (len(X),)
    assert pred.dtype == bool

    proba = clf.predict_proba(X)
    assert proba.shape == (len(X),)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_classifier_save_load(sample_data, tmp_path):
    X, y = sample_data
    clf = LogisticRegressionClassifier()
    clf.train(X, y)
    pred1 = clf.predict_proba(X)

    clf.save(str(tmp_path / "model.pkl"))
    clf2 = LogisticRegressionClassifier()
    clf2.load(str(tmp_path / "model.pkl"))
    pred2 = clf2.predict_proba(X)

    np.testing.assert_array_almost_equal(pred1, pred2)


def test_evaluate_holdout(sample_data):
    X, y = sample_data
    # Split manually
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]
    clf = LogisticRegressionClassifier()
    metrics = clf.evaluate_holdout(X_train, y_train, X_test, y_test)
    assert 0 <= metrics.auc <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1 <= 1
    assert metrics.train_time_sec > 0
    assert metrics.inference_time_ns > 0


def test_threshold_sweep(sample_data):
    X, y = sample_data
    clf = LogisticRegressionClassifier()
    clf.train(X, y)
    sweep = ThresholdSweep(clf)
    results = sweep.sweep(X, y)
    assert len(results) == 7  # default thresholds
    for r in results:
        assert 0 <= r.skip_rate <= 1
        assert 0 <= r.precision <= 1
        assert 0 <= r.recall <= 1


def test_inference_time(sample_data):
    X, y = sample_data
    clf = LogisticRegressionClassifier()
    clf.train(X, y)
    t = clf.inference_time_per_sample(X)
    assert t > 0  # nanoseconds
