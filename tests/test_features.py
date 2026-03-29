"""Tests for feature extraction and approximate distance."""
import numpy as np
import pandas as pd
import pytest

from learn_to_skip.features.approx_distance import RandomProjectionDistance, SimHashDistance
from learn_to_skip.features.extractor import FeatureExtractor, FeatureSet, ALL_FEATURES, LABEL_COLUMN


@pytest.fixture
def sample_trace_df():
    rng = np.random.RandomState(42)
    n = 500
    return pd.DataFrame({
        "candidate_degree": rng.randint(0, 20, n),
        "candidate_layer": rng.randint(0, 3, n),
        "current_layer": rng.randint(0, 3, n),
        "approx_dist": rng.rand(n).astype(np.float32),
        "candidate_rank_in_beam": rng.randint(0, 50, n),
        "beam_size": rng.randint(10, 100, n),
        "inserted_count": rng.randint(0, 1000, n),
        "label_retained": rng.randint(0, 2, n).astype(bool),
    })


def test_random_projection():
    rng = np.random.RandomState(42)
    data = rng.randn(100, 128).astype(np.float32)
    rp = RandomProjectionDistance()
    rp.fit(data)
    d = rp.estimate(data[0], data[1])
    assert isinstance(d, float)
    assert d >= 0

    d_batch = rp.estimate_batch(data[0], data[1:10])
    assert d_batch.shape == (9,)
    assert np.all(d_batch >= 0)


def test_simhash():
    rng = np.random.RandomState(42)
    data = rng.randn(100, 128).astype(np.float32)
    sh = SimHashDistance()
    sh.fit(data)
    d = sh.estimate(data[0], data[1])
    assert 0.0 <= d <= 1.0

    d_batch = sh.estimate_batch(data[0], data[1:10])
    assert d_batch.shape == (9,)


def test_feature_extractor_full(sample_trace_df):
    ext = FeatureExtractor(FeatureSet.FULL)
    X, y = ext.fit_transform(sample_trace_df)
    assert X.shape == (500, 7)
    assert y.shape == (500,)
    assert abs(X.mean()) < 0.5


def test_feature_extractor_minimal(sample_trace_df):
    ext = FeatureExtractor(FeatureSet.MINIMAL)
    X, y = ext.fit_transform(sample_trace_df)
    assert X.shape == (500, 2)


def test_feature_extractor_save_load(sample_trace_df, tmp_path):
    ext = FeatureExtractor(FeatureSet.FULL)
    ext.fit(sample_trace_df)
    ext.save(tmp_path / "extractor.npz")

    ext2 = FeatureExtractor()
    ext2.load(tmp_path / "extractor.npz")
    X1, _ = ext.transform(sample_trace_df)
    X2, _ = ext2.transform(sample_trace_df)
    np.testing.assert_array_almost_equal(X1, X2)
