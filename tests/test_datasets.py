"""Tests for dataset loading."""
import numpy as np
import pytest

from learn_to_skip.datasets import get_dataset, Sift10KDataset


def test_sift10k_load():
    ds = Sift10KDataset()
    ds.ensure_available()
    train = ds.load_train()
    query = ds.load_query()
    meta = ds.metadata()
    assert train.shape == (10_000, 128)
    assert query.shape == (100, 128)
    assert train.dtype == np.float32
    assert meta.dim == 128
    assert meta.metric == "l2"


def test_sift10k_groundtruth():
    ds = Sift10KDataset()
    gt = ds.load_groundtruth(k=10)
    assert gt.shape == (100, 10)
    assert gt.dtype == np.int32
    # IDs should be within range
    assert gt.max() < 10_000
    assert gt.min() >= 0


def test_get_dataset_registry():
    ds = get_dataset("sift10k")
    assert isinstance(ds, Sift10KDataset)


def test_get_dataset_invalid():
    with pytest.raises(ValueError):
        get_dataset("nonexistent")
