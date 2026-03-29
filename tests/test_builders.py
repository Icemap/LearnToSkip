"""Tests for index builders."""
import numpy as np
import pytest

from learn_to_skip.builders import (
    VanillaHNSWBuilder, RandomSkipBuilder,
    DistanceThresholdBuilder, LearnedSkipBuilder,
    CppLearnedSkipBuilder,
)
from learn_to_skip.classifiers import LogisticRegressionClassifier
from learn_to_skip.features.extractor import FeatureExtractor, FeatureSet


@pytest.fixture
def sample_vectors():
    rng = np.random.RandomState(42)
    return rng.randn(200, 16).astype(np.float32)


def test_vanilla_build(sample_vectors):
    builder = VanillaHNSWBuilder()
    result = builder.build(sample_vectors, M=8, ef_construction=50)
    assert result.build_time_seconds > 0
    assert result.index is not None
    result.index.set_ef(50)
    labels, dists = result.index.knn_query(sample_vectors[:5], k=5)
    assert labels.shape == (5, 5)


def test_random_skip_build(sample_vectors):
    builder = RandomSkipBuilder(skip_prob=0.3)
    result = builder.build(sample_vectors, M=4, ef_construction=20)
    assert result.build_time_seconds > 0
    assert result.skipped_computations > 0
    # Verify knn_query works on PythonHNSWIndex
    result.index.set_ef(20)
    labels, dists = result.index.knn_query(sample_vectors[:3], k=3)
    assert labels.shape == (3, 3)


def test_distance_threshold_build(sample_vectors):
    builder = DistanceThresholdBuilder()
    result = builder.build(sample_vectors, M=4, ef_construction=20)
    assert result.build_time_seconds > 0
    assert result.skipped_computations > 0


def test_learned_skip_build(sample_vectors):
    """Test real per-candidate classifier-guided skipping."""
    from learn_to_skip.tracer.hnsw_tracer import HNSWTracer, temporal_split_trace

    # Generate trace data to train the classifier
    tracer = HNSWTracer(dim=16, M=4, ef_construction=20)
    tracer.build(sample_vectors[:100], trace=True)
    trace_df = tracer.get_trace_df()
    train_df, test_df = temporal_split_trace(trace_df)

    extractor = FeatureExtractor(FeatureSet.FULL)
    X_train, y_train = extractor.fit_transform(train_df)

    clf = LogisticRegressionClassifier()
    clf.train(X_train, y_train)

    builder = LearnedSkipBuilder(
        classifier=clf, extractor=extractor, threshold=0.7,
    )
    result = builder.build(sample_vectors[:100], M=4, ef_construction=20)
    assert result.build_time_seconds > 0
    assert result.skipped_computations > 0
    assert result.classifier_overhead_seconds > 0
    assert result.n_classifier_calls > 0

    # Verify knn_query works
    result.index.set_ef(20)
    labels, dists = result.index.knn_query(sample_vectors[:3], k=3)
    assert labels.shape == (3, 3)


def test_cpp_learned_skip_universal():
    """Test C++ skip builder in universal mode with enough data to trigger skips."""
    rng = np.random.RandomState(42)
    data = rng.randn(2000, 128).astype(np.float32)
    builder = CppLearnedSkipBuilder(threshold=0.7, proj_dim=16, mode="universal")
    result = builder.build(data, M=8, ef_construction=50)
    assert result.build_time_seconds > 0
    assert result.distance_computations > 0
    assert result.skipped_computations > 0
    assert result.skip_rate > 0

    # Verify knn_query works
    result.index.set_ef(50)
    labels, dists = result.index.knn_query(data[:5], k=5)
    assert labels.shape == (5, 5)


def test_cpp_learned_skip_online(sample_vectors):
    """Test C++ skip builder in online mode."""
    builder = CppLearnedSkipBuilder(
        threshold=0.7, proj_dim=8, mode="online", train_fraction=0.3)
    result = builder.build(sample_vectors, M=8, ef_construction=50)
    assert result.build_time_seconds > 0
    assert result.distance_computations > 0

    result.index.set_ef(50)
    labels, dists = result.index.knn_query(sample_vectors[:5], k=5)
    assert labels.shape == (5, 5)


def test_cpp_skip_backward_compatible():
    """Vanilla hnswlib API still works unchanged."""
    import hnswlib
    data = np.random.randn(100, 32).astype(np.float32)
    idx = hnswlib.Index(space="l2", dim=32)
    idx.init_index(max_elements=100, M=8, ef_construction=50)
    idx.add_items(data)
    assert idx.get_current_count() == 100
    idx.set_ef(50)
    labels, dists = idx.knn_query(data[:3], k=5)
    assert labels.shape == (3, 5)
