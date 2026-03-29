"""Tests for HNSW tracer."""
import numpy as np
import pytest

from learn_to_skip.tracer.hnsw_tracer import HNSWTracer, temporal_split_trace


def test_tracer_basic():
    rng = np.random.RandomState(42)
    data = rng.randn(200, 32).astype(np.float32)

    tracer = HNSWTracer(dim=32, M=4, ef_construction=20)
    tracer.build(data, trace=True)

    df = tracer.get_trace_df()
    assert len(df) > 0
    assert "candidate_degree" in df.columns
    assert "label_retained" in df.columns
    assert "approx_dist" in df.columns
    assert "insert_id" in df.columns
    assert df["label_retained"].dtype == bool


def test_tracer_columns():
    rng = np.random.RandomState(42)
    data = rng.randn(100, 16).astype(np.float32)

    tracer = HNSWTracer(dim=16, M=4, ef_construction=10)
    tracer.build(data, trace=True)
    df = tracer.get_trace_df()

    expected_cols = {
        "insert_id", "candidate_id", "candidate_degree",
        "candidate_layer", "current_layer", "approx_dist",
        "candidate_rank_in_beam", "beam_size", "inserted_count",
        "label_retained", "exact_dist",
    }
    assert expected_cols.issubset(set(df.columns))


def test_tracer_no_leakage():
    """Verify that encounter-time features don't encode the label."""
    rng = np.random.RandomState(42)
    data = rng.randn(300, 16).astype(np.float32)

    tracer = HNSWTracer(dim=16, M=4, ef_construction=20)
    tracer.build(data, trace=True)
    df = tracer.get_trace_df()

    # candidate_rank_in_beam is per _search_layer() call (resets per layer).
    # Within each (insert_id, current_layer) group it should be monotonically non-decreasing.
    for insert_id in df["insert_id"].unique()[:10]:
        subset = df[df["insert_id"] == insert_id]
        for layer in subset["current_layer"].unique():
            layer_sub = subset[subset["current_layer"] == layer]
            ranks = layer_sub["candidate_rank_in_beam"].values
            assert ranks[0] == 0, f"First rank should be 0, got {ranks[0]}"
            assert all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))


def test_temporal_split():
    rng = np.random.RandomState(42)
    data = rng.randn(200, 16).astype(np.float32)

    tracer = HNSWTracer(dim=16, M=4, ef_construction=10)
    tracer.build(data, trace=True)
    df = tracer.get_trace_df()

    train, test = temporal_split_trace(df, train_frac=0.7)
    assert len(train) > 0
    assert len(test) > 0
    assert train["insert_id"].max() < test["insert_id"].min()


def test_tracer_save(tmp_path, monkeypatch):
    import learn_to_skip.config as cfg
    monkeypatch.setattr(cfg, "TRACES_DIR", tmp_path / "traces")

    rng = np.random.RandomState(42)
    data = rng.randn(100, 16).astype(np.float32)

    tracer = HNSWTracer(dim=16, M=4, ef_construction=10)
    tracer.build(data, trace=True)
    import learn_to_skip.tracer.hnsw_tracer as tracer_mod
    monkeypatch.setattr(tracer_mod, "TRACES_DIR", tmp_path / "traces")
    path = tracer.save_trace("test_ds")
    assert path.exists()
