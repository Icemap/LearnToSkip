"""Integration test: full pipeline on SIFT10K."""
import pytest

from learn_to_skip.orchestrator.runner import ExperimentRunner


@pytest.mark.slow
def test_full_pipeline_sift10k():
    """Run all experiments end-to-end on SIFT10K."""
    runner = ExperimentRunner(datasets=["sift10k"], force=True)
    runner.run_all()
    runner.plot_all()

    # Verify status
    st = runner.status()
    for name, info in st.items():
        assert info["complete"], f"Experiment {name} did not complete"
