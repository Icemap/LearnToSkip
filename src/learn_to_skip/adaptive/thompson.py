"""Thompson Sampling for adaptive ef_construction (REQ-E8)."""
import numpy as np

from learn_to_skip.config import TS_EF_ARMS, TS_ALPHA, SEED


class ThompsonSamplingTuner:
    """Multi-armed bandit with Beta prior for selecting ef_construction."""

    def __init__(
        self,
        arms: list[int] | None = None,
        alpha: float = TS_ALPHA,
        seed: int = SEED,
    ) -> None:
        self.arms = arms or TS_EF_ARMS
        self.alpha = alpha
        self._rng = np.random.RandomState(seed)
        # Beta distribution parameters for each arm
        self._successes = np.ones(len(self.arms))
        self._failures = np.ones(len(self.arms))

    def select_arm(self) -> int:
        """Sample from Beta posterior and return the arm (ef_construction value) with highest sample."""
        samples = [
            self._rng.beta(self._successes[i], self._failures[i])
            for i in range(len(self.arms))
        ]
        best_idx = int(np.argmax(samples))
        return self.arms[best_idx]

    def update(self, ef_construction: int, recall: float, normalized_time: float) -> None:
        """Update Beta parameters based on reward.

        Args:
            ef_construction: The arm that was pulled
            recall: recall@10 achieved
            normalized_time: build time / max_build_time (0-1)
        """
        idx = self.arms.index(ef_construction)
        reward = self.alpha * recall - (1 - self.alpha) * normalized_time
        # Convert reward to Bernoulli outcome
        if reward > 0.5:
            self._successes[idx] += 1
        else:
            self._failures[idx] += 1

    def get_stats(self) -> dict:
        return {
            "arms": self.arms,
            "successes": self._successes.tolist(),
            "failures": self._failures.tolist(),
            "expected": (self._successes / (self._successes + self._failures)).tolist(),
        }
