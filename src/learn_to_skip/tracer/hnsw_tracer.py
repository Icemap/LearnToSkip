"""Simplified Python HNSW implementation for trace data collection (Path A).

Features are recorded at ENCOUNTER TIME (before full search completes) to avoid
data leakage. Labels are computed post-hoc from the final search results.
"""
import heapq
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from learn_to_skip.config import TRACES_DIR, SEED, TRAIN_FRACTION
from learn_to_skip.features.approx_distance import BaseApproxDistance, RandomProjectionDistance


class HNSWTracer:
    """Build HNSW graph in Python while recording all candidate evaluations."""

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        approx_dist: BaseApproxDistance | None = None,
        seed: int = SEED,
    ) -> None:
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.seed = seed
        self.ml = 1.0 / math.log(M)

        self.approx_dist = approx_dist or RandomProjectionDistance()
        self._rng = np.random.RandomState(seed)

        # Graph storage
        self._data: list[np.ndarray] = []
        self._neighbors: list[dict[int, list[int]]] = []
        self._max_layer: list[int] = []
        self._enter_point: int = -1
        self._max_level: int = -1

        # Trace records
        self._trace_records: list[dict] = []

        # Counters
        self._distance_computations: int = 0

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        self._distance_computations += 1
        return float(np.sum((a - b) ** 2))

    def _random_level(self) -> int:
        return int(-math.log(self._rng.uniform()) * self.ml)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
        trace_ctx: dict | None = None,
    ) -> list[tuple[float, int]]:
        """Search a single layer. If trace_ctx is provided, record features at encounter time.

        trace_ctx should have: insert_id (int), encounter_records (list to append to)
        """
        visited = set(entry_points)
        results: list[tuple[float, int]] = []
        candidates: list[tuple[float, int]] = []
        encounter_counter = 0

        for ep in entry_points:
            d = self._distance(query, self._data[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))
            if len(results) > ef:
                heapq.heappop(results)

            if trace_ctx is not None:
                trace_ctx["encounter_records"].append({
                    "insert_id": trace_ctx["insert_id"],
                    "candidate_id": ep,
                    "candidate_degree": len(self._neighbors[ep].get(layer, [])),
                    "candidate_layer": self._max_layer[ep],
                    "current_layer": layer,
                    "approx_dist": self.approx_dist.estimate(query, self._data[ep]),
                    "candidate_rank_in_beam": encounter_counter,
                    "beam_size": len(results),
                    "inserted_count": trace_ctx["insert_id"],
                    "exact_dist": d,
                })
                encounter_counter += 1

        while candidates:
            d_c, c = heapq.heappop(candidates)
            worst_dist = -results[0][0] if results else float("inf")
            if d_c > worst_dist:
                break

            neighbors = self._neighbors[c].get(layer, [])
            for n in neighbors:
                if n in visited:
                    continue
                visited.add(n)

                # Record features BEFORE computing exact distance
                if trace_ctx is not None:
                    approx_d = self.approx_dist.estimate(query, self._data[n])
                    trace_ctx["encounter_records"].append({
                        "insert_id": trace_ctx["insert_id"],
                        "candidate_id": n,
                        "candidate_degree": len(self._neighbors[n].get(layer, [])),
                        "candidate_layer": self._max_layer[n],
                        "current_layer": layer,
                        "approx_dist": approx_d,
                        "candidate_rank_in_beam": encounter_counter,
                        "beam_size": len(results),
                        "inserted_count": trace_ctx["insert_id"],
                        "exact_dist": 0.0,  # placeholder, filled below
                    })

                d_n = self._distance(query, self._data[n])

                if trace_ctx is not None:
                    # Fill in exact_dist for the record we just added
                    trace_ctx["encounter_records"][-1]["exact_dist"] = d_n
                    encounter_counter += 1

                worst_dist = -results[0][0] if results else float("inf")
                if d_n < worst_dist or len(results) < ef:
                    heapq.heappush(candidates, (d_n, n))
                    heapq.heappush(results, (-d_n, n))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(-d, idx) for d, idx in results]

    def _select_neighbors_simple(
        self, candidates: list[tuple[float, int]], M: int
    ) -> list[int]:
        candidates.sort(key=lambda x: x[0])
        return [idx for _, idx in candidates[:M]]

    def build(self, data: np.ndarray, trace: bool = True) -> None:
        """Insert all vectors and optionally record trace data."""
        n = len(data)
        self.approx_dist.fit(data)

        for i in range(n):
            self._insert_point(data[i], i, trace=trace)
            if (i + 1) % 1000 == 0:
                print(f"  Traced {i + 1}/{n} insertions")

    def _insert_point(self, vec: np.ndarray, idx: int, trace: bool) -> None:
        level = self._random_level()
        self._data.append(vec)
        self._neighbors.append({})
        self._max_layer.append(level)

        if self._enter_point == -1:
            self._enter_point = idx
            self._max_level = level
            for l in range(level + 1):
                self._neighbors[idx][l] = []
            return

        # Greedy search from top to insertion level
        curr_ep = [self._enter_point]
        for l in range(self._max_level, level, -1):
            results = self._search_layer(vec, curr_ep, ef=1, layer=l)
            curr_ep = [results[0][1]] if results else curr_ep

        # Search and connect at each layer from level down to 0
        for l in range(min(level, self._max_level), -1, -1):
            trace_ctx = None
            if trace:
                trace_ctx = {"insert_id": idx, "encounter_records": []}

            results = self._search_layer(
                vec, curr_ep, ef=self.ef_construction, layer=l, trace_ctx=trace_ctx
            )
            M_layer = self.M_max0 if l == 0 else self.M

            # Compute labels post-hoc: which candidates are in top-M by exact distance
            if trace and trace_ctx is not None:
                selected_ids = set(self._select_neighbors_simple(results, M_layer))
                for rec in trace_ctx["encounter_records"]:
                    rec["label_retained"] = rec["candidate_id"] in selected_ids
                self._trace_records.extend(trace_ctx["encounter_records"])

            # Connect
            selected = self._select_neighbors_simple(results, M_layer)
            self._neighbors[idx][l] = selected

            for s in selected:
                if l not in self._neighbors[s]:
                    self._neighbors[s][l] = []
                self._neighbors[s][l].append(idx)
                if len(self._neighbors[s][l]) > M_layer:
                    dists = [(self._distance(self._data[s], self._data[n]), n)
                             for n in self._neighbors[s][l]]
                    self._neighbors[s][l] = self._select_neighbors_simple(dists, M_layer)

            curr_ep = [r[1] for r in results[:1]] if results else curr_ep

        if level > self._max_level:
            self._max_level = level
            self._enter_point = idx

    def get_trace_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._trace_records)

    def save_trace(self, dataset_name: str) -> Path:
        out_dir = TRACES_DIR / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"trace_{self.ef_construction}_{self.M}.parquet"
        df = self.get_trace_df()
        df.to_parquet(path, index=False)
        print(f"  Saved trace: {path} ({len(df)} records)")
        return path

    @property
    def n_points(self) -> int:
        return len(self._data)


def temporal_split_trace(
    df: pd.DataFrame, train_frac: float = TRAIN_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split trace data temporally by insert_id."""
    cutoff = int(df["insert_id"].max() * train_frac)
    train = df[df["insert_id"] <= cutoff].copy()
    test = df[df["insert_id"] > cutoff].copy()
    return train, test
