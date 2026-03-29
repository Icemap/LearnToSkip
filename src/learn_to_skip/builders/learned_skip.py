"""LearnToSkip builder - real classifier-guided per-candidate pruning in Python HNSW."""
import heapq
import math
import time

import numpy as np

from learn_to_skip.builders.base import BaseBuilder, BuiltIndex
from learn_to_skip.classifiers.base import BaseSkipClassifier
from learn_to_skip.features.extractor import FeatureExtractor
from learn_to_skip.features.approx_distance import BaseApproxDistance, RandomProjectionDistance
from learn_to_skip.config import SEED


class PythonHNSWIndex:
    """Adapter wrapping a Python-built HNSW graph for knn_query compatibility."""

    def __init__(self, data: np.ndarray, neighbors: list[dict[int, list[int]]],
                 max_layer: list[int], enter_point: int, max_level: int) -> None:
        self._data = data
        self._neighbors = neighbors
        self._max_layer = max_layer
        self._enter_point = enter_point
        self._max_level = max_level
        self._ef = 200

    def set_ef(self, ef: int) -> None:
        self._ef = ef

    def knn_query(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Standard HNSW search (no skipping) for recall evaluation."""
        all_labels = []
        all_dists = []
        for q in queries:
            curr_ep = [self._enter_point]
            for l in range(self._max_level, 0, -1):
                results = self._search_layer(q, curr_ep, ef=1, layer=l)
                curr_ep = [results[0][1]] if results else curr_ep
            results = self._search_layer(q, curr_ep, ef=max(self._ef, k), layer=0)
            results.sort(key=lambda x: x[0])
            top_k = results[:k]
            labels = [idx for _, idx in top_k]
            dists = [d for d, _ in top_k]
            while len(labels) < k:
                labels.append(-1)
                dists.append(float("inf"))
            all_labels.append(labels)
            all_dists.append(dists)
        return np.array(all_labels, dtype=np.int64), np.array(all_dists, dtype=np.float32)

    def _search_layer(self, query: np.ndarray, entry_points: list[int],
                      ef: int, layer: int) -> list[tuple[float, int]]:
        visited = set(entry_points)
        results: list[tuple[float, int]] = []
        candidates: list[tuple[float, int]] = []

        for ep in entry_points:
            d = float(np.sum((query - self._data[ep]) ** 2))
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))
            if len(results) > ef:
                heapq.heappop(results)

        while candidates:
            d_c, c = heapq.heappop(candidates)
            worst_dist = -results[0][0] if results else float("inf")
            if d_c > worst_dist:
                break
            for n in self._neighbors[c].get(layer, []):
                if n in visited:
                    continue
                visited.add(n)
                d_n = float(np.sum((query - self._data[n]) ** 2))
                worst_dist = -results[0][0] if results else float("inf")
                if d_n < worst_dist or len(results) < ef:
                    heapq.heappush(candidates, (d_n, n))
                    heapq.heappush(results, (-d_n, n))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(-d, idx) for d, idx in results]


class LearnedSkipBuilder(BaseBuilder):
    """Build HNSW with real per-candidate classifier-guided pruning.

    Uses batched classifier inference: collects all neighbor candidates from
    one beam-pop iteration, batch-predicts skip probabilities, then processes.
    """

    BATCH_SIZE = 32  # Batch size for classifier inference

    def __init__(
        self,
        classifier: BaseSkipClassifier,
        extractor: FeatureExtractor,
        approx_dist: BaseApproxDistance | None = None,
        threshold: float = 0.7,
        seed: int = SEED,
    ) -> None:
        self.classifier = classifier
        self.extractor = extractor
        self.approx_dist = approx_dist or RandomProjectionDistance()
        self.threshold = threshold
        self.seed = seed
        # Pre-extract normalization params and feature names for speed
        self._mean = extractor._mean
        self._std = extractor._std
        self._feature_names = extractor.feature_names
        self._n_features = extractor.n_features

    @property
    def name(self) -> str:
        return f"LearnToSkip-{self.classifier.name}"

    def build(
        self, data: np.ndarray, M: int, ef_construction: int, metric: str = "l2"
    ) -> BuiltIndex:
        n, dim = data.shape
        ml = 1.0 / math.log(M)
        M_max0 = 2 * M
        rng = np.random.RandomState(self.seed)

        self.approx_dist.fit(data)

        # Pre-project all data for fast approx_dist
        proj = self.approx_dist._projection  # (dim, target_dim)
        data_proj = data @ proj if proj is not None else None

        # Graph storage
        neighbors: list[dict[int, list[int]]] = []
        max_layer: list[int] = []
        enter_point = -1
        max_level = -1

        # Counters
        distance_computations = 0
        skipped_computations = 0
        classifier_overhead_ns = 0
        n_classifier_calls = 0

        t0 = time.time()

        for i in range(n):
            level = int(-math.log(rng.uniform()) * ml)
            neighbors.append({})
            max_layer.append(level)

            if enter_point == -1:
                enter_point = i
                max_level = level
                for l_init in range(level + 1):
                    neighbors[i][l_init] = []
                continue

            # Greedy search from top to insertion level (no skipping)
            curr_ep = [enter_point]
            for l in range(max_level, level, -1):
                visited = set(curr_ep)
                best_d = float(np.sum((data[i] - data[curr_ep[0]]) ** 2))
                distance_computations += 1
                best_id = curr_ep[0]
                changed = True
                while changed:
                    changed = False
                    for nb in neighbors[best_id].get(l, []):
                        if nb in visited:
                            continue
                        visited.add(nb)
                        d = float(np.sum((data[i] - data[nb]) ** 2))
                        distance_computations += 1
                        if d < best_d:
                            best_d = d
                            best_id = nb
                            changed = True
                curr_ep = [best_id]

            # Search and connect at each layer with classifier-guided skipping
            for l in range(min(level, max_level), -1, -1):
                M_layer = M_max0 if l == 0 else M

                visited = set(curr_ep)
                results: list[tuple[float, int]] = []
                candidates: list[tuple[float, int]] = []
                encounter_counter = 0

                for ep in curr_ep:
                    d = float(np.sum((data[i] - data[ep]) ** 2))
                    distance_computations += 1
                    heapq.heappush(candidates, (d, ep))
                    heapq.heappush(results, (-d, ep))
                    if len(results) > ef_construction:
                        heapq.heappop(results)
                    encounter_counter += 1

                while candidates:
                    d_c, c = heapq.heappop(candidates)
                    worst_dist = -results[0][0] if results else float("inf")
                    if d_c > worst_dist:
                        break

                    # Collect all unvisited neighbors for batch prediction
                    nb_list = []
                    for nb in neighbors[c].get(l, []):
                        if nb not in visited:
                            visited.add(nb)
                            nb_list.append(nb)

                    if not nb_list:
                        continue

                    # Build feature matrix for batch prediction
                    t_clf = time.perf_counter_ns()
                    batch_feats = np.empty((len(nb_list), self._n_features), dtype=np.float32)
                    # Map feature names to values
                    feat_map = {
                        "candidate_degree": None,  # filled per candidate
                        "candidate_layer": None,
                        "current_layer": l,
                        "approx_dist": None,
                        "candidate_rank_in_beam": None,
                        "beam_size": len(results),
                        "inserted_count": i,
                    }
                    for j, nb in enumerate(nb_list):
                        feat_map["candidate_degree"] = len(neighbors[nb].get(l, []))
                        feat_map["candidate_layer"] = max_layer[nb]
                        feat_map["candidate_rank_in_beam"] = encounter_counter + j
                        if "approx_dist" in self._feature_names:
                            if data_proj is not None:
                                feat_map["approx_dist"] = float(np.sum((data_proj[i] - data_proj[nb]) ** 2))
                            else:
                                feat_map["approx_dist"] = self.approx_dist.estimate(data[i], data[nb])
                        for k, fname in enumerate(self._feature_names):
                            batch_feats[j, k] = feat_map[fname]

                    # Normalize
                    if self._mean is not None:
                        batch_feats = (batch_feats - self._mean) / self._std

                    # Single batch prediction
                    skip_probas = self.classifier.predict_proba(batch_feats)
                    classifier_overhead_ns += (time.perf_counter_ns() - t_clf)
                    n_classifier_calls += 1

                    # Process results
                    for j, nb in enumerate(nb_list):
                        encounter_counter += 1
                        if skip_probas[j] > self.threshold:
                            skipped_computations += 1
                            continue

                        d_n = float(np.sum((data[i] - data[nb]) ** 2))
                        distance_computations += 1

                        worst_dist = -results[0][0] if results else float("inf")
                        if d_n < worst_dist or len(results) < ef_construction:
                            heapq.heappush(candidates, (d_n, nb))
                            heapq.heappush(results, (-d_n, nb))
                            if len(results) > ef_construction:
                                heapq.heappop(results)

                # Select neighbors and connect
                results_list = [(-d, idx) for d, idx in results]
                results_list.sort(key=lambda x: x[0])
                selected = [idx for _, idx in results_list[:M_layer]]

                neighbors[i][l] = selected
                for s in selected:
                    if l not in neighbors[s]:
                        neighbors[s][l] = []
                    neighbors[s][l].append(i)
                    if len(neighbors[s][l]) > M_layer:
                        dists = [(float(np.sum((data[s] - data[nb2]) ** 2)), nb2)
                                 for nb2 in neighbors[s][l]]
                        distance_computations += len(dists)
                        dists.sort(key=lambda x: x[0])
                        neighbors[s][l] = [idx for _, idx in dists[:M_layer]]

                curr_ep = [selected[0]] if selected else curr_ep

            if level > max_level:
                max_level = level
                enter_point = i

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  LearnedSkip: {i + 1}/{n} inserted ({elapsed:.1f}s), "
                      f"skipped={skipped_computations}, computed={distance_computations}")

        build_time = time.time() - t0

        index = PythonHNSWIndex(
            data=data.copy(),
            neighbors=neighbors,
            max_layer=max_layer,
            enter_point=enter_point,
            max_level=max_level,
        )

        return BuiltIndex(
            index=index,
            build_time_seconds=build_time,
            distance_computations=distance_computations,
            skipped_computations=skipped_computations,
            classifier_overhead_seconds=classifier_overhead_ns / 1e9,
            n_classifier_calls=n_classifier_calls,
        )
