"""Random-skip baseline builder - real per-candidate random skipping in Python HNSW."""
import heapq
import math
import time

import numpy as np

from learn_to_skip.builders.base import BaseBuilder, BuiltIndex
from learn_to_skip.builders.learned_skip import PythonHNSWIndex
from learn_to_skip.config import RANDOM_SKIP_PROB, SEED


class RandomSkipBuilder(BaseBuilder):
    """Skip candidates randomly with probability p during HNSW construction.

    For each candidate neighbor encountered during beam search, flip a coin.
    If skip: do not compute exact distance, do not add to beam.
    """

    def __init__(self, skip_prob: float = RANDOM_SKIP_PROB, seed: int = SEED) -> None:
        self.skip_prob = skip_prob
        self.seed = seed

    @property
    def name(self) -> str:
        return f"Random-Skip(p={self.skip_prob})"

    def build(
        self, data: np.ndarray, M: int, ef_construction: int, metric: str = "l2"
    ) -> BuiltIndex:
        n, dim = data.shape
        ml = 1.0 / math.log(M)
        M_max0 = 2 * M
        rng = np.random.RandomState(self.seed)

        graph_data: list[np.ndarray] = []
        neighbors: list[dict[int, list[int]]] = []
        max_layer: list[int] = []
        enter_point = -1
        max_level = -1

        distance_computations = 0
        skipped_computations = 0

        t0 = time.time()

        for i in range(n):
            level = int(-math.log(rng.uniform()) * ml)
            graph_data.append(data[i])
            neighbors.append({})
            max_layer.append(level)

            if enter_point == -1:
                enter_point = i
                max_level = level
                for l in range(level + 1):
                    neighbors[i][l] = []
                continue

            curr_ep = [enter_point]
            for l in range(max_level, level, -1):
                visited = set(curr_ep)
                best_d = float(np.sum((data[i] - graph_data[curr_ep[0]]) ** 2))
                distance_computations += 1
                best_id = curr_ep[0]
                changed = True
                while changed:
                    changed = False
                    for nb in neighbors[best_id].get(l, []):
                        if nb in visited:
                            continue
                        visited.add(nb)
                        d = float(np.sum((data[i] - graph_data[nb]) ** 2))
                        distance_computations += 1
                        if d < best_d:
                            best_d = d
                            best_id = nb
                            changed = True
                curr_ep = [best_id]

            for l in range(min(level, max_level), -1, -1):
                M_layer = M_max0 if l == 0 else M

                visited = set(curr_ep)
                results: list[tuple[float, int]] = []
                candidates: list[tuple[float, int]] = []

                for ep in curr_ep:
                    d = float(np.sum((data[i] - graph_data[ep]) ** 2))
                    distance_computations += 1
                    heapq.heappush(candidates, (d, ep))
                    heapq.heappush(results, (-d, ep))
                    if len(results) > ef_construction:
                        heapq.heappop(results)

                while candidates:
                    d_c, c = heapq.heappop(candidates)
                    worst_dist = -results[0][0] if results else float("inf")
                    if d_c > worst_dist:
                        break

                    for nb in neighbors[c].get(l, []):
                        if nb in visited:
                            continue
                        visited.add(nb)

                        # Random skip decision
                        if rng.random() < self.skip_prob:
                            skipped_computations += 1
                            continue

                        d_n = float(np.sum((data[i] - graph_data[nb]) ** 2))
                        distance_computations += 1

                        worst_dist = -results[0][0] if results else float("inf")
                        if d_n < worst_dist or len(results) < ef_construction:
                            heapq.heappush(candidates, (d_n, nb))
                            heapq.heappush(results, (-d_n, nb))
                            if len(results) > ef_construction:
                                heapq.heappop(results)

                results_list = [(-d, idx) for d, idx in results]
                results_list.sort(key=lambda x: x[0])
                selected = [idx for _, idx in results_list[:M_layer]]

                neighbors[i][l] = selected
                for s in selected:
                    if l not in neighbors[s]:
                        neighbors[s][l] = []
                    neighbors[s][l].append(i)
                    if len(neighbors[s][l]) > M_layer:
                        dists = [(float(np.sum((graph_data[s] - graph_data[nb2]) ** 2)), nb2)
                                 for nb2 in neighbors[s][l]]
                        distance_computations += len(dists)
                        dists.sort(key=lambda x: x[0])
                        neighbors[s][l] = [idx for _, idx in dists[:M_layer]]

                curr_ep = [selected[0]] if selected else curr_ep

            if level > max_level:
                max_level = level
                enter_point = i

        build_time = time.time() - t0

        index = PythonHNSWIndex(
            data=np.array(graph_data),
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
        )
