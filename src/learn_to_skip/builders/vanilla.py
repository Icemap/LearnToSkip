"""Vanilla HNSW builder using hnswlib."""
import time

import numpy as np
import hnswlib

from learn_to_skip.builders.base import BaseBuilder, BuiltIndex


class VanillaHNSWBuilder(BaseBuilder):
    @property
    def name(self) -> str:
        return "Vanilla-HNSW"

    def build(
        self, data: np.ndarray, M: int, ef_construction: int, metric: str = "l2"
    ) -> BuiltIndex:
        n, dim = data.shape
        space = "l2" if metric == "l2" else "cosine"
        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(max_elements=n, M=M, ef_construction=ef_construction)

        t0 = time.time()
        index.add_items(data, np.arange(n))
        build_time = time.time() - t0

        return BuiltIndex(
            index=index,
            build_time_seconds=build_time,
            distance_computations=n * ef_construction,  # approximate
            skipped_computations=0,
        )
