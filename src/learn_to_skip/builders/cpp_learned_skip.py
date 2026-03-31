"""C++ hnswlib-integrated LearnToSkip builder with classifier-guided candidate pruning."""
import time

import numpy as np
import hnswlib

from learn_to_skip.builders.base import BaseBuilder, BuiltIndex
from learn_to_skip.config import SEED


# Default universal tree params (trained on SIFT10K, depth-3)
DEFAULT_TREE_PARAMS = {
    "rank_thresholds": np.array([545.455, 205.390, 921.405]),
    "adist_thresholds": np.array([13.823, 15.035, 10.380]),
    "ins_t0": 2146.184,
    "leaf_probas": np.array([0.133092, 0.241549, 0.345719, 0.537761,
                             0.559158, 0.763484, 0.738662, 0.870165]),
}


class CppLearnedSkipBuilder(BaseBuilder):
    """Build HNSW using C++ hnswlib fork with classifier-guided candidate pruning.

    Two modes:
    - universal: Use pre-trained tree, skip from the start. Zero training cost.
    - online: Insert first `train_fraction` vanilla, train tree on trace, skip the rest.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        proj_dim: int = 16,
        mode: str = "universal",
        train_fraction: float = 0.3,
        tree_params: dict | None = None,
        colocated: bool = True,
        seed: int = SEED,
    ) -> None:
        self.threshold = threshold
        self.proj_dim = proj_dim
        self.mode = mode
        self.train_fraction = train_fraction
        self.tree_params = tree_params or DEFAULT_TREE_PARAMS
        self.colocated = colocated
        self.seed = seed

    @property
    def name(self) -> str:
        return f"CppLearnToSkip-{self.mode}"

    def build(
        self, data: np.ndarray, M: int, ef_construction: int, metric: str = "l2"
    ) -> BuiltIndex:
        n, dim = data.shape

        # Random projection matrix
        rng = np.random.RandomState(self.seed)
        proj_matrix = rng.randn(dim, self.proj_dim).astype(np.float32)
        proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)
        data_proj = (data @ proj_matrix).astype(np.float32)

        # Create index
        index = hnswlib.Index(space=metric, dim=dim)
        index.init_index(max_elements=n, M=M, ef_construction=ef_construction,
                         random_seed=self.seed)

        if self.colocated:
            index.enable_projection_storage(self.proj_dim)

        t0 = time.time()

        if self.mode == "universal":
            self._build_universal(index, data, data_proj, n)
        elif self.mode == "online":
            self._build_online(index, data, data_proj, n, M, ef_construction, metric)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        build_time = time.time() - t0

        metrics = index.get_construction_metrics()

        # Cleanup skip functor
        index.deactivate_skip()
        index.clear_skip_functor()

        return BuiltIndex(
            index=index,
            build_time_seconds=build_time,
            distance_computations=metrics["distance_computations"],
            skipped_computations=metrics["skipped_computations"],
        )

    def _build_universal(self, index, data, data_proj, n):
        """Universal mode: pre-trained tree, skip from point 0."""
        index.create_skip_functor(threshold=self.threshold, proj_dim=self.proj_dim)
        index.set_skip_projected_data(data_proj)
        self._apply_tree_params(index)
        index.activate_skip()
        index.add_items_with_skip(data, data_proj)

    def _build_online(self, index, data, data_proj, n, M, ef_construction, metric):
        """Online mode: vanilla first, train tree on trace, skip the rest."""
        split = int(n * self.train_fraction)

        # Phase 1: Vanilla insertion (multi-threaded)
        index.add_items(data[:split], ids=np.arange(split))

        # Phase 2: Train tree from trace
        # For online mode, we use the HNSWTracer to collect features and train
        # Then export denormalized tree params
        tree_params = self._train_tree_from_trace(
            data[:split], data_proj[:split], M, ef_construction, metric
        )

        # Phase 3: Skip insertion (single-threaded)
        index.create_skip_functor(threshold=self.threshold, proj_dim=self.proj_dim)
        index.set_skip_projected_data(data_proj)
        if tree_params is not None:
            index.set_skip_tree_params(**tree_params)
        else:
            self._apply_tree_params(index)
        index.activate_skip()

        remaining_ids = np.arange(split, n)
        index.add_items_with_skip(data[split:], data_proj[split:], ids=remaining_ids)

    def _apply_tree_params(self, index):
        """Apply tree parameters to the skip functor."""
        index.set_skip_tree_params(
            rank_thresholds=self.tree_params["rank_thresholds"],
            adist_thresholds=self.tree_params["adist_thresholds"],
            ins_t0=self.tree_params["ins_t0"],
            leaf_probas=self.tree_params["leaf_probas"],
        )

    def _train_tree_from_trace(self, data, data_proj, M, ef_construction, metric):
        """Train a decision tree on trace data and return denormalized params.

        Returns None if training fails (falls back to universal tree).
        """
        try:
            from learn_to_skip.trace.tracer import HNSWTracer
            from sklearn.tree import DecisionTreeClassifier

            tracer = HNSWTracer(dim=data.shape[1], M=M, ef_construction=ef_construction)
            tracer.build(data)
            df = tracer.get_trace_df()

            if len(df) < 100:
                return None

            features = ["candidate_rank_in_beam", "approx_dist", "inserted_count"]
            available = [f for f in features if f in df.columns]
            if len(available) < 3:
                return None

            X = df[available].values
            y = df["is_wasted"].values

            tree = DecisionTreeClassifier(max_depth=3, random_state=self.seed)
            tree.fit(X, y)

            return self._extract_denormalized_params(tree, available)
        except Exception:
            return None

    def _extract_denormalized_params(self, tree, feature_names):
        """Extract denormalized tree parameters from a sklearn tree.

        Maps the depth-3 tree structure to our fixed TreeParams format.
        Falls back to default if structure doesn't match.
        """
        try:
            t = tree.tree_
            if t.node_count < 7:
                return None

            # Extract thresholds by traversing the tree
            # Our tree structure: root splits on rank, then each child splits again
            rank_thresholds = np.zeros(3)
            adist_thresholds = np.zeros(3)
            ins_t0 = 0.0
            leaf_probas = np.zeros(8)

            # Simple extraction: use threshold values directly
            feature_idx = {name: i for i, name in enumerate(feature_names)}
            rank_idx = feature_idx.get("candidate_rank_in_beam", 0)
            adist_idx = feature_idx.get("approx_dist", 1)
            ins_idx = feature_idx.get("inserted_count", 2)

            # Walk tree to extract thresholds and leaf values
            thresholds = t.threshold
            features = t.feature
            values = t.value

            # Collect leaf probabilities
            leaf_count = 0
            for i in range(t.node_count):
                if t.children_left[i] == -1:  # leaf
                    total = values[i].sum()
                    if total > 0:
                        proba = values[i][0][1] / total if values[i].shape[1] > 1 else 0.0
                    else:
                        proba = 0.0
                    if leaf_count < 8:
                        leaf_probas[leaf_count] = proba
                    leaf_count += 1

            # Collect split thresholds by feature type
            rank_splits = []
            adist_splits = []
            ins_splits = []
            for i in range(t.node_count):
                if thresholds[i] != -2:  # not a leaf
                    feat = features[i]
                    if feat == rank_idx:
                        rank_splits.append(thresholds[i])
                    elif feat == adist_idx:
                        adist_splits.append(thresholds[i])
                    elif feat == ins_idx:
                        ins_splits.append(thresholds[i])

            # Fill in thresholds (pad with defaults if not enough)
            for i in range(3):
                rank_thresholds[i] = rank_splits[i] if i < len(rank_splits) else DEFAULT_TREE_PARAMS["rank_thresholds"][i]
                adist_thresholds[i] = adist_splits[i] if i < len(adist_splits) else DEFAULT_TREE_PARAMS["adist_thresholds"][i]
            ins_t0 = ins_splits[0] if ins_splits else DEFAULT_TREE_PARAMS["ins_t0"]

            return {
                "rank_thresholds": rank_thresholds,
                "adist_thresholds": adist_thresholds,
                "ins_t0": ins_t0,
                "leaf_probas": leaf_probas,
            }
        except Exception:
            return None
