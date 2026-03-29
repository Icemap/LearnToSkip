"""Dataset registry."""
from learn_to_skip.datasets.base import BaseDataset, DatasetMetadata
from learn_to_skip.datasets.sift import Sift1MDataset, Sift10KDataset
from learn_to_skip.datasets.gist import Gist1MDataset
from learn_to_skip.datasets.glove import GloVe200Dataset
from learn_to_skip.datasets.deep import Deep1MDataset
from learn_to_skip.datasets.streaming import StreamingDataGenerator

DATASET_REGISTRY: dict[str, type[BaseDataset]] = {
    "sift1m": Sift1MDataset,
    "sift10k": Sift10KDataset,
    "gist1m": Gist1MDataset,
    "glove200": GloVe200Dataset,
    "deep1m": Deep1MDataset,
}


def get_dataset(name: str) -> BaseDataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]()


__all__ = [
    "BaseDataset", "DatasetMetadata", "get_dataset", "DATASET_REGISTRY",
    "Sift1MDataset", "Sift10KDataset", "Gist1MDataset", "GloVe200Dataset",
    "Deep1MDataset", "StreamingDataGenerator",
]
