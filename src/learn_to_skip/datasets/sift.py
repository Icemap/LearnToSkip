"""SIFT1M and SIFT10K dataset loaders."""
import struct
import tarfile
from pathlib import Path

import numpy as np
import requests

from learn_to_skip.datasets.base import BaseDataset, DatasetMetadata


def _read_fvecs(path: Path) -> np.ndarray:
    """Read .fvecs format file."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) == 0:
        return np.array([], dtype=np.float32)
    dim = struct.unpack("i", data[:4])[0]
    record_size = 4 + dim * 4  # 4 bytes for dim + dim*4 bytes for floats
    n = len(data) // record_size
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        offset = i * record_size + 4  # skip dim field
        vectors[i] = np.frombuffer(data[offset : offset + dim * 4], dtype=np.float32)
    return vectors


def _read_ivecs(path: Path) -> np.ndarray:
    """Read .ivecs format file."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) == 0:
        return np.array([], dtype=np.int32)
    dim = struct.unpack("i", data[:4])[0]
    record_size = 4 + dim * 4
    n = len(data) // record_size
    vectors = np.zeros((n, dim), dtype=np.int32)
    for i in range(n):
        offset = i * record_size + 4
        vectors[i] = np.frombuffer(data[offset : offset + dim * 4], dtype=np.int32)
    return vectors


class Sift1MDataset(BaseDataset):
    """ANN-benchmarks SIFT1M dataset."""

    @property
    def name(self) -> str:
        return "sift1m"

    def download(self) -> None:
        url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        tar_path = self._raw_dir / "sift.tar.gz"
        if not tar_path.exists():
            print(f"Downloading SIFT1M from {url}...")
            # Use requests via http mirror
            http_url = "http://corpus-texmex.irisa.fr/sift.tar.gz"
            resp = requests.get(http_url, stream=True)
            resp.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        # Extract
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(self._raw_dir)

    def load_train(self) -> np.ndarray:
        self.ensure_available()
        return _read_fvecs(self._raw_dir / "sift" / "sift_base.fvecs")

    def load_query(self) -> np.ndarray:
        self.ensure_available()
        return _read_fvecs(self._raw_dir / "sift" / "sift_query.fvecs")

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="sift1m", dim=128, metric="l2", n_train=1_000_000, n_query=10_000
        )


class Sift10KDataset(BaseDataset):
    """Small SIFT subset for fast testing. Generated from SIFT1M or synthesized."""

    @property
    def name(self) -> str:
        return "sift10k"

    def download(self) -> None:
        # Generate synthetic SIFT-like data for fast testing
        rng = np.random.RandomState(42)
        train = rng.randn(10_000, 128).astype(np.float32)
        query = rng.randn(100, 128).astype(np.float32)
        np.save(self._raw_dir / "train.npy", train)
        np.save(self._raw_dir / "query.npy", query)

    def load_train(self) -> np.ndarray:
        self.ensure_available()
        return np.load(self._raw_dir / "train.npy")

    def load_query(self) -> np.ndarray:
        self.ensure_available()
        return np.load(self._raw_dir / "query.npy")

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="sift10k", dim=128, metric="l2", n_train=10_000, n_query=100
        )
