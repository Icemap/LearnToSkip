"""GIST1M dataset loader (HDF5 format from ann-benchmarks)."""
import numpy as np
import h5py
import requests

from learn_to_skip.datasets.base import BaseDataset, DatasetMetadata


class Gist1MDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "gist1m"

    def download(self) -> None:
        url = "http://ann-benchmarks.com/gist-960-euclidean.hdf5"
        h5_path = self._raw_dir / "gist-960-euclidean.hdf5"
        if not h5_path.exists():
            print(f"Downloading GIST1M from {url}...")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(h5_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

    def load_train(self) -> np.ndarray:
        self.ensure_available()
        with h5py.File(self._raw_dir / "gist-960-euclidean.hdf5", "r") as f:
            return np.array(f["train"], dtype=np.float32)

    def load_query(self) -> np.ndarray:
        self.ensure_available()
        with h5py.File(self._raw_dir / "gist-960-euclidean.hdf5", "r") as f:
            return np.array(f["test"], dtype=np.float32)

    def load_groundtruth(self, k: int = 100) -> np.ndarray:
        """Use HDF5-bundled ground truth."""
        self.ensure_available()
        with h5py.File(self._raw_dir / "gist-960-euclidean.hdf5", "r") as f:
            gt = np.array(f["neighbors"], dtype=np.int32)
        return gt[:, :k]

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="gist1m", dim=960, metric="l2", n_train=1_000_000, n_query=1_000
        )
