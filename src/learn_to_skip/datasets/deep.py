"""Deep1M (deep-image-96) dataset loader."""
import numpy as np
import h5py
import requests

from learn_to_skip.datasets.base import BaseDataset, DatasetMetadata


class Deep1MDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "deep1m"

    def download(self) -> None:
        url = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"
        h5_path = self._raw_dir / "deep-image-96-angular.hdf5"
        if not h5_path.exists():
            print(f"Downloading Deep1M from {url}...")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(h5_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

    def load_train(self) -> np.ndarray:
        self.ensure_available()
        with h5py.File(self._raw_dir / "deep-image-96-angular.hdf5", "r") as f:
            return np.array(f["train"], dtype=np.float32)

    def load_query(self) -> np.ndarray:
        self.ensure_available()
        with h5py.File(self._raw_dir / "deep-image-96-angular.hdf5", "r") as f:
            return np.array(f["test"], dtype=np.float32)

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="deep1m", dim=96, metric="l2", n_train=1_000_000, n_query=10_000
        )
