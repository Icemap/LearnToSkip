"""GloVe-200 dataset loader."""
import numpy as np
import h5py
import requests

from learn_to_skip.datasets.base import BaseDataset, DatasetMetadata


class GloVe200Dataset(BaseDataset):
    @property
    def name(self) -> str:
        return "glove200"

    def download(self) -> None:
        # ann-benchmarks HDF5 format
        url = "http://ann-benchmarks.com/glove-200-angular.hdf5"
        h5_path = self._raw_dir / "glove-200-angular.hdf5"
        if not h5_path.exists():
            print(f"Downloading GloVe-200 from {url}...")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(h5_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

    def load_train(self) -> np.ndarray:
        self.ensure_available()
        with h5py.File(self._raw_dir / "glove-200-angular.hdf5", "r") as f:
            return np.array(f["train"], dtype=np.float32)

    def load_query(self) -> np.ndarray:
        self.ensure_available()
        with h5py.File(self._raw_dir / "glove-200-angular.hdf5", "r") as f:
            return np.array(f["test"], dtype=np.float32)

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="glove200", dim=200, metric="cosine", n_train=1_183_514, n_query=10_000
        )
