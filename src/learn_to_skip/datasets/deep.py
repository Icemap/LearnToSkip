"""Deep1M (deep-image-96) dataset loader. Uses first 1M vectors from ann-benchmarks."""
import numpy as np
import h5py
import requests

from learn_to_skip.datasets.base import BaseDataset, DatasetMetadata

_N_TRAIN = 1_000_000  # Use first 1M of the 9.99M vectors


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
            return np.array(f["train"][:_N_TRAIN], dtype=np.float32)

    def load_query(self) -> np.ndarray:
        self.ensure_available()
        with h5py.File(self._raw_dir / "deep-image-96-angular.hdf5", "r") as f:
            return np.array(f["test"], dtype=np.float32)

    def load_groundtruth(self, k: int = 100) -> np.ndarray:
        """Use HDF5-bundled ground truth, but recompute for truncated 1M subset."""
        gt_path = self._processed_dir / f"groundtruth_k{k}_n{_N_TRAIN}.npy"
        if gt_path.exists():
            return np.load(gt_path)
        # Recompute GT for truncated training set
        gt = self._compute_groundtruth(k)
        np.save(gt_path, gt)
        return gt

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="deep1m", dim=96, metric="cosine", n_train=_N_TRAIN, n_query=10_000
        )
