"""GIST1M dataset loader."""
import tarfile

import numpy as np
import requests

from learn_to_skip.datasets.base import BaseDataset, DatasetMetadata
from learn_to_skip.datasets.sift import _read_fvecs


class Gist1MDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "gist1m"

    def download(self) -> None:
        url = "http://corpus-texmex.irisa.fr/gist.tar.gz"
        tar_path = self._raw_dir / "gist.tar.gz"
        if not tar_path.exists():
            print(f"Downloading GIST1M from {url}...")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(self._raw_dir)

    def load_train(self) -> np.ndarray:
        self.ensure_available()
        return _read_fvecs(self._raw_dir / "gist" / "gist_base.fvecs")

    def load_query(self) -> np.ndarray:
        self.ensure_available()
        return _read_fvecs(self._raw_dir / "gist" / "gist_query.fvecs")

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="gist1m", dim=960, metric="l2", n_train=1_000_000, n_query=1_000
        )
