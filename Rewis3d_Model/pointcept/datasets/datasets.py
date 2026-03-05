from .builder import DATASETS

from .transform_3d import Compose, TRANSFORMS
from .transform_2d import Compose2D

from torch.utils.data import Dataset

import os
import glob
import numpy as np
import sys

# NumPy 2.x -> <2.x pickle compatibility shim
# When object arrays/dicts are pickled with NumPy 2.x, the module path 'numpy._core'
# may be referenced. Older NumPy versions don't have this package, causing
# ModuleNotFoundError during unpickling inside np.load(..., allow_pickle=True).
# We alias 'numpy._core' to 'numpy.core' to restore compatibility.
try:
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        # Optionally alias a few common submodules if already imported
        for _name in (
            "_multiarray_umath",
            "multiarray",
            "numeric",
            "fromnumeric",
            "arrayprint",
        ):
            _src = f"numpy.core.{_name}"
            _dst = f"numpy._core.{_name}"
            if _src in sys.modules and _dst not in sys.modules:
                sys.modules[_dst] = sys.modules[_src]
except Exception:
    # Best-effort shim; continue without blocking if anything goes wrong
    pass


@DATASETS.register_module()
class DefaultReconstructedDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/preprocessed_dataset",
        transform=None,
        transform_2d=None,
        test_mode=False,
        loop=1,
        **kwargs,
    ):
        """
        Args:
            split (str): dataset split - 'train', 'val', 'test', etc.
            data_root (str): base directory containing preprocessed data
            transform_3d (callable): transform function for 3D data
            transform_2d (callable): transform function for 2D data
            test_mode (bool): if True, disables looping/augmentation
            loop (int): how many times to virtually repeat the dataset
        """
        self.split = split
        self.data_root = data_root
        self.transform = Compose(transform)
        self.transform_2d = Compose2D(transform_2d)
        self.test_mode = test_mode
        self.loop = loop if not test_mode else 1

        # Find all .npz files in the split directory
        self.split_dir = os.path.join(self.data_root, split)
        assert os.path.exists(self.split_dir), f"Split path not found: {self.split_dir}"

        self.files = sorted(glob.glob(os.path.join(self.split_dir, "*.npz")))

        assert len(self.files) > 0, f"No .npz files found in {self.split_dir}"

    def __len__(self):
        return len(self.files) * self.loop

    def get_data(self, idx):
        file_path = self.files[idx % len(self.files)]
        with np.load(file_path, allow_pickle=True) as data:
            data_3d = dict(data["data_3d"].item())
            data_2d = dict(data["data_2d"].item())
            if "conf" in data_3d:
                data_3d["conf"] = data_3d["conf"][:, np.newaxis]

            if data_2d["student_mask_1"].ndim == 3:
                data_2d["student_mask_1"] = data_2d["student_mask_1"][:, :, 0]

        return data_2d, data_3d

    def __getitem__(self, idx):
        data_2d, data_3d = self.get_data(idx)

        data_3d = self.transform(data_3d)
        data_2d = self.transform_2d(data_2d)

        return {**data_3d, **data_2d}
