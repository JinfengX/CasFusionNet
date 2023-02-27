from typing import List

import numpy as np
from torch.utils.data import Dataset

from src.data.preprocessing.base_preprocessing import load_yaml


class SSCPC(Dataset):
    def __init__(self, data_path, train: bool, transform=None):
        mode = "train" if train else "test"

        data_attr = load_yaml(f"{data_path}/{mode}_database.yaml")
        self.data_path: List[str] = [
            f'{data_path}/{attr["scene"]}/{attr["scene_name"]}' for attr in data_attr
        ]

        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        in_xyz = np.load(f"{self.data_path[index]}_input.npy")
        gt_xyzl = np.load(f"{self.data_path[index]}_gt.npy")

        return in_xyz, gt_xyzl
