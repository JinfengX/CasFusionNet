import multiprocessing
import os
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml
from joblib import Parallel, delayed
from loguru import logger
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(filepath):
    with open(filepath, encoding="utf-8") as sv_fl:
        file = yaml.load(sv_fl, Loader=yaml.CSafeLoader)
    return file


def save_yaml(path, file):
    with open(path, "w", encoding="utf-8") as sv_fl:
        yaml.safe_dump(file, sv_fl, default_style=None, default_flow_style=False)


def create_label_database(class_map, color_map, save_dir, ignor_class=()) -> dict:
    label_database = dict()
    for class_name, class_id in class_map.items():
        label_database[class_id] = {
            "color": color_map[class_id].tolist(),
            "name": class_name,
            "validation": class_id not in ignor_class,
        }

    save_yaml(save_dir / "label_database.yaml", label_database)
    return label_database


def pcd_downsample(pcd, number):
    pcd_ts = torch.tensor(
        pcd, dtype=torch.float, device=torch.device("cuda"), requires_grad=False
    )
    pcd_ts = pcd_ts.unsqueeze(0)
    ds_idx = furthest_point_sample(pcd_ts[:, :, :3].contiguous(), number)
    pcd_ds = gather_operation(pcd_ts.permute(0, 2, 1).contiguous(), ds_idx)
    pcd_ds = pcd_ds.permute(0, 2, 1).squeeze(0).cpu().numpy()
    return pcd_ds


def built_o3d_pcd(pcd: np.ndarray, colored: bool = False):
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[:, :3]))
    if colored:
        assert pcd.shape[1] > 3, f"No colors information !!!"
        colors = pcd[:, 3:6]
        assert (
            (colors >= 0 - 1e-5) & (colors <= 1 + 1e-5)
        ).all(), f"Please make sure the range of color fall into [0,1]"
        colors = np.where(colors > 1.0, 1.0, colors)
        colors = np.where(colors < 0.0, 0.0, colors)
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    return pcd_o3d


def save_pcd_ply(path, pcd: np.ndarray, colored: bool = False):
    try:
        pcd_o3d = built_o3d_pcd(pcd, colored)
    except Exception as e:
        logger.error(f"Error when building o3d point cloud !!! Failed to save {path}")
        raise e
    o3d.io.write_point_cloud(str(path), pcd_o3d)


class BasePreprocessing:
    def __init__(
        self,
        data_dir: str,
        save_dir: str,
        modes: tuple,
        n_jobs: int = -1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.n_jobs = n_jobs
        self.modes = modes

        logger.info(f"Data folder: {self.data_dir}")
        logger.info(f"Save folder: {self.save_dir}")
        if not self.data_dir.exists():
            logger.error("data folder doesn't exist")
            raise FileNotFoundError
        if self.save_dir.exists() is False:
            self.save_dir.mkdir(parents=True)

        self.files = {}
        for data_type in self.modes:
            self.files.update({data_type: []})

    @logger.catch
    def preprocess(self):
        self.n_jobs = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
        for mode in self.modes:
            database = []
            logger.info(f"Tasks for {mode}: {len(self.files[mode])}")
            parallel_results = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(self.process_file)(file, mode) for file in self.files[mode]
            )
            assert isinstance(parallel_results, Iterable)
            for filebase in parallel_results:
                database.append(filebase)
            self.save_database(database, mode)
        self.joint_database(self.modes)

    def process_file(self, filepath, mode):
        """process_file.

        Args:
            filepath: path to the main file
            mode: typically train, test or validation

        Returns:
            filebase: info about file
        """

        raise NotImplementedError

    def save_database(self, database, mode):
        for element in database:
            self._dict_to_yaml(element)
        save_yaml(self.save_dir / (mode + "_database.yaml"), database)

    @classmethod
    def _dict_to_yaml(cls, dictionary):
        if not isinstance(dictionary, dict):
            return
        for k, value in dictionary.items():
            if isinstance(value, dict):
                cls._dict_to_yaml(value)
            if isinstance(value, np.ndarray):
                dictionary[k] = value.tolist()
            if isinstance(value, Path):
                dictionary[k] = str(value)

    def joint_database(self, train_modes: tuple, **kwargs):
        joint_db = []
        for mode in train_modes:
            joint_db.extend(load_yaml(self.save_dir / (mode + "_database.yaml")))
        save_yaml(self.save_dir / "train_test_database.yaml", joint_db)
