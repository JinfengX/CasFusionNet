import os
import random
import re
from typing import List

import numpy as np
import pyrootutils
from loguru import logger
from natsort import natsorted

from base_preprocessing import (
    BasePreprocessing,
    built_o3d_pcd,
    load_yaml,
    pcd_downsample,
    save_pcd_ply,
    save_yaml,
    setup_seed,
    create_label_database,
)

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set your gpu id here
setup_seed(55)


class SscPcPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir=root / "data/data_ori/SceneCAD_Dataset_Original/",
        save_dir=root / "data/SSC-PC/",
        config_dir=root / "configs/data",
        modes: tuple = ("Bathroom", "Bedroom", "Livingroom", "Office"),
        split: bool = True,
        n_jobs: int = 20,  # reduce this number if you have less CPU cores and gpu memory
    ) -> None:
        super().__init__(data_dir, save_dir, modes, n_jobs)

        sum_files_num = 0
        for mode in self.modes:
            self.files[mode] = natsorted((self.data_dir / mode).glob(f"*_{mode}_*_*"))
            # remove low quality files
            if mode == "Bedroom":
                for i in range(1, 5):
                    self.files[mode].remove(self.data_dir / mode / f"08_Bedroom_61_{i}")
            sum_files_num += len(self.files[mode])
        assert sum_files_num == 1941, f"Total files number is not correct !!!"
        logger.info(f"Total files: {sum_files_num}")

        data_mapping = load_yaml(config_dir / "ssc_pc_mapping.yaml")
        self.color_map = np.array(list(data_mapping["color_map"].values()))
        self.class_map = data_mapping["class_map"]
        create_label_database(self.class_map, self.color_map, self.save_dir)

        dataset_param = load_yaml(config_dir / "ssc_pc_param.yaml")
        self.input_ds_num = dataset_param["input_ds_num"]
        self.gt_ds_num = dataset_param["groundtruth_ds_num"]
        self.save_opts = dataset_param["save_options"]

        self.split = split

    def process_file(self, filepath, mode):
        """process_file.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            file_base: info about file
        """
        (
            scene_id,
            scene,
            sub_scene,
            view,
            scene_name,
        ) = self._parse_id_scene_subscene_view(filepath.name)
        assert mode == scene
        file_base = {
            "filepath": self.save_dir / mode / scene_name,
            "id": scene_id,
            "scene": scene,
            "sub_scene": sub_scene,
            "view": view,
            "scene_name": scene_name,
            "raw_filepath": filepath,
        }

        # get ground truth
        gt_pcd = self._get_gt_pcd(filepath / f"{filepath.name}_pointcloud.txt")

        # get camera information
        cam_pose, cam_quat = self._get_camera_info(
            filepath / f"{filepath.name}_camera.txt"
        )

        # get input by approximating the visibility of the ground truth from given views
        select_idx = self._approximate_visibility(gt_pcd, cam_pose, factor=200)
        input_pcd = gt_pcd[select_idx][:, :3]

        # normalize the input and gt point cloud
        gt_pcd[:, :3], center, scale = self.normalize_pcd(gt_pcd[:, :3])
        input_pcd, *_ = self.normalize_pcd(input_pcd, center, scale)

        # downsample the input and gt point cloud
        input_pcd = pcd_downsample(input_pcd, self.input_ds_num)
        gt_pcd = pcd_downsample(gt_pcd, self.gt_ds_num)

        data_collection = {"data": {"input": input_pcd, "gt": gt_pcd}}
        save_dir = self.save_dir / mode
        save_dir.mkdir(exist_ok=True, parents=False)
        self._save_data(
            save_dir, file_base["scene_name"], **{**data_collection, **self.save_opts}
        )

        return file_base

    @classmethod
    def _parse_id_scene_subscene_view(cls, filename):
        scene_name = filename
        match = re.match(r"(\d{2})_(\w+)_(\d{2})_(\d)", scene_name)
        scene_id, mode, sub_scene, view = (
            int(match.group(1)),
            str(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )

        return scene_id, mode, sub_scene, view, scene_name

    @classmethod
    def _get_gt_pcd(cls, filepath):
        def _parse_line(line):
            return list(map(float, line.split()))

        with open(filepath, "r") as pcd_fl:
            pcd = pcd_fl.readlines()
        pcd = np.array(list(map(_parse_line, pcd)))

        return pcd

    @classmethod
    def _get_camera_info(cls, filepath):
        with open(filepath, "r") as cam_fl:
            camera_info = cam_fl.readlines()
        pose: List = list(map(float, camera_info[0].split()[1:4]))
        pose[0] = -pose[0]
        quat = list(map(float, camera_info[1].split()[1:5]))
        quat.append(quat.pop(0))

        return np.array(pose), np.array(quat)

    @classmethod
    def _approximate_visibility(cls, gt_pcd, cam_pose, factor=200):
        # convert the ground truth point cloud to open3d format
        gt_pcd_o3d = built_o3d_pcd(gt_pcd)

        diameter = np.linalg.norm(
            np.asarray(gt_pcd_o3d.get_max_bound())
            - np.asarray(gt_pcd_o3d.get_min_bound())
        )
        radius = diameter * factor

        _, index = gt_pcd_o3d.hidden_point_removal(cam_pose, radius)
        return index

    @staticmethod
    def normalize_pcd(pcd, center=None, scale=None):
        assert pcd.shape[1] == 3, "point cloud should be Nx3"
        if center is None:
            center = np.mean(pcd, axis=0) if center is None else center
            pcd = pcd - center
            assert np.allclose(np.mean(pcd, axis=0), 0.0)
        else:
            pcd = pcd - center

        if scale is None:
            scale = (
                np.max(np.sqrt(np.sum(pcd**2, axis=1)), axis=0)
                if scale is None
                else scale
            )
            pcd = pcd / scale
            assert np.allclose(np.max(np.sqrt(np.sum(pcd**2, axis=1))), 1.0)
        else:
            pcd = pcd / scale

        return pcd, center, scale

    def _save_data(self, save_dir, scene, **save_options):
        data_collect = save_options["data"]
        save_npy_opts = save_options["save_npy"]
        visual_opts = save_options["visualization"]

        if save_npy_opts.get("input") is not None:
            np.save(save_dir / f"{scene}_input.npy", data_collect["input"])
        if save_npy_opts.get("ground_truth") is not None:
            np.save(save_dir / f"{scene}_gt.npy", data_collect["gt"])

        if visual_opts.get("input") is not None:
            save_pcd_ply(save_dir / f"{scene}_vis_input.ply", data_collect["input"])
        if visual_opts.get("ground_truth") is not None:
            gt_pcd = data_collect["gt"]
            gt_pcd = np.concatenate(
                (gt_pcd[:, :3], self.color_map[gt_pcd[:, 3].astype(int)] / 255), axis=1
            )
            save_pcd_ply(save_dir / f"{scene}_vis_gt.ply", gt_pcd, colored=True)

    def joint_database(self, train_modes: tuple, **kwargs):
        """Overrides the joint_database method in the base class to
        split the training set and validation set.

        :param train_modes:
        :return: None
        """
        if kwargs.get("read_file", False):
            joint_db = load_yaml(self.save_dir / "train_test_database.yaml")
        else:
            logger.info("Jointing the training and validation set")
            joint_db = []
            for mode in train_modes:
                joint_db.extend(load_yaml(self.save_dir / (mode + "_database.yaml")))
            save_yaml(self.save_dir / "train_test_database.yaml", joint_db)

        if self.split:
            logger.info("Splitting the training and validation set")
            ids = [item["id"] for item in joint_db]
            scenes = [item["scene"] for item in joint_db]
            sub_scene = [item["sub_scene"] for item in joint_db]
            assert len(joint_db) == len(ids) == len(scenes) == len(sub_scene) == 1941

            last_id, last_scene, last_room = ids[0], scenes[0], sub_scene[0]
            unique_scene, tmp_scene = [], []
            for i, db in enumerate(joint_db):
                scene_id, scene, room = ids[i], scenes[i], sub_scene[i]
                tmp_scene.append(db)
                if (
                    scene_id != last_id
                    or scene != last_scene
                    or room != last_room
                    or i == len(joint_db) - 1
                ):
                    unique_scene.append(
                        tmp_scene if i == len(joint_db) - 1 else tmp_scene[0:-1]
                    )
                    tmp_scene = [tmp_scene[-1]]
                last_id, last_scene, last_room = scene_id, scene, room

            train_set, test_set = [], []
            for views in unique_scene:
                test_eles = random.sample(views, 1)
                test_set.append(test_eles)
                for ele in test_eles:
                    views.remove(ele)
                train_set.append(views)
            test_set = [item for sub_list in test_set for item in sub_list]
            extra = random.sample(test_set, 92)  # random select 92 scenes for training
            train_set.append(extra)
            for item in extra:
                test_set.remove(item)
            train_set = [item for sub_list in train_set for item in sub_list]

            # check
            assert len(train_set) + len(test_set) == 1941
            for test_data in test_set:
                assert test_data not in train_set
            logger.info(f"train set: {len(train_set)}, test set: {len(test_set)}")

            save_yaml(self.save_dir / "train_database.yaml", train_set)
            save_yaml(self.save_dir / "test_database.yaml", test_set)


if __name__ == "__main__":
    ssc_pc_preprocess = SscPcPreprocessing(n_jobs=20, split=True)
    ssc_pc_preprocess.preprocess()
    # ssc_pc_preprocess.joint_database((), read_file=True)
