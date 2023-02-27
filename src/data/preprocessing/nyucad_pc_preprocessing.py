import os
import re
from operator import itemgetter

import cv2
import numpy as np
import open3d as o3d
import pyrootutils
import scipy
from ismember import ismember
from loguru import logger
from natsort import natsorted

from base_preprocessing import (
    BasePreprocessing,
    load_yaml,
    pcd_downsample,
    save_pcd_ply,
    setup_seed,
    create_label_database,
)

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set your gpu id here
setup_seed(55)


class NYUCADPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir=root / "data/data_ori/NYUCAD/",
        save_dir=root / "data/NYUCAD-PC/",
        config_dir=root / "configs/data",
        modes: tuple = ("train", "test"),
        check: bool = True,
        n_jobs: int = 20,  # reduce this number if you have less CPU cores and gpu memory
    ) -> None:
        super().__init__(data_dir, save_dir, modes, n_jobs)
        self.check = check

        dataset_mapping = load_yaml(config_dir / "nyucad_mapping.yaml")
        self.class_map = dataset_mapping["class_map"]
        self.nyu36to11_ids = dataset_mapping["nyu36to11_ids"]
        self.color_map = np.array(list(dataset_mapping["color_map"].values()))
        self.ignor_class = dataset_mapping["ignor"]
        create_label_database(
            self.class_map, self.color_map, self.save_dir, self.ignor_class
        )

        dataset_param = load_yaml(config_dir / "nyucad_param.yaml")
        self.cam_k = np.array(dataset_param["cam_k"])
        self.vox_unit = dataset_param["voxel_unit"]
        self.vox_size = np.array(dataset_param["voxel_size"])
        self.vox_size_model = np.array(dataset_param["voxel_size_model"])
        self.vox_size_cam = np.array(dataset_param["voxel_size_cam"])
        self.height_blfloor = dataset_param["height_belowfloor"]
        self.image_size = dataset_param["image_size"]
        self.sampler = dataset_param["sampler"]
        self.ds = dataset_param["voxel_downsample"]
        self.input_ds_num = dataset_param["input_ds_num"]
        self.gt_ds_num = dataset_param["groundtruth_ds_num"]
        self.save_opts = dataset_param["save_options"]

        (
            self.nyu894_class,
            self.nyu894to40_ids,
            self.nyu40to36_ids,
        ) = self._parse_class_mat(self.data_dir / "ClassMapping.mat")

        self.scene_models = natsorted((self.data_dir / "NYUCAD_3D/mat").glob("*.mat"))
        self.depth_check_files = natsorted(
            list((self.data_dir / "depthbin/NYUCADtrain").glob("*.png"))
            + list((self.data_dir / "depthbin/NYUCADtest").glob("*.png")),
            key=lambda x: x.stem,
        )
        self.bin_check_files = natsorted(
            list((self.data_dir / "depthbin/NYUCADtrain").glob("*.bin"))
            + list((self.data_dir / "depthbin/NYUCADtest").glob("*.bin")),
            key=lambda x: x.stem,
        )
        assert (
            len(self.scene_models)
            == len(self.depth_check_files)
            == len(self.bin_check_files)
        )

        sum_files_num = 0
        for mode in self.modes:
            self.files[mode] = natsorted(
                (self.data_dir / f"NYUCAD{mode}_npz").glob("*.npz")
            )
            sum_files_num += len(self.files[mode])
        logger.info(f"Total files: {sum_files_num}.")

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
            model_path,
        ) = self._parse_id_scene_subscene_view(filepath.name)
        file_base = {
            "filepath": self.save_dir / mode / scene_name,
            "id": scene_id,
            "scene": scene,
            "sub_scene": sub_scene,
            "view": view,
            "scene_name": scene_name,
            "raw_filepath": filepath,
            "model_path": model_path,
        }

        # get ground truth
        scene_model = self._load_mat(model_path)["model"]
        gt_pcd, gt_pcd_crop, gt_mesh_o3d, *trans_param = self._gt_compile(scene_model)
        gt_mesh_colored_o3d = self._mesh_colored_o3d(gt_mesh_o3d)

        # get rgb and depth
        rgb, depth, *_ = self._load_npy(
            filepath, *("rgb", "depth", "target_hr", "target_lr")
        )
        depth = depth.squeeze()
        rgb = rgb.transpose(1, 2, 0)

        # get input
        input_pcd, input_pcd_crop, crop_idx = self._input_compile(depth, *trans_param)
        input_colored_crop = self._attach_rgb_to_pcd(input_pcd_crop, rgb, crop_idx)

        # (optional) check if Depth images etc. are equal in sscnet and SSC
        if self.check:
            try:
                self._check_data(
                    scene_id, **{"depth": depth, "trans_param": trans_param}
                )
            except Exception as e:
                logger.error(e)

        # downsample input and output
        input_colored_ds = pcd_downsample(input_colored_crop, self.input_ds_num)
        input_ds = input_colored_ds[:, :3]
        gt_ds = pcd_downsample(gt_pcd_crop, self.gt_ds_num)

        data_collection = {
            "data": {
                "input": input_ds,
                "gt": gt_ds,
                "rgb": rgb,
                "depth": depth,
                "input_colored": input_colored_ds,
                "mesh_colored": gt_mesh_colored_o3d,
            }
        }
        save_dir = self.save_dir / mode
        save_dir.mkdir(exist_ok=True, parents=False)
        self._save_data(
            save_dir,
            file_base["scene_name"],
            **{
                **data_collection,
                **self.save_opts,
            },
        )

        return file_base

    def _parse_id_scene_subscene_view(self, name):
        scene_id = int(re.match(r"NYU(\d{4})_0000", name).group(1))
        model_path = self.scene_models[scene_id - 1]
        scene_name = model_path.stem
        match = re.match(r"(\d{1,4})_(\w+)_(\d{4})_(\d+)", scene_name)
        scene_id_ck, scene, sub_scene, view = (
            int(match.group(1)),
            str(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        )
        assert scene_id == scene_id_ck
        return scene_id, scene, sub_scene, view, scene_name, model_path

    def _parse_class_mat(self, fl_name):
        class_mat = self._load_mat(fl_name)
        assert isinstance(class_mat, dict)
        keys = [
            "elevenClass",
            "nyu894class",
            "mapNYU894To40",
            "nyu40class",
            "mapNYU40to36",
            "p5d36class",
        ]
        (
            eleven_class,
            nyu894_class,
            nyu894to40,
            nyu40_class,
            nyu40to36,
            p5d36_class,
        ) = itemgetter(*keys)(class_mat)
        assert list(eleven_class) == list(self.class_map.keys())[1:-1]
        _, nyu894to40_ids = ismember(nyu894to40, nyu40_class)
        _, nyu40to36_ids = ismember(nyu40to36, p5d36_class)
        nyu894to40_ids = np.insert(nyu894to40_ids + 1, 0, 0)
        nyu40to36_ids = np.insert(nyu40to36_ids + 1, 0, 0)
        return nyu894_class, nyu894to40_ids, nyu40to36_ids

    @classmethod
    def _load_npy(cls, fl_name, *key):
        with np.load(str(fl_name)) as fl:
            content = itemgetter(*key)(fl)
        return content

    @classmethod
    def _load_mat(cls, fl_name):
        return scipy.io.loadmat(str(fl_name), struct_as_record=False, squeeze_me=True)

    def _gt_compile(self, scene_model):
        floor_id = 0
        for obj_id in range(len(scene_model.objects)):
            if scene_model.objects[obj_id].model.label == "floor":
                floor_id = obj_id
                break
        floor_height = scene_model.objects[floor_id].model.surfaces.polygon.pts[0].y

        ext_cam2world = self._get_extrinsics(floor_height, scene_model.camera.R.T)
        cam_pose, vox_origin_inwld = self._get_vox_origin_inwld(ext_cam2world)

        vertices, surfaces = self._get_mesh(scene_model, floor_height, vox_origin_inwld)
        mesh_o3d = self._build_mesh_o3d(vertices[:, :3], surfaces)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertices[:, 3:] / 255)
        pcd = self._get_pcd_from_mesh_o3d(mesh_o3d)

        pcd_inview_idx = self._get_pcd_inview_idx(
            pcd[:, :3], vox_origin_inwld, ext_cam2world
        )
        pcd_inview = pcd[pcd_inview_idx]
        pcd_invox_idx = self._get_pcd_invox_idx(
            np.rint(pcd_inview[:, :3] / self.vox_unit), self.vox_size_model
        )
        pcd_crop = pcd_inview[pcd_invox_idx]
        pcd_crop[:, [0, 1, 2]] = (
            pcd_crop[
                :,
                [1, 2, 0],
            ]
            / self.ds
        )
        return pcd, pcd_crop, mesh_o3d, cam_pose, vox_origin_inwld

    @classmethod
    def _get_extrinsics(cls, transform_z, rotation):
        transform = np.array([0.0, 0.0, -transform_z])
        mat1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]).astype(float)
        mat2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).astype(float)
        mat3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(float)
        rot = mat1 @ mat2 @ rotation @ mat3
        ext = np.hstack((rot, transform.reshape(-1, 1)))
        return ext

    def _get_vox_origin_inwld(self, ext_cam2world):
        vox_range = np.array(
            [
                np.append(-self.vox_size_cam[:2] * self.vox_unit / 2.0, 0),
                np.append(-self.vox_size_cam[:2] * self.vox_unit / 2.0, 2)
                + self.vox_size_cam * self.vox_unit,
            ]
        ).T
        cam_pose = np.hstack((ext_cam2world.T, [[0], [0], [0], [1]])).flatten("F")
        vox_origin_incam = np.mean(vox_range, 1)
        vox_origin_inwld = (
            ext_cam2world[:3, :3] @ vox_origin_incam
            + ext_cam2world[:3, 3]
            - self.vox_size_model / 2 * self.vox_unit
        )
        vox_origin_inwld[2] = self.height_blfloor
        return cam_pose, vox_origin_inwld

    def _get_mesh(self, scene_model, floor_height, origin):
        pts_cnt, vertices, surfaces = 0, np.empty((0, 6)), np.empty((0, 3))
        for obj_id in range(len(scene_model.objects)):
            obj = scene_model.objects[obj_id]
            obj_cls = self._class_mapping(obj.model.label)
            obj_mesh = obj.mesh
            comp_len = (
                len(obj_mesh.comp) if isinstance(obj_mesh.comp, np.ndarray) else 1
            )
            for comp_id in range(comp_len):
                comp = obj_mesh.comp if comp_len == 1 else obj_mesh.comp[comp_id]
                comp_faces = comp.faces.reshape(-1, 3).astype(int)  # avoid overflow
                acc_faces = comp_faces + pts_cnt - 1
                surfaces = np.concatenate((surfaces, acc_faces))
                comp_vertices = comp.vertices.reshape(-1, 3)
                comp_labels = np.full((comp_vertices.shape[0], 3), obj_cls)
                comp_vertices = np.hstack((comp_vertices, comp_labels))
                vertices = np.concatenate((vertices, comp_vertices))
                pts_cnt += len(comp.vertices)
        surfaces = surfaces.astype(int)
        vertices[:, 1] = vertices[:, 1] - floor_height
        vertices[:, [0, 2, 1]] = vertices[:, [0, 1, 2]]
        vertices[:, 1] = -vertices[:, 1]
        vertices[:, :3] -= origin  # shift to voxel position
        vertices = vertices

        return vertices, surfaces

    def _class_mapping(self, label) -> np.ndarray:
        obj_nyu894_id = int(ismember([label], self.nyu894_class)[1])
        obj_cls = self.nyu36to11_ids[
            self.nyu40to36_ids[self.nyu894to40_ids[obj_nyu894_id + 1]]
        ]
        return obj_cls

    @classmethod
    def _build_mesh_o3d(cls, verts, surfs):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(surfs)
        return mesh_o3d

    def _get_pcd_from_mesh_o3d(self, mesh_o3d):
        if self.sampler["name"] == "poisson_disk_sampling":
            pcd_o3d = mesh_o3d.sample_points_poisson_disk(
                self.sampler["pts_number"], self.sampler["factor"]
            )
        elif self.sampler["name"] == "uniform_sampling":
            pcd_o3d = mesh_o3d.sample_points_uniformly(self.sampler["pts_number"])
        else:
            raise f"{self.sampler['name']} does not exist"
        points = np.asarray(pcd_o3d.points)
        labels = np.around(np.asarray(pcd_o3d.colors) * 255)
        return np.concatenate((points, labels), axis=1)

    def _get_pcd_inview_idx(self, points, vox_origin_inwld, ext_cam2world):
        ext_world2cam = np.linalg.inv(np.vstack((ext_cam2world, [0, 0, 0, 1])))
        pts_incam = ext_world2cam[:3, :3] @ (
            points + vox_origin_inwld  # vox origin is subtracted in mesh
        ).T + ext_world2cam[:3, 3].reshape(-1, 1)
        pix_x = pts_incam[0, :] * self.cam_k[0, 0] / pts_incam[2, :] + self.cam_k[0, 2]
        pix_y = pts_incam[1, :] * self.cam_k[1, 1] / pts_incam[2, :] + self.cam_k[1, 2]
        pcd_inview_idx = np.where(
            ~(
                (pix_x <= 0)
                | (pix_x > self.image_size["x"])
                | (pix_y <= 0)
                | (pix_y > self.image_size["y"])
            )
        )
        return pcd_inview_idx

    @classmethod
    def _get_pcd_invox_idx(cls, points, vox_size):
        pcd_invox_idx = np.where(
            (0 <= points[:, 0])
            & (points[:, 0] < vox_size[0])
            & (0 <= points[:, 1])
            & (points[:, 1] < vox_size[1])
            & (0 <= points[:, 2])
            & (points[:, 2] < vox_size[2])
        )
        return pcd_invox_idx

    def _mesh_colored_o3d(self, mesh_o3d):
        labels = np.array(mesh_o3d.vertex_colors)[:, 0] * 255
        colors = self.color_map[labels.astype(int)] / 255
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh_o3d

    def _input_compile(self, depth, cam_pose, vox_origin_inwld):
        cam_pose = cam_pose.reshape((4, 4))
        vox_size = self.vox_size // self.ds
        pts_incam = self._depth_to_pcd(depth)

        # ---- Get point in world coordinate
        pts_inwld = self._pcd_cam_to_world(pts_incam, cam_pose)

        # align with voxel model
        pts_inwld = pts_inwld - vox_origin_inwld
        # crop depth point cloud
        pts_inwld = pts_inwld.reshape(-1, 3) / self.ds
        pts_inwld[:, [0, 1, 2]] = pts_inwld[:, [0, 2, 1]]

        pts_invox_idx = self._get_pcd_invox_idx(
            np.rint(pts_inwld / self.vox_unit), vox_size
        )
        pts_inwld_crop = pts_inwld[pts_invox_idx]
        # voxel_binary = self._convert_pcd_to_binvox(pts_inwld_crop, vox_size)
        pts_inwld_crop[:, [0, 1, 2]] = pts_inwld_crop[:, [2, 1, 0]]
        return pts_inwld, pts_inwld_crop, pts_invox_idx

    def _depth_to_pcd(self, depth):
        grid_x, grid_y = np.meshgrid(
            range(self.image_size["x"]), range(self.image_size["y"])
        )
        pts_incam = np.zeros(
            (self.image_size["y"], self.image_size["x"], 3), dtype=np.float32
        )
        pts_incam[:, :, 0] = (grid_x - self.cam_k[0][2]) * depth / self.cam_k[0][0]
        pts_incam[:, :, 1] = (grid_y - self.cam_k[1][2]) * depth / self.cam_k[1][1]
        pts_incam[:, :, 2] = depth  # z, in meter
        return pts_incam

    def _pcd_cam_to_world(self, pcd_incam, cam_pose):
        pts_inwld = np.zeros(
            (self.image_size["y"], self.image_size["x"], 3), dtype=float
        )
        pts_inwld[:, :, 0] = (
            cam_pose[0][0] * pcd_incam[:, :, 0]
            + cam_pose[0][1] * pcd_incam[:, :, 1]
            + cam_pose[0][2] * pcd_incam[:, :, 2]
            + cam_pose[0][3]
        )
        pts_inwld[:, :, 1] = (
            cam_pose[1][0] * pcd_incam[:, :, 0]
            + cam_pose[1][1] * pcd_incam[:, :, 1]
            + cam_pose[1][2] * pcd_incam[:, :, 2]
            + cam_pose[1][3]
        )
        pts_inwld[:, :, 2] = (
            cam_pose[2][0] * pcd_incam[:, :, 0]
            + cam_pose[2][1] * pcd_incam[:, :, 1]
            + cam_pose[2][2] * pcd_incam[:, :, 2]
            + cam_pose[2][3]
        )
        return pts_inwld

    @classmethod
    def _convert_pcd_to_binvox(cls, pcd, voxel_size):
        voxel_binary = np.zeros(voxel_size, dtype=np.float32)
        for i_idx in range(len(pcd)):
            i_x, i_y, i_z = pcd[i_idx, :]
            if (
                0 <= i_x < voxel_size[0]
                and 0 <= i_y < voxel_size[1]
                and 0 <= i_z < voxel_size[2]
            ):
                voxel_binary[i_x][i_y][i_z] = 1
        return voxel_binary

    @classmethod
    def _attach_rgb_to_pcd(cls, points, rgb, crop_idx=None):
        rgb_flat = rgb.reshape(-1, 3)
        if crop_idx is not None:
            rgb_flat = rgb_flat[crop_idx]
        assert len(points) == len(rgb_flat)
        return np.concatenate((points, rgb_flat), axis=1)

    def _check_data(self, scene_id, **check_part):
        with open(self.bin_check_files[scene_id - 1], "rb") as bin_fl:
            vox_origin_ck = np.fromfile(
                bin_fl, np.float32, 3
            ).T  # Read voxel origin in world coordinates
            cam_pose_ck = np.fromfile(bin_fl, np.float32, 16)  # Read camera pose
            _ = np.squeeze(
                np.fromfile(bin_fl, np.uint32).reshape((-1, 1)).T
            )  # Read voxel label data from file

        depth_ck = (
            cv2.imread(str(self.depth_check_files[scene_id - 1]), cv2.IMREAD_UNCHANGED)
            / 8000.0
        )  # read depth image

        if (trans_param := check_part.get("trans_param")) is not None:
            assert np.allclose(
                cam_pose_ck, trans_param[0]
            ), f"scene {scene_id}: Inconsistency of camera pose is detected"
            assert np.allclose(
                vox_origin_ck, trans_param[1], rtol=1e-4
            ), f"scene {scene_id}: Inconsistency of voxel origin is detected"

        # compare depth image
        if (depth := check_part.get("depth")) is not None:
            assert np.allclose(
                depth, depth_ck
            ), f"scene {scene_id}: Inconsistency of depth image is detected"

    def _save_data(self, save_dir, scene, **save_options):
        data_collect = save_options["data"]
        save_npy_opts = save_options["save_npy"]
        visual_opts = save_options["visualization"]

        if save_npy_opts.get("input") is not None:
            np.save(save_dir / f"{scene}_input.npy", data_collect["input"])
        if save_npy_opts.get("ground_truth") is not None:
            np.save(save_dir / f"{scene}_gt.npy", data_collect["gt"][:, :4])
        if save_npy_opts.get("rgb") is not None:
            np.save(save_dir / f"{scene}_rgb.npy", data_collect["rgb"])
        if save_npy_opts.get("depth") is not None:
            np.save(save_dir / f"{scene}_depth.npy", data_collect["depth"])
        if save_npy_opts.get("input_colored") is not None:
            np.save(
                save_dir / f"{scene}_inputcolored.npy", data_collect["input_colored"]
            )

        if visual_opts.get("input") is not None:
            save_pcd_ply(save_dir / f"{scene}_vis_input.ply", data_collect["input"])
        if visual_opts.get("rgb") is not None:
            rgb_img = self._rgb_to_img(data_collect["rgb"])[:, :, [2, 1, 0]]  # rgb->bgr
            cv2.imwrite(str(save_dir / f"{scene}_vis_rgb.png"), rgb_img)
        if visual_opts.get("ground_truth") is not None:
            data_collect["gt"][:, 3:6] = (
                self.color_map[data_collect["gt"][:, 3].astype(int)] / 255
            )
            save_pcd_ply(
                save_dir / f"{scene}_vis_gt.ply", data_collect["gt"], colored=True
            )
        if visual_opts.get("depth") is not None:
            depth_img = self._depth_to_img(data_collect["depth"])
            cv2.imwrite(str(save_dir / f"{scene}_vis_depth.png"), depth_img)
        if visual_opts.get("input_colored") is not None:
            data_collect["input_colored"][:, 3:6] = (
                self._rgb_to_img(data_collect["input_colored"][:, 3:6]) / 255.0
            )
            save_pcd_ply(
                save_dir / f"{scene}_vis_inputcolored.ply",
                data_collect["input_colored"],
                colored=True,
            )
        if visual_opts.get("mesh_colored") is not None:
            o3d.io.write_triangle_mesh(
                str(save_dir / f"{scene}_vis_mesh.ply"), data_collect["mesh_colored"]
            )

    @classmethod
    def _rgb_to_img(cls, rgb) -> np.ndarray:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (rgb * std + mean) * 255.0
        return img

    @classmethod
    def _depth_to_img(cls, depth):
        return (depth * 8000.0).astype(np.uint16)


if __name__ == "__main__":
    nyucad_preprocess = NYUCADPreprocessing(n_jobs=20)
    nyucad_preprocess.preprocess()
