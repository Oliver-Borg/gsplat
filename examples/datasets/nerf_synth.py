import json
from typing import Iterable
import cv2
from line_profiler import profile
import numpy as np
import os

from .normalize import transform_cameras


def get_rays_np(H, W, K, c2w):
    """
    Get ray origins, directions from a pinhole camera.
    Adapted from https://github.com/bmild/nerf
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    fx_i = K[0, 0]
    fy_i = K[1, 1]
    cx_i = K[0, 2]
    cy_i = K[1, 2]
    dirs = np.stack([(i - cx_i) / fx_i, (j - cy_i) / fy_i, np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


class SimpleParser:
    """A simple parser for JSON transforms data."""

    def __init__(
        self,
        path: str,
        test_every: int = 8,
        factor: int = 1,
        transform: np.ndarray = np.eye(4),
        max_points: int = 100000,
        create_pointcloud: bool = False,
    ):
        self.image_names = []
        self.camtoworlds = []
        self.Ks_dict = {}
        self.imsize_dict = {}
        self.camera_ids = []
        self.points = np.empty((0, 3))
        self.points_rgb = np.empty((0, 3))
        self.image_paths = []
        self.mapx_dict = {}
        self.mapy_dict = {}
        self.max_points = max_points

        self.test_every = test_every
        self.path = path
        self.params_dict = {}
        self.roi_undist_dict = {}  # This should be empty if there is no distortion
        self.mask_dict = {}
        self.depths = {}
        self.depths_conf = {}
        self.transform = transform
        self.factor = factor
        self.R_align = np.eye(3)
        self.t_align = np.zeros(3)

        self._load_json_data(create_pointcloud)

    @profile
    def _load_json_data(self, create_pointcloud: bool = False):
        with open(self.path, "r") as f:
            data = json.load(f)

        cam_angle = data.get("camera_angle_x", 0.0)
        frames = data.get("frames", [])
        frames = sorted(frames, key=lambda x: x["file_path"])

        def get_intrinsics(w: int, h: int):

            fl_x = 0.5 * w / np.tan(0.5 * cam_angle)
            fl_y = fl_x
            cx = w / 2.0
            cy = h / 2.0

            return w, h, fl_x, fl_y, cx, cy

        num_frames = len(frames)
        max_points = self.max_points
        points_per_frame = max_points // num_frames

        for i, frame in enumerate(frames):
            fname = frame["file_path"] + ".png"
            base_dir = os.path.dirname(self.path)
            im_path = os.path.join(base_dir, fname)
            if not os.path.exists(im_path):
                continue

            im_header = cv2.imread(im_path)
            w_i, h_i, fx_i, fy_i, cx_i, cy_i = get_intrinsics(im_header.shape[1], im_header.shape[0])

            name = os.path.basename(fname)
            self.image_names.append(name)

            im_folder = os.path.dirname(im_path)
            existing_files = os.listdir(im_folder)
            im_name = name.split(".")[0]
            depth_fname = (
                list(filter(lambda x: "depth" in x and x.startswith(im_name + "_depth"), existing_files))
                or [im_name + "_depth_0001.png"]
            )[0]

            full_depth_path = os.path.join(im_folder, depth_fname)
            if os.path.exists(full_depth_path):
                depth = cv2.imread(full_depth_path, cv2.IMREAD_UNCHANGED)
                # 0 is background
                if len(depth.shape) > 2:
                    depth = depth[..., 0]
                self.depths[name] = (255.0 - depth.astype(np.float32)) / 255.0 * 8.0
                self.depths[name][depth == 0] = np.nan

            c2w = np.array(frame["transform_matrix"])
            c2w[0:3, 1:3] *= -1
            self.camtoworlds.append(c2w)

            K = np.eye(3)
            K[0, 0] = fx_i
            K[1, 1] = fy_i
            K[0, 2] = cx_i
            K[1, 2] = cy_i

            cam_id = i
            self.camera_ids.append(cam_id)
            self.Ks_dict[cam_id] = K
            self.imsize_dict[cam_id] = (int(w_i), int(h_i))
            self.image_paths.append(im_path)
            self.params_dict[cam_id] = np.empty(0, dtype=np.float32)
            self.mask_dict[cam_id] = None

            if name in self.depths and create_pointcloud:
                # Unproject depths to 3D points
                depth_map = self.depths[name]
                us = np.arange(depth_map.shape[0])
                vs = np.arange(depth_map.shape[1])
                vs, us = np.meshgrid(vs, us)
                valid_mask: np.ndarray = ~np.isnan(depth_map)  # h, w

                # Randomly sample points_per_frame points
                if valid_mask.sum() > points_per_frame:
                    valid_mask = valid_mask.reshape(-1)
                    indices = np.random.choice(np.arange(valid_mask.sum()), points_per_frame, replace=False)
                    true_indices = np.where(valid_mask)[0]
                    indices = true_indices[indices]
                    valid_mask[:] = False
                    valid_mask[indices] = True
                    valid_mask = valid_mask.reshape(depth_map.shape)

                vs = vs[valid_mask]
                us = us[valid_mask]

                # Calculate rays only for valid sampled points
                fx_k, fy_k = K[0, 0], K[1, 1]
                cx_k, cy_k = K[0, 2], K[1, 2]

                dirs = np.stack([(vs - cx_k) / fx_k, (us - cy_k) / fy_k, np.ones_like(vs)], axis=-1)
                rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
                norm = np.linalg.norm(rays_d, axis=-1, keepdims=True)
                rays_d = rays_d / norm

                rays_o = c2w[:3, -1]
                world_points = rays_o + rays_d * depth_map[valid_mask][:, None]

                if im_header.shape[:2] != depth_map.shape[:2]:
                    im_header = cv2.resize(im_header, (depth_map.shape[1], depth_map.shape[0]))
                rgb_values = cv2.cvtColor(im_header, cv2.COLOR_BGR2RGB)[us, vs]

                self.points = np.concatenate([self.points, world_points])
                self.points_rgb = np.concatenate([self.points_rgb, rgb_values])

        self.camtoworlds = transform_cameras(self.transform, np.array(self.camtoworlds))

    def get_camera_positions(self, names: list[str]):
        indices = [self.image_names.index(name) for name in names]
        return np.array([self.camtoworlds[i] for i in indices])

    def get_camera_names(self, indices: Iterable[int]) -> list[str]:
        names = [self.image_names[i] for i in indices]
        return names


def load_json_data(path: str) -> SimpleParser:
    return SimpleParser(path)
