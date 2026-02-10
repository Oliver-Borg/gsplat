import json
import cv2
import numpy as np
import os


class SimpleParser:
    """A simple parser for JSON transforms data."""

    def __init__(self):
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


def load_json_data(path: str) -> SimpleParser:
    with open(path, "r") as f:
        data = json.load(f)

    parser = SimpleParser()
    cam_angle = data.get("camera_angle_x", 0.0)
    frames = data.get("frames", [])
    frames = sorted(frames, key=lambda x: x["file_path"])

    def get_intrinsics(w: int, h: int):

        fl_x = 0.5 * w / np.tan(0.5 * cam_angle)
        fl_y = fl_x
        cx = w / 2.0
        cy = h / 2.0

        return w, h, fl_x, fl_y, cx, cy
    
    

    for i, frame in enumerate(frames):
        fname = frame["file_path"] + ".png"
        base_dir = os.path.dirname(path)
        im_path = os.path.join(base_dir, fname)
        im = cv2.imread(im_path)
        frame_intrinsics = get_intrinsics(im.shape[0], im.shape[1])
        
        name = os.path.basename(fname)
        parser.image_names.append(name)

        c2w = np.array(frame["transform_matrix"])
        c2w[0:3, 1:3] *= -1
        parser.camtoworlds.append(c2w)

        
        if frame_intrinsics:
            w_i, h_i, fx_i, fy_i, cx_i, cy_i = frame_intrinsics
        else:
            w_i, h_i = 100, 100
            fx_i, fy_i, cx_i, cy_i = 100, 100, 50, 50

        K = np.eye(3)
        K[0, 0] = fx_i
        K[1, 1] = fy_i
        K[0, 2] = cx_i
        K[1, 2] = cy_i

        cam_id = i
        parser.camera_ids.append(cam_id)
        parser.Ks_dict[cam_id] = K
        parser.imsize_dict[cam_id] = (int(w_i), int(h_i))
        parser.image_paths.append(im_path)

    return parser
