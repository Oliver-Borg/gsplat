import json
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

    frames = data.get("frames", [])
    frames = sorted(frames, key=lambda x: x["file_path"])

    def get_intrinsics(props):
        w = props.get("w")
        h = props.get("h")
        fl_x = props.get("fl_x")
        fl_y = props.get("fl_y")
        cx = props.get("cx")
        cy = props.get("cy")

        if w is None or h is None:
            return None

        if fl_x is None and "camera_angle_x" in props:
            fl_x = 0.5 * w / np.tan(0.5 * props["camera_angle_x"])
        if fl_y is None and "camera_angle_y" in props:
            fl_y = 0.5 * h / np.tan(0.5 * props["camera_angle_y"])
        if fl_y is None:
            fl_y = fl_x

        if cx is None:
            cx = w / 2.0
        if cy is None:
            cy = h / 2.0

        return w, h, fl_x, fl_y, cx, cy

    global_intrinsics = get_intrinsics(data)

    for i, frame in enumerate(frames):
        fname = frame["file_path"] + ".png"
        name = os.path.basename(fname)
        parser.image_names.append(name)

        c2w = np.array(frame["transform_matrix"])
        c2w[0:3, 1:3] *= -1
        parser.camtoworlds.append(c2w)

        frame_intrinsics = get_intrinsics(frame)
        if frame_intrinsics:
            w_i, h_i, fx_i, fy_i, cx_i, cy_i = frame_intrinsics
        elif global_intrinsics:
            w_i, h_i, fx_i, fy_i, cx_i, cy_i = global_intrinsics
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

        base_dir = os.path.dirname(path)
        parser.image_paths.append(os.path.join(base_dir, fname))

    return parser
