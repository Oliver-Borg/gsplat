from dataclasses import dataclass, field, replace
import datetime
import json
from pathlib import Path
from typing import Literal
import os
import subprocess
import argparse
import concurrent.futures
from queue import Queue

from line_profiler import profile
from tqdm import tqdm

from vggt.plot_metrics import plot_graph, _parse_gsplat_json

import examples.evaluation

GSPLAT_PYTHON = os.path.expanduser("~/.conda/envs/gsplat/bin/python")
GSPLAT_TORCHRUN = os.path.expanduser("~/.conda/envs/gsplat/bin/torchrun")
VGGT_PYTHON = os.path.expanduser("~/.conda/envs/vggt/bin/python")


@dataclass
class Dataset:
    name: str
    factor: int
    directory: str
    data_folder_name: str
    gt_train_data_dir: str | None = None
    gt_eval_data_dir: str | None = None

    @property
    def scene_name(self):
        return f"{self.name}_{self.factor}"

    @property
    def eval_names(self):
        data_dir = Path(self.directory) / Path(self.data_folder_name)
        images = sorted(os.listdir(data_dir))
        return images[::8]


datasets = {
    "bicycle": Dataset(
        name="bicycle",
        factor=4,
        directory="../vggt/data/360_v2/bicycle",
        gt_eval_data_dir="../vggt/data/360_v2/bicycle",
        gt_train_data_dir="../vggt/data/360_v2/bicycle",
        data_folder_name="images_4",
    ),
    "bonsai": Dataset(
        name="bonsai",
        factor=2,
        directory="../vggt/data/360_v2/bonsai",
        gt_eval_data_dir="../vggt/data/360_v2/bonsai",
        gt_train_data_dir="../vggt/data/360_v2/bonsai",
        data_folder_name="images_2",
    ),
    "lego": Dataset(
        name="lego",
        factor=1,
        directory="../vggt/data/nerf_synthetic/lego",
        gt_train_data_dir="../vggt/data/nerf_synthetic/lego/transforms_train.json",
        gt_eval_data_dir="../vggt/data/nerf_synthetic/lego/transforms_val.json",
        data_folder_name="train",
    ),
    "bonsai_4": Dataset(
        name="bonsai",
        factor=4,
        directory="../vggt/data/360_v2/bonsai",
        gt_eval_data_dir="../vggt/data/360_v2/bonsai",
        gt_train_data_dir="../vggt/data/360_v2/bonsai",
        data_folder_name="images_4",
    ),
    "blender_radial": Dataset(
        name="blender_radial",
        factor=1,
        directory="../vggt/data/blender",
        data_folder_name="dataset_N100_R5.5_H1.0_C-4.0_0.0_2.0_SIMPLE_RADIAL_K-0.02_RMEevee_no_windows",
    ),
    "blender_pinhole": Dataset(
        name="blender_pinhole",
        factor=1,
        directory="../vggt/data/blender",
        data_folder_name="dataset_N100_R5.5_H1.0_C-4.0_0.0_2.0_SIMPLE_PINHOLE_K-0.02_RMEevee_no_windows",
    ),
}


@dataclass
class Config:
    choice: Literal["vggt", "colmap", "gt", "combined"] = "vggt"
    num_images: int = 30
    dataset: Dataset = datasets["lego"]
    seed: int = 42
    conf_thres_value: float = 0.0
    num_points_per_image: float = 1100
    num_points_value: int | None = None
    sampling_mode: Literal["voxels", "random", "confidence", "ba", "vox3"] = "voxels"
    image_mode: Literal["shuffle", "distributed", "mfps", "farthestpose"] = "farthestpose"
    copy_mode: Literal[None, "crop", "square", "tiles"] = None
    gt_eval: bool = True
    use_gt_extrinsics: bool = False
    use_gt_intrinsics: bool = False
    use_gt_points: bool = False
    pose_opt: bool = False
    eval_opt: bool = False
    all_opt: bool = False
    num_cameras: int | None = None
    depth_loss: bool = False
    depth_lambda: float = 0.0
    depth_conf: bool = False
    error_opa: bool = False
    camera_type: Literal["SIMPLE_RADIAL", "SIMPLE_PINHOLE"] = "SIMPLE_PINHOLE"
    num_steps: Literal[7000, 15000, 30000] = 15000
    colmap_mode: Literal["default", "relaxed"] = "default"
    nomcmc: bool = False
    camera_src: Literal["vggt", "colmap", "gt"] = "vggt"
    pcd_src: Literal["vggt", "colmap", "gt", "both"] = "vggt"
    align_mode: Literal["local", "global"] = "local"

    construction_data: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        instance = cls()
        instance.choice = data["choice"]
        instance.num_images = data.get("num_images", instance.num_images)
        instance.dataset = datasets[data.get("dataset", "lego")]
        instance.seed = data.get("seed", instance.seed)
        instance.conf_thres_value = data.get("conf_thres_value", instance.conf_thres_value)
        instance.num_points_per_image = data.get("num_points_per_image", instance.num_points_per_image)
        instance.num_points_value = data.get("num_points_value", instance.num_points_value)
        instance.sampling_mode = data.get("sampling_mode", instance.sampling_mode)
        instance.image_mode = data.get("image_mode", instance.image_mode)
        instance.copy_mode = data.get("copy_mode", instance.copy_mode)
        instance.gt_eval = data.get("gt_eval", instance.gt_eval)
        instance.use_gt_extrinsics = data.get("use_gt_extrinsics", instance.use_gt_extrinsics)
        instance.use_gt_intrinsics = data.get("use_gt_intrinsics", instance.use_gt_intrinsics)
        instance.use_gt_extrinsics |= data.get("use_gt_cams", False)
        instance.use_gt_intrinsics |= data.get("use_gt_cams", False)
        instance.use_gt_points = data.get("use_gt_points", instance.use_gt_points)
        instance.pose_opt = data.get("pose_opt", instance.pose_opt)
        instance.eval_opt = data.get("eval_opt", instance.eval_opt)
        instance.pose_opt |= data.get("all_opt", instance.all_opt)
        instance.eval_opt |= data.get("all_opt", instance.all_opt)
        instance.num_cameras = data.get("num_cameras", instance.num_cameras)
        instance.depth_loss = data.get("depth_loss", instance.depth_loss)
        instance.depth_lambda = data.get("depth_lambda", instance.depth_lambda)
        instance.depth_conf = data.get("depth_conf", instance.depth_conf)
        instance.error_opa = data.get("error_opa", instance.error_opa)
        instance.camera_type = data.get("camera_type", instance.camera_type)
        instance.num_steps = data.get("num_steps", instance.num_steps)
        instance.colmap_mode = data.get("colmap_mode", instance.colmap_mode)
        instance.nomcmc = data.get("nomcmc", instance.nomcmc)
        instance.camera_src = data.get("camera_src", instance.camera_src)
        instance.pcd_src = data.get("pcd_src", instance.pcd_src)
        instance.align_mode = data.get("align_mode", instance.align_mode)

        instance.construction_data = data
        return replace(instance)

    def __post_init__(self):

        if self.choice == "gt":
            self.gt_eval = False
            self.pose_opt = False
            self.eval_opt = False
            self.use_gt_extrinsics = False
            self.use_gt_intrinsics = False
            self.use_gt_points = False
            self.depth_conf = False
            self.num_images = len(os.listdir(Path(self.dataset.directory) / Path(self.dataset.data_folder_name)))
            self.num_cameras = self.num_images

        if self.choice == "combined" and self.camera_src == self.pcd_src:
            self.camera_src = "vggt" if self.pcd_src == "colmap" else "colmap"

        self.depth_conf = self.depth_conf and self.choice == "vggt"

        if self.depth_conf and self.choice == "vggt":
            self.depth_loss = True
        else:
            self.depth_conf = False

        if self.depth_lambda < 0.0:
            self.depth_lambda = 0.0

        if self.depth_lambda > 0.0:
            self.depth_loss = True

        if self.use_gt_extrinsics or self.use_gt_intrinsics or self.use_gt_points:
            self.gt_eval = True

        if self.eval_opt and self.gt_eval:  # TODO Figure out a good way to have both enabled
            self.eval_opt = False

    @property
    def num_points(self):
        if self.num_points_value is not None:
            return self.num_points_value
        return int(self.num_points_per_image * self.num_images)

    @property
    def num_cams(self):
        if self.num_cameras is not None:
            return self.num_cameras
        return self.num_images

    @property
    def input_name(self):
        parts = [
            self.dataset.name,
            str(self.dataset.factor),
            f"n{self.num_images}",
            f"s{self.seed}",
        ]

        if self.choice == "colmap":
            parts.append(self.colmap_mode)
            parts.append(self.image_mode)

            if self.copy_mode is not None:
                parts.append(self.copy_mode)
        elif self.choice == "vggt":
            if self.sampling_mode == "ba":
                parts.append(self.camera_type.lower().replace("simple_", "m"))
                parts.append(self.sampling_mode)
            else:
                parts.extend(
                    [
                        f"c{self.conf_thres_value}",
                        f"p{self.num_points}",
                        self.sampling_mode,
                    ]
                )

            if self.error_opa:
                parts.append("errconf")
            parts.append(self.image_mode)

            if self.copy_mode is not None:
                parts.append(self.copy_mode)
        elif self.choice == "combined":
            parts.append("combined")
            parts.append(f"{self.camera_src}cams")
            parts.append(f"{self.pcd_src}pcd")
            if self.align_mode == "local":
                parts.append("amlocal")
            else:
                parts.append("amglobal")

            parts.append(self.colmap_mode)

            if self.sampling_mode == "ba" and self.camera_src == "vggt":
                parts.append(self.camera_type.lower().replace("simple_", "m"))
                parts.append(self.sampling_mode)
            elif self.pcd_src == "vggt" or self.pcd_src == "both":
                parts.extend(
                    [
                        f"c{self.conf_thres_value}",
                        f"p{self.num_points}",
                        self.sampling_mode,
                    ]
                )

            if self.error_opa:
                parts.append("errconf")

            parts.append(self.image_mode)

            if self.copy_mode is not None:
                parts.append(self.copy_mode)

        elif self.choice == "gt":
            pass

        parts = "_".join(parts)
        return f"{self.choice}_outputs/{parts}"

    @property
    def output_name(self):
        parts = [
            f"i{self.num_cams}",
        ]
        if self.gt_eval:
            parts.append("gteval")
        if self.use_gt_extrinsics:
            parts.append("gtext")
        if self.use_gt_intrinsics:
            parts.append("gtint")
        if self.use_gt_points:
            parts.append("gtpcd")
        if self.pose_opt:
            parts.append("poseopt")
        if self.eval_opt:
            parts.append("evalopt")
        if self.depth_loss:
            parts.append("depth")
            parts.append(f"dl{self.depth_lambda}")
        if self.error_opa:
            parts.append("erroropa")
        if self.depth_conf and self.choice == "vggt" and self.depth_loss:
            parts.append("conf")
        if self.nomcmc:
            parts.append("nomcmc")
        parts.append(f"steps{self.num_steps}")

        parts = "_".join(parts)
        return f"{self.input_name}_{parts}"

    @property
    def result_dir(self):
        return f"./results/{self.output_name}"

    @property
    def renders_folder(self):
        return f"{self.result_dir}/renders"

    @property
    def splatting_val_path(self):
        return f"{self.result_dir}/stats/val_step{self.num_steps - 1}.json"

    @property
    def splatting_time(self):
        return Path(self.splatting_val_path).stat().st_mtime

    @property
    def data_dir(self):
        if self.choice == "gt":
            return self.dataset.directory

        return f"../vggt/{self.input_name}"

    @property
    def force_splat(self):

        # if self.depth_loss or self.depth_conf:
        #     return True

        if not self.is_splatted:
            return True

        if self.pose_opt and self.gt_eval and self.splatting_time < datetime.datetime(2026, 4, 16, 14, 0).timestamp():
            # Runs before this would not align before each evaluation
            return True

        if (
            not self.choice == "gt"
            and self.is_splatted
            and self.is_reconstructed
            and self.splatting_time < self.reconstruction_time
        ):
            return True

        if self.depth_loss and self.splatting_time < datetime.datetime(2026, 4, 23, 16, 00, 0).timestamp():
            return True

        if self.gt_eval and self.is_splatted:
            if len(list(filter(lambda x: "6999" in x, os.listdir(self.renders_folder)))) < len(self.dataset.eval_names):
                return True
        return False
        return self.sampling_mode == "ba" and self.camera_type == "SIMPLE_RADIAL" and self.choice == "vggt"

    @property
    def force_reconstruct(self):
        if self.choice == "gt":
            return False

        # if self.error_opa:
        #     return True

        # if self.sampling_mode == "confidence":
        #     return True

        if self.choice == "combined" and self.align_mode == "local" and self.is_reconstructed:
            if self.reconstruction_time < datetime.datetime(2026, 4, 23, 15, 50, 0).timestamp():
                return True

        if self.is_splatted and self.choice == "vggt":
            with open(self.splatting_val_path, "r") as f:
                stats = _parse_gsplat_json(json.load(f), self.splatting_val_path)
                psnr = stats.get("psnr", 0.0)
            if psnr is None or (psnr < 15 and self.num_images > 20):
                return True
            return False

        return False

    @property
    def reconstruction_stat_path(self):
        return os.path.join(self.data_dir, "stat.json")

    @property
    def reconstruction_time(self):
        return Path(self.reconstruction_stat_path).stat().st_mtime

    @property
    def is_reconstructed(self):
        return self.choice == "gt" or os.path.exists(self.reconstruction_stat_path)

    @property
    def reconstruct_args(self):
        args = {
            "input": (Path(self.dataset.directory) / Path(self.dataset.data_folder_name)).absolute().as_posix(),
            "name": f"{self.dataset.name}_{self.dataset.factor}",
            "choice": self.choice,
            "num_images": self.num_images,
            "num_points": self.num_points,
            "seed": self.seed,
            "conf_thres_value": self.conf_thres_value,
            "sampling_mode": self.sampling_mode,
            "image_mode": self.image_mode,
            "camera_type": self.camera_type,
            "colmap_mode": self.colmap_mode,
        }

        if (self.depth_conf or self.depth_loss or self.error_opa) and self.choice == "vggt":
            args["require_depth_conf"] = True
        if self.error_opa:
            args["save_conf_as_errors"] = True
        if self.force_reconstruct:
            args["force"] = True

        if self.copy_mode is not None:
            args["copy_mode"] = self.copy_mode

        return args

    def replace_choice(self, choice: Literal["vggt", "colmap", "gt"]) -> "Config":
        return Config.from_dict({**self.construction_data, "choice": choice})

    def reconstruct(self, force: bool = False):
        force |= self.force_reconstruct
        if self.is_reconstructed and not force:
            print(Path(self.data_dir), "has already been constructed.\nUse --force to force reconstruction.")
            return 0

        if self.choice == "combined":
            configs = {
                "vggt": self.replace_choice("vggt"),
                "colmap": self.replace_choice("colmap"),
                "gt": self.replace_choice("gt"),
            }

            configs[self.camera_src].reconstruct(force)
            pcd_choices = ["vggt", "colmap"] if self.pcd_src == "both" else [self.pcd_src]
            for pcd_src in pcd_choices:
                configs[pcd_src].reconstruct(force)

            cam_config = configs[self.camera_src]
            if self.pcd_src == "both":
                if self.camera_src == "vggt":
                    pcd_config = configs["colmap"]
                elif self.camera_src == "colmap" or self.camera_src == "gt":
                    pcd_config = configs["vggt"]
                else:
                    raise ValueError(f"Unrecognised camera src {self.camera_src}")
            else:
                pcd_config = configs[self.pcd_src]

            command = [
                VGGT_PYTHON,
                "-m",
                "combine_clouds",
            ]

            command.extend(["--camera_source", cam_config.data_dir])
            command.extend(["--point_source", pcd_config.data_dir])
            if self.pcd_src == "both":
                command.extend(["--use_both_pcds"])
            command.extend(["--output_dir", self.data_dir])
            if self.align_mode == "local":
                command.append("--align_each_point_set")

            try:
                output = subprocess.run(command, check=True, cwd="../vggt")
                print(output)
                return output.returncode
            except subprocess.CalledProcessError as e:
                print(e)
                return e.returncode

        command = [
            VGGT_PYTHON,
            "-m",
            "reconstruct",
            "single",
        ]
        for key, value in self.reconstruct_args.items():
            if (key == "force" or key == "require_depth_conf" or key == "save_conf_as_errors") and value is True:
                command.extend([f"--{key}"])
            elif value is not None:
                command.extend([f"--{key}", str(value)])

        try:
            output = subprocess.run(command, check=True, cwd="../vggt")
            print(output)
            return output.returncode
        except subprocess.CalledProcessError as e:
            print(e)
            return e.returncode

    @property
    def eval_path(self):
        return os.path.join(self.data_dir, "eval_results.json")

    @property
    def eval_time(self):
        return Path(self.eval_path).stat().st_mtime

    @property
    def is_evaluated(self):
        return os.path.exists(self.eval_path)

    @property
    def force_eval(self):
        if not self.is_evaluated:
            return True

        threshold_date = datetime.datetime(2026, 4, 4, 11, 20)  # This is when I fixed the eval script
        threshold_timestamp = threshold_date.timestamp()
        file_timestamp = self.eval_time

        if file_timestamp < threshold_timestamp:
            return True

        if self.is_reconstructed and not self.choice == "gt" and file_timestamp < self.reconstruction_time:
            return True
        return False

    def eval(self):
        if self.is_evaluated and not self.force_eval:
            return 0
        try:
            examples.evaluation.main(self.data_dir, self.dataset.directory, force=self.force_eval)
            return 0
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 1

    @property
    def is_splatted(self):
        return os.path.exists(self.splatting_val_path)

    @profile
    def run(self, gpu: str | None = None, force_splat: bool = False):
        force_splat = self.force_splat or force_splat
        if self.is_splatted and not force_splat:
            print(f"{self.splatting_val_path} found. Skipping splatting")
            return 0

        if self.force_splat:
            print("Forcing splatting")
        else:
            print(f"{Path(self.splatting_val_path)} not found. Running splatting")
        print(f"Using data from: {Path(self.data_dir)}")
        print(f"Result dir: {Path(self.result_dir)}")
        command = [
            GSPLAT_PYTHON,
            "examples/simple_trainer.py",
            "default" if self.nomcmc else "mcmc",
            "--data_dir",
            self.data_dir,
            "--data_factor",
            str(self.dataset.factor) if self.choice == "gt" else "1",  # We resize the data when copying in VGGT already
            "--eval_data_factor",
            str(self.dataset.factor),
            "--result-dir",
            self.result_dir,
            "--disable_viewer",
            "--max_train_cameras",
            str(self.num_cams),
            "--max_steps",
            str(self.num_steps),
        ]

        if self.pose_opt:
            command.append("--pose_opt")
        if self.gt_eval:
            if self.dataset.gt_train_data_dir is not None:
                command.extend(
                    [
                        "--gt_train_data_dir",
                        self.dataset.gt_train_data_dir,
                    ]
                )
            if self.dataset.gt_eval_data_dir is not None:
                command.extend(
                    [
                        "--gt_eval_data_dir",
                        self.dataset.gt_eval_data_dir,
                    ]
                )
        if self.use_gt_extrinsics:
            command.append("--use_gt_extrinsics")
        if self.use_gt_intrinsics:
            command.append("--use_gt_intrinsics")
        if self.use_gt_points:
            command.append("--use_gt_points")
        if self.eval_opt:
            command.append("--eval_opt")

        if self.depth_loss:
            command.append("--depth_loss")
            command.append("--depth_lambda")
            command.append(str(self.depth_lambda))

        if self.depth_conf:
            command.append("--depth_conf")

        if self.error_opa:
            command.append("--error_opa")

        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu

        try:
            output = subprocess.run(command, check=True, env=env)
            print(output)
            return output.returncode
        except subprocess.CalledProcessError as e:
            print(e)
            return e.returncode


def generate_configs(
    experiment_config: dict[str, str | float | bool | int | list[str | float | bool | int]],
) -> list[dict]:
    """
    Recurse through a config and create new configs with the lists unrolled.
    """
    if not any([isinstance(val, list) for val in experiment_config.values()]):
        return [experiment_config]
    to_return = []
    for key, val in experiment_config.items():
        if isinstance(val, list):
            for item in val:
                next_dict = experiment_config.copy()
                next_dict[key] = item
                to_return.extend(generate_configs(next_dict))
            break
    return to_return


@dataclass
class PlotConfig:
    x_axis: str
    split_param: str | None = None
    filter: str | None = None
    metric_keys: list[str] = field(default_factory=lambda: ["rre", "rte", "psnr", "lpips", "ssim", "num_GS"])
    single_legend: bool = True
    split_choice: bool = False


@dataclass
class Experiment:
    name: str
    group: int
    description: str
    config_dict: dict
    plot_args: PlotConfig | None = None
    include_gt: bool = False
    val_steps: list[int] = field(default_factory=lambda: [15000])
    render_filter_override: dict = field(default_factory=lambda: {})

    @property
    def render_config_dict(self):
        render_config_dict = self.config_dict.copy()
        render_config_dict.update(self.render_filter_override)
        return render_config_dict

    def get_configs(self, dataset_name: str, renders: bool = False) -> list[Config]:
        self.config_dict["dataset"] = dataset_name
        config_dicts = generate_configs(self.render_config_dict if renders else self.config_dict)
        configs = [Config.from_dict(config_dict) for config_dict in config_dicts]
        gt_configs = []
        if self.include_gt:
            for config in configs:
                gt_config = replace(config, choice="gt")
                gt_configs.append(gt_config)

        configs += gt_configs
        config_set = set()
        unique_configs: list[Config] = []
        for config in configs:
            if (config.result_dir, config.splatting_val_path, config.data_dir) not in config_set:
                unique_configs.append(config)
            config_set.add((config.result_dir, config.splatting_val_path, config.data_dir))
        return unique_configs

    def get_render_configs(self, dataset_name: str) -> list[Config]:
        return self.get_configs(dataset_name, renders=True)

    def bulk_reconstruct(self, configs: list[Config]):
        reconstruct_args = []
        for config in configs:
            if config.choice == "combined":
                for choice in ("vggt", "colmap"):
                    reconstruct_args.append(config.replace_choice(choice).reconstruct_args)
            else:
                reconstruct_args.append(config.reconstruct_args)

        temp_dir = Path("../vggt/configs/")
        temp_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config_path = temp_dir / f"reconstruct_args_{self.name}_{timestamp}.json"
        with open(config_path, "w") as f:
            json.dump(reconstruct_args, f, indent=4)

        command = [
            VGGT_PYTHON,
            "-m",
            "reconstruct",
            "batch",
            "--config_path",
            config_path.absolute().as_posix(),
        ]

        output = subprocess.run(command, check=True, cwd="../vggt")

        for config in configs:
            if config.choice == "combined":
                config.reconstruct()

    def run(
        self,
        dataset_name: str,
        do_reconstruct: bool = True,
        do_splatting: bool = True,
        bulk_reconstruct: bool = True,
        force_all: bool = False,
        force_none: bool = False,
        cuda_devices: list[str] | None = None,
    ):
        print(self.progress_stats(dataset_name))
        configs = self.get_configs(dataset_name)
        splat_failures: list[Config] = []
        eval_failures: list[Config] = []

        if do_reconstruct and bulk_reconstruct:
            self.bulk_reconstruct(configs)

        if cuda_devices:
            gpu_queue = Queue()
            for gpu in cuda_devices:
                gpu_queue.put(gpu)

            def process_config(config: Config):
                gpu = gpu_queue.get()
                splat_fail = False
                eval_fail = False
                try:
                    if do_reconstruct and not bulk_reconstruct:
                        reconstruction_returncode = config.reconstruct(force_all)
                        if reconstruction_returncode != 0:
                            return config, True, False

                    eval_returncode = config.eval()
                    if eval_returncode != 0:
                        eval_fail = True

                    if do_splatting:
                        returncode = config.run(gpu=gpu, force_splat=force_all)
                        if returncode != 0:
                            splat_fail = True
                    return config, splat_fail, eval_fail
                finally:
                    gpu_queue.put(gpu)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(cuda_devices)) as executor:
                futures = [executor.submit(process_config, config) for config in configs]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(configs)):
                    config, splat_fail, eval_fail = future.result()
                    if splat_fail:
                        splat_failures.append(config)
                    if eval_fail:
                        eval_failures.append(config)
        else:
            for config in tqdm(configs):
                if do_reconstruct and not bulk_reconstruct:
                    reconstruction_returncode = config.reconstruct(force=force_all)
                    if reconstruction_returncode != 0:
                        splat_failures.append(config)
                        continue

                eval_returncode = config.eval()
                if eval_returncode != 0:
                    eval_failures.append(config)

                if do_splatting:
                    returncode = config.run(force_splat=force_all)
                    if returncode != 0:
                        splat_failures.append(config)

        if len(splat_failures) > 0:
            print("Splat Failures:")
            for failure in splat_failures:
                print("", failure.output_name, sep="\t")

        if len(eval_failures) > 0:
            print("Eval Failures:")
            for failure in eval_failures:
                print("", failure.output_name, sep="\t")

    def get_output_paths(self, dataset_name: str):
        configs = self.get_configs(dataset_name)
        return [Path(config.output_name) for config in configs]

    def get_input_paths(self, dataset_name: str):
        configs = self.get_configs(dataset_name)
        return [Path(config.input_name) for config in configs]

    def get_render_output_paths(self, dataset_name: str):
        configs = self.get_render_configs(dataset_name)
        return [Path(config.output_name) for config in configs]

    def plot(self, dataset_name: str, create_pcp: bool = False, include_title: bool = False):
        if self.plot_args is None:
            return
        print(self.progress_stats(dataset_name))
        plot_graph(
            name=datasets[dataset_name].scene_name,
            prefix=self.name,
            x_axis=self.plot_args.x_axis,
            split_param=self.plot_args.split_param,
            filter=self.plot_args.filter,
            # TODO This may cause incorrect data to be added to the plot but should be safe because of th i{n} part
            folders=list(
                zip(
                    [path.name for path in self.get_input_paths(dataset_name)],
                    [path.name for path in self.get_output_paths(dataset_name)],
                )
            ),
            render_folders=(
                list(path.name for path in self.get_render_output_paths(dataset_name))
                if self.render_filter_override != {}
                else None
            ),
            create_pcp=create_pcp,
            val_steps=self.val_steps,
            title=self.description,
            metric_keys=self.plot_args.metric_keys,
            dataset_name=datasets[dataset_name].name,
            experiment_name=f"{self.group:02d}_{self.name}",
            config_dict=self.config_dict,
            single_legend=self.plot_args.single_legend,
            create_table=True,
            print_title=include_title,
            split_choice=self.plot_args.split_choice,
        )

    def progress_stats(self, dataset_name: str, print_progress_bars: bool = False) -> str:
        configs = self.get_configs(dataset_name)

        reconstructed = 0
        evaluated = 0
        splatted = 0

        force_reconstruct = 0
        force_eval = 0
        force_splat = 0

        for config in configs:
            if config.is_reconstructed:
                reconstructed += 1
                if config.force_reconstruct:
                    force_reconstruct += 1
            if config.is_evaluated:
                evaluated += 1
                if config.force_eval:
                    force_eval += 1
            if config.is_splatted:
                splatted += 1
                if config.force_splat:
                    force_splat += 1

        if print_progress_bars:
            with tqdm(total=len(configs), desc="Reconstructed", unit="config") as pbar_recon:
                for config in configs:
                    if config.is_reconstructed and not config.force_reconstruct:
                        pbar_recon.update(1)
            with tqdm(total=len(configs), desc="Evaluated", unit="config") as pbar_eval:
                for config in configs:
                    if config.is_evaluated and not config.force_eval:
                        pbar_eval.update(1)
            with tqdm(total=len(configs), desc="Splatted", unit="config") as pbar_splat:
                for config in configs:
                    if config.is_splatted and not config.force_splat:
                        pbar_splat.update(1)

        return (
            f"Reconstructed: {reconstructed} / {len(configs)} | Evaluated: {evaluated} / {len(configs)} | Splatted: {splatted} / {len(configs)}"
            f"\nForce Reconstruct: {force_reconstruct} | Force Eval: {force_eval} | Force Splat: {force_splat}"
            f"\nFinal Reconstruct: {reconstructed - force_reconstruct} / {len(configs)} "
            f"| Final Evaluated: {evaluated - force_eval} / {len(configs)} | Final Splatted: {splatted - force_splat} / {len(configs)}"
        )


experiments = [
    Experiment(
        "num_images",
        1,
        "COLMAP versus VGGT over various number of images",
        {
            "seed": [42, 43, 44],
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["voxels"],
            "gt_eval": [True],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="", single_legend=True),
        render_filter_override={
            "num_images": [20, 30, 40, 100],
            "seed": [42],
        },
    ),
    Experiment(
        "num_images_pose_opt",
        1,
        "COLMAP versus VGGT over various number of images with pose optimization",
        {
            "seed": [42, 43, 44],
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["voxels"],
            "pose_opt": [True, False],
            "gt_eval": [True],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="pose_opt"),
        render_filter_override={
            "num_images": [20, 30, 40, 100],
            "seed": [42],
            "pose_opt": [True],
        },
    ),
    Experiment(
        "num_images_fixed_points",
        1,
        "COLMAP versus VGGT over various number of images",
        {
            "seed": [42],  # , 43, 44
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["voxels"],  # , "random"
            "gt_eval": [True],
            # "pose_opt": [True, False],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
            "num_points_value": 100000,
        },
        PlotConfig(x_axis="num_images", split_param=""),
    ),
    Experiment(
        "num_images_30000",
        1,
        "COLMAP versus VGGT over various number of images for different validation steps",
        {
            "seed": [42],  # , 43, 44
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["voxels"],
            "gt_eval": [True],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
            "num_steps": [30000],
        },
        PlotConfig(x_axis="num_images", split_param="val_step", metric_keys=["psnr", "lpips", "ssim"]),
        val_steps=[7000, 15000, 30000],
    ),
    Experiment(
        "pose_opt",
        4,
        "COLMAP versus VGGT with pose optimization",
        {
            "seed": [42, 43, 44],
            "num_images": [100],
            "pose_opt": [True, False],
            "gt_eval": [True],
            "choice": ["vggt", "colmap"],
            "num_steps": [30000],
        },
        PlotConfig(
            x_axis="", split_param="choice,pose_opt", metric_keys=["eval_rre", "eval_rte", "psnr", "lpips", "ssim"]
        ),
        val_steps=[30000],
    ),
    Experiment(
        "num_points",
        2,
        "COLMAP versus VGGT initialized with different numbers of points",
        {
            "seed": [42],  #
            "num_images": [100],
            "sampling_mode": ["voxels"],
            "num_points_per_image": [10, 50, 100, 200, 300, 500, 750, 1000, 2500, 5000, 10000],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
            "pose_opt": [False],
            "gt_eval": True,
        },
        PlotConfig(x_axis="num_points", split_param="", metric_keys=["psnr", "lpips", "ssim"]),
        render_filter_override={
            "num_points_per_image": [10, 100, 1000, 10000],
        },
    ),
    Experiment(
        "num_points_pose_opt",
        2,
        "COLMAP versus VGGT initialized with different numbers of points and with pose optimization",
        {
            "seed": [42],  #
            "num_images": [100],
            "sampling_mode": ["voxels"],
            "num_points_per_image": [10, 50, 100, 200, 300, 500, 750, 1000, 2500, 5000, 10000],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
            "pose_opt": [True, False],
            "gt_eval": True,
        },
        PlotConfig(x_axis="num_points", split_param="pose_opt", metric_keys=["psnr", "lpips", "ssim"]),
        render_filter_override={
            "num_points_per_image": [10, 100, 1000, 10000],
            "pose_opt": [True],
        },
    ),
    Experiment(
        "sampling_mode",
        3,
        "A comparison of different VGGT point cloud sampling modes",
        {
            "seed": [42, 43, 44],
            "num_images": [100],
            "sampling_mode": ["voxels", "random", "confidence", "ba"],
            "num_points_per_image": [10, 50, 100, 200, 300, 500, 750, 1000, 2500, 5000, 10000],
            "choice": ["vggt", "colmap"],
            "use_gt_cams": False,
            "gt_eval": True,
        },
        PlotConfig(x_axis="num_points", split_param="sampling_mode", metric_keys=["psnr", "lpips", "ssim", "quality"]),
        render_filter_override={
            "seed": [42],
            "num_points_per_image": [1000],
        },
    ),
    Experiment(
        "sampling_mode_gt_cams",
        3,
        "A comparison of different VGGT point cloud sampling modes with GT cameras",
        {
            "seed": [42, 43, 44],
            "num_images": [100],
            "sampling_mode": ["voxels", "random", "confidence", "ba"],
            "num_points_per_image": [10, 50, 100, 200, 300, 500, 750, 1000, 2500, 5000, 10000],
            "choice": ["vggt", "colmap"],
            "use_gt_cams": [True],
            "gt_eval": True,
        },
        PlotConfig(x_axis="num_points", split_param="sampling_mode", metric_keys=["psnr", "lpips", "ssim", "quality"]),
        render_filter_override={
            "seed": [42],
            "num_points_per_image": [2500],
        },
    ),
    Experiment(
        "test",
        99,
        "Small test for functionality",
        {
            "seed": [42, 43, 44],  #
            "num_images": [100],
            "sampling_mode": ["random", "confidence", "voxels", "ba"],
            "num_points_per_image": [1000],
            "pose_opt": [True, False],
            "gt_eval": True,
            "choice": ["vggt", "colmap"],
            "num_steps": [7000],
        },
        PlotConfig(x_axis="val_step", split_param="sampling_mode,pose_opt", metric_keys=["eval_rre", "eval_rte"]),
        val_steps=[1, 7000],
    ),
    Experiment(
        "combined",
        11,
        "Combining cameras and point clouds",
        {
            "seed": [42],  #
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["random"],  # , "confidence", "voxels", "ba"
            "num_points_per_image": [1000],
            "pose_opt": [False],  # True,
            "gt_eval": True,
            "choice": ["combined", "vggt", "colmap"],
            "pcd_src": ["vggt", "colmap", "both"],
            "align_mode": ["global"],
            "camera_src": ["vggt", "colmap"],
            "num_steps": [15000],
        },
        PlotConfig(
            x_axis="num_images",
            split_param="sampling_mode,camera_src,pcd_src",
            metric_keys=["eval_rre", "eval_rte", "psnr", "lpips", "ssim", "quality"],
        ),
        val_steps=[15000],
        render_filter_override={
            "seed": [42],
            "sampling_mode": ["random"],
            "num_images": [100],
        },
    ),
    Experiment(
        "combined_align",
        11,
        "Combining cameras and point clouds",
        {
            "seed": [42],  #
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["random"],  # , "confidence", "voxels", "ba"
            "num_points_per_image": [1000],
            "pose_opt": [False],  # True,
            "gt_eval": True,
            "choice": ["combined", "vggt", "colmap"],
            "pcd_src": ["both"],
            "align_mode": ["local", "global"],
            "camera_src": ["colmap"],
            "num_steps": [15000],
        },
        PlotConfig(
            x_axis="num_images",
            split_param="sampling_mode,align_mode",
            metric_keys=["eval_rre", "eval_rte", "psnr", "lpips", "ssim", "quality"],
        ),
        val_steps=[15000],
        render_filter_override={
            "seed": [42],
            "sampling_mode": ["random"],
            "num_images": [100],
        },
    ),
    Experiment(
        "combined_align_relaxed",
        11,
        "Combining cameras and point clouds",
        {
            "seed": [42],  #
            "num_images": [20, 30],
            "sampling_mode": ["random"],  # , "confidence", "voxels", "ba"
            "num_points_per_image": [1000],
            "pose_opt": [False],  # True,
            "colmap_mode": ["relaxed"],
            "gt_eval": True,
            "choice": ["combined", "vggt", "colmap"],
            "pcd_src": ["both"],
            "align_mode": ["local", "global"],
            "camera_src": ["colmap"],
            "num_steps": [15000],
        },
        PlotConfig(
            x_axis="num_images",
            split_param="sampling_mode,align_mode",
            metric_keys=["eval_rre", "eval_rte", "psnr", "lpips", "ssim", "quality"],
        ),
        val_steps=[15000],
        render_filter_override={
            "seed": [42],
            "sampling_mode": ["random"],
            "num_images": [100],
        },
    ),
    Experiment(
        "pose_opt_validation",
        4,
        "Pose optimization validation",
        {
            "seed": [42],  # , 43, 44
            "num_images": [100],
            "sampling_mode": ["random"],  # , "confidence", "voxels", "ba"
            "num_points_per_image": [1000],
            "pose_opt": [True],
            "gt_eval": True,
            "choice": ["vggt"],
            "num_steps": [7000, 15000, 30000],
        },
        PlotConfig(
            x_axis="val_step",
            split_param="num_steps",
            metric_keys=["eval_rre", "eval_rte", "quality"],
        ),
        val_steps=[1, 3_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000],
    ),
    Experiment(
        "training_step_validation",
        99,
        "Training step validation",
        {
            "seed": [42],  # , 43, 44
            "num_images": [100],
            "sampling_mode": ["random"],  # , "confidence", "voxels", "ba"
            "num_points_per_image": [1000],
            "gt_eval": True,
            "choice": ["vggt"],
            "num_steps": [7000, 15000, 30000],
        },
        PlotConfig(x_axis="val_step", split_param="num_steps", metric_keys=["psnr", "lpips", "ssim"]),
        val_steps=[1, 3_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000],
    ),
    Experiment(
        "error_opa",
        5,
        "Opacity initialization using error-based confidence",
        {
            "seed": [42],  # , 43, 44
            "num_images": [100],
            "sampling_mode": ["voxels"],
            "choice": ["vggt"],
            "error_opa": [True, False],
            "nomcmc": [True, False],
        },
        PlotConfig(x_axis="", split_param="error_opa,splatting_strategy"),
    ),
    Experiment(
        "splatting_strategy",
        9,
        "Splatting strategies with different number of images",
        {
            "seed": [42],  # , 43, 44
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["voxels"],
            "choice": ["vggt"],
            "nomcmc": [True, False],
        },
        PlotConfig(x_axis="num_images", split_param="splatting_strategy"),
    ),
    Experiment(
        "splatting_strategy_points",
        9,
        "Splatting strategies with different number of points",
        {
            "seed": [42],  # , 43, 44
            "num_images": [100],
            "num_points_per_image": [10, 50, 100, 200, 300, 500, 750, 1000, 2500, 5000, 10000],
            "sampling_mode": ["voxels"],
            "choice": ["vggt"],
            "nomcmc": [True, False],
        },
        PlotConfig(x_axis="num_points", split_param="splatting_strategy"),
    ),
    Experiment(
        "splatting_strategy_pose_opt",
        9,
        "Splatting strategies with pose optimization for VGGT with 100 images",
        {
            "seed": [42],  # , 43, 44
            "num_images": [100],
            "num_points_per_image": [5000],
            "sampling_mode": ["voxels"],
            "choice": ["vggt"],
            "nomcmc": [True, False],
            "pose_opt": [True, False],
        },
        PlotConfig(x_axis="", split_param="splatting_strategy,pose_opt"),
        val_steps=[15000],
    ),
    Experiment(
        "colmap_mode",
        8,
        "Default versus Relaxed COLMAP arguments over various number of images",
        {
            "seed": [42],  # , 43, 44
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "gt_eval": [True],
            "colmap_mode": ["default", "relaxed"],
            "image_mode": ["farthestpose"],
            "choice": ["colmap"],
            "num_steps": [15000],
        },
        PlotConfig(x_axis="num_images", split_param="colmap_mode"),
        val_steps=[15000],
    ),
    Experiment(
        "gt",
        6,
        "Small test for gt evaluation",
        {
            "choice": ["gt", "vggt", "colmap"],
        },
        PlotConfig(x_axis="", split_param="num_images"),
    ),
    # Experiment(
    #     "depth",
    #     "COLMAP versus VGGT with depth loss",
    #     {
    #         "seed": [42],  # , 43, 44
    #         "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #         "sampling_mode": ["voxels"],
    #         "depth_loss": [True, False],
    #         "depth_conf": [True, False],
    #         "pose_opt": [True],
    #         "choice": ["vggt", "colmap"],
    #         # "num_points_per_image": [10000, 1100],
    #         "num_points_per_image": [10000],
    #         # "depth_lambda": [0.01, 0.1, 1, 10],
    #     },
    #     PlotConfig(x_axis="num_images", split_param="choice,depth_loss,depth_lambda,depth_conf,sampling_mode"),
    # ),
    Experiment(
        "depth_lambda",
        5,
        "COLMAP versus VGGT with depth loss",
        {
            "seed": [42],  # , 43, 44
            "num_images": [20, 30, 40, 100],
            "sampling_mode": ["random"],
            "depth_loss": [True],
            "pose_opt": [True],
            "choice": ["vggt", "colmap"],
            "depth_lambda": [0.0, 0.01, 0.1, 1, 10],
            "depth_conf": [True, False],
        },
        PlotConfig(x_axis="depth_lambda", split_param="depth_conf,num_images", metric_keys=["psnr", "lpips", "ssim"]),
        render_filter_override={
            "seed": [42],
            "num_images": [30],
            "depth_lambda": [0.0, 0.01, 1],
        },
    ),
    Experiment(
        "camera_type",
        10,
        "A comparison of different camera types with bundle adjustment",
        {
            "seed": [42],
            "num_images": [80],
            "sampling_mode": ["ba"],
            "all_opt": [False],
            "camera_type": ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="", split_param="camera_type"),
    ),
    Experiment(
        "camera_type_ext",
        10,
        "Test for camera mode (extended)",
        {
            "seed": [42, 43, 44],
            "num_images": [100],
            "sampling_mode": ["ba", "voxels"],
            "all_opt": [True, False],
            "camera_type": ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
            "num_points_per_image": [1100, 2200],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="", split_param="camera_type,pose_opt,eval_opt,sampling_mode,num_images"),
    ),
    Experiment(
        "dataset_type",
        99,
        "Test for dataset types",
        {
            "seed": [42, 43, 44],
            "num_images": [50, 100],
            "sampling_mode": ["voxels"],
            "all_opt": [False],
            "num_points_per_image": [1100, 2200],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="num_points"),
    ),
    Experiment(
        "copy_mode",
        99,
        "A comparison of different image cropping modes",
        {
            "num_images": [100],
            "seed": [42, 43, 44],
            "sampling_mode": ["voxels"],
            "all_opt": [False],
            "copy_mode": [None, "crop", "square"],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="", split_param="copy_mode"),
    ),
    Experiment(
        "gt_cams",
        6,
        "The effect of using ground truth extrinsics, intrinsics and points on VGGT and COLMAP results",
        {
            "seed": [42],  # , 43, 44
            "num_images": [100],
            "gt_eval": True,
            "use_gt_extrinsics": [True, False],
            "use_gt_intrinsics": [True, False],
            "use_gt_points": [True, False],
            # "pose_opt": [True, False],
            "choice": ["vggt", "colmap"],
            "num_steps": [15000],
        },
        PlotConfig(
            x_axis="",
            split_param="use_gt_extrinsics,use_gt_intrinsics,use_gt_points",
            metric_keys=["quality"],
            split_choice=True,
        ),
        val_steps=[15000],
    ),
]

experiment_dict = {exp.name: exp for exp in experiments}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--experiment_names", type=str, required=True, help="Name of the experiment to run", nargs="+")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--skip_splatting", action="store_true", help="Whether to skip splatting")
    parser.add_argument("--do_reconstruct", action="store_true", help="Whether to run reconstruction")
    parser.add_argument("--plot_only", action="store_true", help="Whether to only plot")
    parser.add_argument("--include_gt", action="store_true", help="Whether to include the ground truth splatted data")
    parser.add_argument(
        "--force_all", action="store_true", help="Whether to force all experiments to rerun"
    )  # TODO add these properly
    parser.add_argument("--force_none", action="store_true", help="Ignore force calculation")
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Comma separated list of GPU IDs to use for parallel splatting (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--procs_per_gpu",
        type=int,
        default=1,
        help="Number of parallel processes to run per GPU when using --cuda_visible_devices",
    )
    parser.add_argument(
        "--check_only", action="store_true", help="Only check the status of the experiments without running anything"
    )

    args = parser.parse_args()

    cuda_devices = args.cuda_visible_devices.split(",") if args.cuda_visible_devices else None

    if args.cuda_visible_devices and args.procs_per_gpu > 1 and cuda_devices is not None:
        cuda_devices = [gpu_id for gpu_id in cuda_devices for _ in range(args.procs_per_gpu)]

    for arg_experiment in args.experiment_names:
        for experiment in experiments:
            if experiment.name == arg_experiment or arg_experiment == "all":
                experiment = replace(experiment, include_gt=args.include_gt)

                if args.check_only:
                    print(f"Experiment: {experiment.name}")
                    print(experiment.progress_stats(args.dataset_name, print_progress_bars=True))
                    print()
                    continue

                if not args.plot_only:
                    experiment.run(
                        args.dataset_name,
                        do_reconstruct=args.do_reconstruct,
                        do_splatting=not args.skip_splatting,
                        cuda_devices=cuda_devices,
                    )
                experiment.plot(args.dataset_name)
