from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os
import subprocess
import argparse

from tqdm import tqdm

from vggt.plot_metrics import plot_graph

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


datasets = {
    "lego": Dataset(
        name="lego",
        factor=1,
        directory="../vggt/data/nerf_synthetic/lego",
        gt_train_data_dir="../vggt/data/nerf_synthetic/lego/transforms_train.json",
        gt_eval_data_dir="../vggt/data/nerf_synthetic/lego/transforms_val.json",
        data_folder_name="train",
    ),
    "bonsai": Dataset(name="bonsai", factor=2, directory="../vggt/data/360_v2/bonsai", data_folder_name="images_2"),
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
        data_folder_name="dataset_N100_R5.5_H1.0_C-4.0_0.0_2.0_SIMPLE_PINHOLE",
    ),
}


@dataclass
class Config:
    choice: Literal["vggt", "colmap"] = "vggt"
    num_images: int = 30
    dataset: Dataset = datasets["lego"]
    seed: int = 42
    conf_thres_value: float = 0.0
    num_points_value: int = 35000
    sampling_mode: Literal["voxels", "random", "confidence", "ba"] = "voxels"
    image_mode: Literal["shuffle", "distributed"] = "shuffle"
    gt_eval: bool = False
    pose_opt: bool = False
    eval_opt: bool = False
    all_opt: bool = False
    num_cameras: int | None = None
    depth_loss: bool = False
    depth_conf: bool = False
    camera_type: Literal["SIMPLE_RADIAL", "SIMPLE_PINHOLE"] = "SIMPLE_PINHOLE"

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        instance = cls()
        instance.choice = data["choice"]
        instance.num_images = data.get("num_images", instance.num_images)
        instance.dataset = datasets[data.get("dataset", "lego")]
        instance.seed = data.get("seed", instance.seed)
        instance.conf_thres_value = data.get("conf_thres_value", instance.conf_thres_value)
        instance.num_points_value = data.get("num_points_value", instance.num_points_value)
        instance.sampling_mode = data.get("sampling_mode", instance.sampling_mode)
        instance.image_mode = data.get("image_mode", instance.image_mode)
        instance.gt_eval = data.get("gt_eval", instance.gt_eval)
        instance.pose_opt = data.get("pose_opt", instance.pose_opt)
        instance.eval_opt = data.get("eval_opt", instance.eval_opt)
        instance.pose_opt |= data.get("all_opt", instance.all_opt)
        instance.eval_opt |= data.get("all_opt", instance.all_opt)
        instance.num_cameras = data.get("num_cameras", instance.num_cameras)
        instance.depth_loss = data.get("depth_loss", instance.depth_loss)
        instance.depth_conf = data.get("depth_conf", instance.depth_conf)
        instance.camera_type = data.get("camera_type", instance.camera_type)
        return instance

    def __post_init__(self):
        self.depth_conf = self.depth_conf and self.choice == "vggt" and self.depth_loss

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
            parts.append(self.image_mode)
        elif self.choice == "vggt":
            if self.sampling_mode == "ba":
                parts.append(self.camera_type.lower().replace("simple_", "m"))
                parts.append(self.sampling_mode)
            else:
                parts.extend(
                    [
                        f"c{self.conf_thres_value}",
                        f"p{self.num_points_value}",
                        self.sampling_mode,
                    ]
                )
            parts.append(self.image_mode)

        parts = "_".join(parts)
        return f"{self.choice}_outputs/{parts}"

    @property
    def output_name(self):
        parts = [
            f"i{self.num_cams}",
        ]
        if self.gt_eval:
            parts.append("gteval")
        if self.pose_opt:
            parts.append("poseopt")
        if self.eval_opt:
            parts.append("evalopt")
        if self.depth_loss:
            parts.append("depth")
        if self.depth_conf and self.choice == "vggt" and self.depth_loss:
            parts.append("conf")

        parts = "_".join(parts)
        return f"{self.input_name}_{parts}"

    @property
    def result_dir(self):
        return f"./results/{self.output_name}"

    @property
    def stats_dir(self):
        return f"{self.result_dir}/stats/val_step6999.json"

    @property
    def data_dir(self):
        return f"../vggt/{self.input_name}"

    @property
    def force(self):
        return False
        return self.sampling_mode == "ba" and self.camera_type == "SIMPLE_RADIAL" and self.choice == "vggt"

    def reconstruct(self):
        if os.path.exists(os.path.join(self.data_dir, "stat.json")) and not self.force:
            print(Path(self.data_dir), "has already been constructed.\nUse --force to force reconstruction.")
            return 0

        command = [
            VGGT_PYTHON,
            "-m",
            "reconstruct",
            "--input",
            Path(self.dataset.directory) / Path(self.dataset.data_folder_name),
            "--name",
            f"{self.dataset.name}_{self.dataset.factor}",
            "--choice",
            self.choice,
            "--num_images",
            str(self.num_images),
            "--num_points",
            str(self.num_points_value),
            "--seed",
            str(self.seed),
            "--conf_thres_value",
            str(self.conf_thres_value),
            "--sampling_mode",
            self.sampling_mode,
            "--image_mode",
            self.image_mode,
            "--camera_type",
            self.camera_type,
        ]
        if self.force:
            command.append("--force")

        try:
            output = subprocess.run(command, check=True, cwd="../vggt")
            print(output)
            return output.returncode
        except subprocess.CalledProcessError as e:
            print(e)
            return e.returncode

    def run(self):

        if os.path.exists(self.stats_dir) and not self.force:
            print(f"{self.stats_dir} found. Skipping splatting")
            return 0

        if self.force:
            print("Forcing splatting")
        else:
            print(f"{Path(self.stats_dir)} not found. Running splatting")
        print(f"Using data from: {Path(self.data_dir)}")
        print(f"Result dir: {Path(self.result_dir)}")
        command = [
            GSPLAT_PYTHON,
            "examples/simple_trainer.py",
            "mcmc",
            "--data_dir",
            self.data_dir,
            "--data_factor",
            "1",
            "--result-dir",
            self.result_dir,
            "--disable_viewer",
            "--max_train_cameras",
            str(self.num_cams),
            "--max_steps",
            "7000",
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
        if self.eval_opt:
            command.append("--eval_opt")

        if self.depth_loss:
            command.append("--depth_loss")

        if self.depth_conf:
            command.append("--depth_conf")

        try:
            output = subprocess.run(command, check=True)
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


@dataclass
class Experiment:
    name: str
    description: str
    config_dict: dict
    plot_args: PlotConfig | None = None

    def get_configs(self, dataset_name: str) -> list[Config]:
        self.config_dict["dataset"] = dataset_name
        config_dicts = generate_configs(self.config_dict)
        configs = [Config.from_dict(config_dict) for config_dict in config_dicts]
        config_set = set()
        unique_configs: list[Config] = []
        for config in configs:
            if (config.result_dir, config.stats_dir, config.data_dir) not in config_set:
                unique_configs.append(config)
            config_set.add((config.result_dir, config.stats_dir, config.data_dir))
        return unique_configs

    def run(self, dataset_name: str, do_reconstruct: bool = True):
        configs = self.get_configs(dataset_name)
        failures: list[Config] = []
        for config in tqdm(configs):
            if do_reconstruct:
                reconstruction_returncode = config.reconstruct()
                if reconstruction_returncode != 0:
                    failures.append(config)
                    continue

            returncode = config.run()
            if returncode != 0:
                failures.append(config)

        if len(failures) > 0:
            print("Failures:")
            for failure in failures:
                print("", failure.output_name, sep="\t")

    def plot(self, dataset_name: str):
        if self.plot_args is None:
            return
        configs = self.get_configs(dataset_name)
        plot_graph(
            name=datasets[dataset_name].scene_name,
            prefix=self.name,
            x_axis=self.plot_args.x_axis,
            split_param=self.plot_args.split_param,
            filter=self.plot_args.filter,
            folders=[config.output_name.split("/")[-1] for config in configs],
        )


experiments = [
    Experiment(
        "num_images",
        "Test the behaviour of splatting over various number of images",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [10, 20, 30, 40, 50, 100],
            "seed": [42, 43, 44],
        },
        PlotConfig(x_axis="num_images", split_param="choice"),
    ),
    Experiment(
        "num_images_pose_opt",
        "Test the behaviour of splatting over various number of images",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [10, 20, 30, 40, 50, 100],
            "seed": [42, 43, 44],
            "pose_opt": [True],
            "eval_opt": [True],
        },
        PlotConfig(x_axis="num_images", split_param="choice"),
    ),
    Experiment(
        "pose_opt",
        "Test the behaviour of different combinations of pose optimization",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [30],
            "seed": [42, 43, 44],
            "pose_opt": [True, False],
            "eval_opt": [True, False],
            "gt_eval": [True, False],
        },
        PlotConfig(x_axis="num_images", split_param="choice,pose_opt,eval_opt,gt_eval"),
    ),
    Experiment(
        "num_points",
        "Test the behaviour of different numbers of points and sampling modes",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [30],
            "seed": [42, 43, 44],
            "sampling_mode": ["voxels", "random", "confidence", "ba"],
            "num_points_value": [1000, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 500000, 1000000],
        },
        PlotConfig(x_axis="num_points_value", split_param="choice"),
    ),
    Experiment(
        "test",
        "Small test for functionality",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [30],
            "seed": [42],
        },
        PlotConfig(x_axis="num_images", split_param="choice"),
    ),
    Experiment(
        "depth",
        "Test for depth loss and depth conf",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [30],
            "seed": [42],
            "sampling_mode": ["voxels", "ba"],
            "depth_loss": [True, False],
            "depth_conf": [True, False],
            "pose_opt": [True],
            "eval_opt": [True],
        },
        PlotConfig(x_axis="num_images", split_param="choice,depth_loss,depth_conf,sampling_mode"),
    ),
    Experiment(
        "camera_type",
        "Test for camera mode",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [100],
            "seed": [42, 43, 44],
            "sampling_mode": ["ba"],
            "all_opt": [False],
            "camera_type": ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
        },
        PlotConfig(x_axis="num_images", split_param="camera_type"),
    ),
    Experiment(
        "camera_type_ext",
        "Test for camera mode (extended)",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [100],
            "seed": [42, 43, 44],
            "sampling_mode": ["ba", "voxels"],
            "all_opt": [True, False],
            "camera_type": ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
            "num_points_value": [35000, 75000],
        },
        PlotConfig(x_axis="num_images", split_param="camera_type,pose_opt,eval_opt,sampling_mode"),
    ),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment to run")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--do_reconstruct", action="store_true", help="Whether to run reconstruction")
    args = parser.parse_args()

    for experiment in experiments:
        if experiment.name == args.experiment_name or args.experiment_name == "all":
            experiment.run(args.dataset_name, do_reconstruct=args.do_reconstruct)
            experiment.plot(args.dataset_name)
