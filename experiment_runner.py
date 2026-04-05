from dataclasses import dataclass, replace
import datetime
import json
from pathlib import Path
from typing import Literal
import os
import subprocess
import argparse

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


datasets = {
    "lego": Dataset(
        name="lego",
        factor=1,
        directory="../vggt/data/nerf_synthetic/lego",
        gt_train_data_dir="../vggt/data/nerf_synthetic/lego/transforms_train.json",
        gt_eval_data_dir="../vggt/data/nerf_synthetic/lego/transforms_val.json",
        data_folder_name="train",
    ),
    "bonsai": Dataset(
        name="bonsai",
        factor=2,
        directory="../vggt/data/360_v2/bonsai",
        gt_eval_data_dir="../vggt/data/360_v2/bonsai",
        gt_train_data_dir="../vggt/data/360_v2/bonsai",
        data_folder_name="images_2",
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
    choice: Literal["vggt", "colmap", "gt"] = "vggt"
    num_images: int = 30
    dataset: Dataset = datasets["lego"]
    seed: int = 42
    conf_thres_value: float = 0.0
    num_points_value: int = 35000
    sampling_mode: Literal["voxels", "random", "confidence", "ba"] = "voxels"
    image_mode: Literal["shuffle", "distributed"] = "shuffle"
    copy_mode: Literal[None, "crop", "square", "tiles"] = None
    gt_eval: bool = True
    pose_opt: bool = False
    eval_opt: bool = False
    all_opt: bool = False
    num_cameras: int | None = None
    depth_loss: bool = False
    depth_conf: bool = False
    camera_type: Literal["SIMPLE_RADIAL", "SIMPLE_PINHOLE"] = "SIMPLE_PINHOLE"
    num_steps: Literal[7000, 30000] = 7000

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
        instance.copy_mode = data.get("copy_mode", instance.copy_mode)
        instance.gt_eval = data.get("gt_eval", instance.gt_eval)
        instance.pose_opt = data.get("pose_opt", instance.pose_opt)
        instance.eval_opt = data.get("eval_opt", instance.eval_opt)
        instance.pose_opt |= data.get("all_opt", instance.all_opt)
        instance.eval_opt |= data.get("all_opt", instance.all_opt)
        instance.num_cameras = data.get("num_cameras", instance.num_cameras)
        instance.depth_loss = data.get("depth_loss", instance.depth_loss)
        instance.depth_conf = data.get("depth_conf", instance.depth_conf)
        instance.camera_type = data.get("camera_type", instance.camera_type)
        instance.num_steps = data.get("num_steps", instance.num_steps)
        if instance.eval_opt and instance.gt_eval:  # TODO Figure out a good way to have both enabled
            instance.eval_opt = False
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
                        f"p{self.num_points_value}",
                        self.sampling_mode,
                    ]
                )
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
    def renders_folder(self):
        return f"{self.result_dir}/renders"

    @property
    def stats_dir(self):
        return f"{self.result_dir}/stats/val_step{self.num_steps - 1}.json"

    @property
    def data_dir(self):
        if self.choice == "gt":
            return self.dataset.directory

        return f"../vggt/{self.input_name}"

    @property
    def force_splat(self):
        if self.force_reconstruct: return True
        if self.gt_eval and self.is_splatted:
            if len(list(filter(lambda x : "6999" in x, os.listdir(self.renders_folder)))) < 30:
                return True
        return False
        return self.sampling_mode == "ba" and self.camera_type == "SIMPLE_RADIAL" and self.choice == "vggt"

    @property
    def force_reconstruct(self):
        if self.is_splatted and self.choice == "vggt":
            with open(self.stats_dir, "r") as f:
                stats = _parse_gsplat_json(json.load(f), self.stats_dir)
                psnr = stats.get("psnr", 0.0)
            if psnr is None or (psnr < 15 and self.num_images > 20):
                return True
            return False
        else:
            return self.sampling_mode != "ba" and self.choice == "vggt"

    @property
    def is_reconstructed(self):
        return os.path.exists(os.path.join(self.data_dir, "stat.json")) or self.choice == "gt"

    def reconstruct(self):
        if self.is_reconstructed and not self.force_reconstruct:
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
        if self.force_reconstruct:
            command.append("--force")
        if self.copy_mode is not None:
            command.extend(["--copy_mode", self.copy_mode])

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
    def is_evaluated(self):
        return os.path.exists(self.eval_path)

    @property
    def force_eval(self):
        if not self.is_evaluated:
            return True

        threshold_date = datetime.datetime(2026, 4, 4, 11, 20)  # This is when I fixed the eval script
        threshold_timestamp = threshold_date.timestamp()
        file_timestamp = Path(self.eval_path).stat().st_mtime
        return file_timestamp < threshold_timestamp


    def eval(self):
        if self.is_evaluated:
            return 0
        try:
            examples.evaluation.main(self.data_dir, self.dataset.directory, force=self.force_eval)
            return 0
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 1

    @property
    def is_splatted(self):
        return os.path.exists(self.stats_dir)

    @profile
    def run(self):
        if self.is_splatted and not self.force_splat:
            print(f"{self.stats_dir} found. Skipping splatting")
            return 0

        if self.force_splat:
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
    include_gt: bool = False

    def get_configs(self, dataset_name: str) -> list[Config]:
        self.config_dict["dataset"] = dataset_name
        config_dicts = generate_configs(self.config_dict)
        configs = [Config.from_dict(config_dict) for config_dict in config_dicts]
        gt_configs = []
        if self.include_gt:
            for config in configs:
                gt_config = replace(config, choice="gt")
                num_images = len(
                    os.listdir(Path(gt_config.dataset.directory) / gt_config.dataset.data_folder_name)
                )  # This will not work for lego yet in splatting
                gt_config.num_images = num_images
                gt_configs.append(gt_config)

        configs += gt_configs
        config_set = set()
        unique_configs: list[Config] = []
        for config in configs:
            if (config.result_dir, config.stats_dir, config.data_dir) not in config_set:
                unique_configs.append(config)
            config_set.add((config.result_dir, config.stats_dir, config.data_dir))
        return unique_configs

    def run(self, dataset_name: str, do_reconstruct: bool = True):
        configs = self.get_configs(dataset_name)
        splat_failures: list[Config] = []
        eval_failures: list[Config] = []
        for config in tqdm(configs):
            if do_reconstruct:
                reconstruction_returncode = config.reconstruct()
                if reconstruction_returncode != 0:
                    splat_failures.append(config)
                    continue

            eval_returncode = config.eval()
            if eval_returncode != 0:
                eval_failures.append(config)

            returncode = config.run()
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

    def plot(self, dataset_name: str, create_pcp: bool = True):
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
            create_pcp=create_pcp,
        )

    def progress_stats(self, dataset_name: str) -> str:
        configs = self.get_configs(dataset_name)

        reconstructed = 0
        splatted = 0
        for config in configs:
            if config.is_reconstructed:
                reconstructed += 1
            if config.is_splatted:
                splatted += 1

        return f"Reconstructed: {reconstructed} / {len(configs)} | Splatted: {splatted} / {len(configs)}"


experiments = [
    Experiment(
        "num_images",
        "Test the behaviour of splatting over various number of images",
        {
            "seed": [42, 43, 44],
            "num_images": [10, 20, 30, 40, 50, 100],
            "sampling_mode": ["voxels", "ba"],
            "gt_eval": [True],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="gt_eval,sampling_mode"),
    ),
    Experiment(
        "pose_opt",
        "Test the behaviour of different combinations of pose optimization",
        {
            "seed": [42, 43, 44],
            "num_images": [30],
            "pose_opt": [True, False],
            "eval_opt": [True, False],
            "gt_eval": [True, False],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="choice,pose_opt,eval_opt,gt_eval"),
    ),
    Experiment(
        "num_points",
        "Test the behaviour of different numbers of points",
        {
            "seed": [42, 43, 44],
            "num_images": [30],
            "sampling_mode": ["voxels", "random"],
            "num_points_value": [1000, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 500000, 1000000],
            "choice": ["vggt", "colmap"],
            "gt_eval": True,
        },
        PlotConfig(x_axis="num_points", split_param="sampling_mode"),
    ),
    Experiment(
        "sampling_mode",
        "Test the behaviour of different sampling modes",
        {
            "seed": [42, 43, 44],
            "num_images": [30],
            "sampling_mode": ["voxels", "random", "confidence", "ba"],
            "num_points_value": [100000],
            "choice": ["vggt", "colmap"],
            "gt_eval": True,
        },
        PlotConfig(x_axis="num_points", split_param="sampling_mode"),
    ),
    Experiment(
        "test",
        "Small test for functionality",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [30],
            "seed": [42],
            "num_steps": [30000],
        },
        PlotConfig(x_axis="val_step", split_param=""),
    ),
    Experiment(
        "depth",
        "Test for depth loss and depth conf",
        {
            "seed": [42],
            "num_images": [30],
            "sampling_mode": ["voxels", "ba"],
            "depth_loss": [True, False],
            "depth_conf": [True, False],
            "pose_opt": [True],
            "eval_opt": [True],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="choice,depth_loss,depth_conf,sampling_mode"),
    ),
    # Experiment(
    #     "camera_type",
    #     "Test for camera mode",
    #     {
    #         "seed": [42, 43, 44],
    #         "num_images": [50, 100],
    #         "sampling_mode": ["voxels"],
    #         "all_opt": [False],
    #         "camera_type": ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
    #         "num_points": [35000],
    #         "choice": ["vggt", "colmap"],
    #     },
    #     PlotConfig(x_axis="num_images", split_param="camera_type"),
    # ),
    # Experiment(
    #     "camera_type_ext",
    #     "Test for camera mode (extended)",
    #     {
    #         "seed": [42, 43, 44],
    #         "num_images": [100],
    #         "sampling_mode": ["ba", "voxels"],
    #         "all_opt": [True, False],
    #         "camera_type": ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
    #         "num_points_value": [35000, 75000],
    #         "choice": ["vggt", "colmap"],
    #     },
    #     PlotConfig(x_axis="num_images", split_param="camera_type,pose_opt,eval_opt,sampling_mode"),
    # ),
    # Experiment(
    #     "dataset_type",
    #     "Test for dataset types",
    #     {
    #         "seed": [42, 43, 44],
    #         "num_images": [50, 100],
    #         "sampling_mode": ["voxels"],
    #         "all_opt": [False],
    #         "num_points": [35000, 75000],
    #         "choice": ["vggt", "colmap"],
    #     },
    #     PlotConfig(x_axis="num_images", split_param="num_points"),
    # ),
    # Experiment(
    #     "copy_mode",
    #     "Test for copy modes",
    #     {
    #         "num_images": [50, 100],
    #         "seed": [42, 43, 44],
    #         "sampling_mode": ["voxels"],
    #         "all_opt": [False],
    #         "copy_mode": [None, "crop", "square"],
    #         "choice": ["vggt", "colmap"],
    #     },
    #     PlotConfig(x_axis="num_images", split_param="copy_mode"),
    # ),
    Experiment(
        "num_images_pose_opt",
        "Test the behaviour of splatting over various number of images",
        {
            "seed": [42, 43, 44],
            "num_images": [10, 20, 30, 40, 50, 100],
            "pose_opt": [True, False],
            "eval_opt": [False],
            "gt_eval": [True],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="choice,pose_opt"),
    ),
]

experiment_dict = {exp.name: exp for exp in experiments}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment to run")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--do_reconstruct", action="store_true", help="Whether to run reconstruction")
    parser.add_argument("--plot_only", action="store_true", help="Whether to only plot")
    parser.add_argument("--include_gt", action="store_true", help="Whether to include the ground truth splatted data")
    args = parser.parse_args()

    for experiment in experiments:
        if experiment.name == args.experiment_name or args.experiment_name == "all":
            experiment = replace(experiment, include_gt=args.include_gt)
            if not args.plot_only:
                experiment.run(args.dataset_name, do_reconstruct=args.do_reconstruct)
            experiment.plot(args.dataset_name)
