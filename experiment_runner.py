from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os
import subprocess
import argparse


@dataclass
class Dataset:
    name: str
    factor: int
    gt_train_data_dir: str | None = None
    gt_eval_data_dir: str | None = None


datasets = {
    "lego": Dataset(
        name="lego",
        factor=1,
        gt_train_data_dir="../vggt/data/nerf_synthetic/lego/transforms_train.json",
        gt_eval_data_dir="../vggt/data/nerf_synthetic/lego/transforms_val.json",
    ),
    "bonsai": Dataset(
        name="bonsai",
        factor=1,
    ),
}


@dataclass
class Config:
    choice: Literal["vggt", "colmap"] = "vggt"
    num_images: int = 30
    dataset: Dataset = datasets["lego"]
    seed: int = 42
    conf_thres_value: float = 0.0
    num_points_value: int = 30000
    sampling_mode: Literal["voxels", "random", "confidence", "ba"] = "voxels"
    image_mode: Literal["shuffle", "distributed"] = "shuffle"
    gt_eval: bool = False
    pose_opt: bool = False
    eval_opt: bool = False
    num_cameras: int | None = None

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
        instance.num_cameras = data.get("num_cameras", instance.num_cameras)
        return instance

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

        parts = "_".join(parts)
        return f"{self.input_name}_{parts}"

    def run(self):
        result_dir = f"./results/{self.output_name}"
        stats_dir = f"{result_dir}/stats/val_step6999.json"
        data_dir = f"../vggt/{self.input_name}"

        if os.path.exists(stats_dir):
            print(f"{stats_dir} found. Skipping splatting")
            return 0

        print(f"{Path(stats_dir)} not found. Running splatting")
        print(f"Using data from: {Path(data_dir)}")
        print(f"Result dir: {Path(result_dir)}")
        command = [
            "python",
            "examples/simple_trainer.py",
            "mcmc",
            "--data_dir",
            data_dir,
            "--data_factor",
            f"{self.dataset.factor}",
            "--result-dir",
            result_dir,
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
class Experiment:
    name: str
    description: str
    config_dict: dict

    def get_configs(self, dataset_name: str):
        self.config_dict["dataset"] = dataset_name
        config_dicts = generate_configs(self.config_dict)
        configs = [Config.from_dict(config_dict) for config_dict in config_dicts]
        return configs

    def run(self, dataset_name: str):
        configs = self.get_configs(dataset_name)
        failures: list[Config] = []
        for config in configs:
            returncode = config.run()
            if returncode != 0:
                failures.append(config)
        
        if len(failures) > 0:
            print("Failures:")
            for failure in failures:
                print("", failure.output_name, sep="\t")


experiments = [
    Experiment(
        "num_images",
        "Test the behaviour of splatting over various number of images",
        {
            "choice": ["vggt", "colmap"],
            "num_images": [10, 20, 30, 40, 50, 100],
            "seed": [42, 43, 44],
        },
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
    ),
]

# TODO Add automatic plotting
# TODO Add progress bar
# TODO Add automatic vggt running

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment to run")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    args = parser.parse_args()

    for experiment in experiments:
        if args.experiment_name is None or experiment.name == args.experiment_name:
            experiment.run(args.dataset_name)
