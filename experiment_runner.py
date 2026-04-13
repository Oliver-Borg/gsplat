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
    choice: Literal["vggt", "colmap", "gt"] = "vggt"
    num_images: int = 30
    dataset: Dataset = datasets["lego"]
    seed: int = 42
    conf_thres_value: float = 0.0
    num_points_per_image: float = 1100
    num_points_value: int | None = None
    sampling_mode: Literal["voxels", "random", "confidence", "ba"] = "voxels"
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
    depth_conf: bool = False
    camera_type: Literal["SIMPLE_RADIAL", "SIMPLE_PINHOLE"] = "SIMPLE_PINHOLE"
    num_steps: Literal[7000, 15000, 30000] = 15000
    colmap_mode: Literal["default", "relaxed"] = "default"

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
        instance.use_gt_points = data.get("use_gt_points", instance.use_gt_points)
        instance.pose_opt = data.get("pose_opt", instance.pose_opt)
        instance.eval_opt = data.get("eval_opt", instance.eval_opt)
        instance.pose_opt |= data.get("all_opt", instance.all_opt)
        instance.eval_opt |= data.get("all_opt", instance.all_opt)
        instance.num_cameras = data.get("num_cameras", instance.num_cameras)
        instance.depth_loss = data.get("depth_loss", instance.depth_loss)
        instance.depth_conf = data.get("depth_conf", instance.depth_conf)
        instance.camera_type = data.get("camera_type", instance.camera_type)
        instance.num_steps = data.get("num_steps", instance.num_steps)
        instance.colmap_mode = data.get("colmap_mode", instance.colmap_mode)
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

        self.depth_conf = self.depth_conf and self.choice == "vggt" and self.depth_loss

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
    def splatting_val_path(self):
        return f"{self.result_dir}/stats/val_step{self.num_steps - 1}.json"

    @property
    def data_dir(self):
        if self.choice == "gt":
            return self.dataset.directory

        return f"../vggt/{self.input_name}"

    @property
    def force_splat(self):
        if self.choice == "colmap" and self.depth_loss:
            return True
        if self.force_reconstruct:
            return True
        if self.gt_eval and self.is_splatted:
            if len(list(filter(lambda x: "6999" in x, os.listdir(self.renders_folder)))) < 30:
                return True
        return False
        return self.sampling_mode == "ba" and self.camera_type == "SIMPLE_RADIAL" and self.choice == "vggt"

    @property
    def force_reconstruct(self):
        if self.choice == "gt":
            return False

        if self.is_splatted and self.choice == "vggt":
            with open(self.splatting_val_path, "r") as f:
                stats = _parse_gsplat_json(json.load(f), self.splatting_val_path)
                psnr = stats.get("psnr", 0.0)
            if psnr is None or (psnr < 15 and self.num_images > 20):
                return True
            return False
        else:
            return self.sampling_mode != "ba" and self.choice == "vggt"

    @property
    def reconstruction_stat_path(self):
        return os.path.join(self.data_dir, "stat.json")

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

        if self.force_reconstruct:
            args["force"] = True

        if self.copy_mode is not None:
            args["copy_mode"] = self.copy_mode

        return args

    def reconstruct(self):
        if self.is_reconstructed and not self.force_reconstruct:
            print(Path(self.data_dir), "has already been constructed.\nUse --force to force reconstruction.")
            return 0

        command = [
            VGGT_PYTHON,
            "-m",
            "reconstruct",
            "single",
        ]
        for key, value in self.reconstruct_args.items():
            if key == "force" and value is True:
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
    def is_evaluated(self):
        return os.path.exists(self.eval_path)

    @property
    def force_eval(self):
        if not self.is_evaluated:
            return True

        threshold_date = datetime.datetime(2026, 4, 4, 11, 20)  # This is when I fixed the eval script
        threshold_timestamp = threshold_date.timestamp()
        file_timestamp = Path(self.eval_path).stat().st_mtime

        if file_timestamp < threshold_timestamp:
            return True

        if (
            self.is_reconstructed
            and not self.choice == "gt"
            and file_timestamp < Path(self.reconstruction_stat_path).stat().st_mtime
        ):
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
    def run(self, gpu: str | None = None):
        if self.is_splatted and not self.force_splat:
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

        if self.depth_conf:
            command.append("--depth_conf")

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


@dataclass
class Experiment:
    name: str
    description: str
    config_dict: dict
    plot_args: PlotConfig | None = None
    include_gt: bool = False
    val_steps: list[int] = field(default_factory=lambda: [15000])

    def get_configs(self, dataset_name: str) -> list[Config]:
        self.config_dict["dataset"] = dataset_name
        config_dicts = generate_configs(self.config_dict)
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

    def bulk_reconstruct(self, configs: list[Config]):
        reconstruct_args = []
        for config in configs:
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
                        reconstruction_returncode = config.reconstruct()
                        if reconstruction_returncode != 0:
                            return config, True, False

                    eval_returncode = config.eval()
                    if eval_returncode != 0:
                        eval_fail = True

                    if do_splatting:
                        returncode = config.run(gpu=gpu)
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
                    reconstruction_returncode = config.reconstruct()
                    if reconstruction_returncode != 0:
                        splat_failures.append(config)
                        continue

                eval_returncode = config.eval()
                if eval_returncode != 0:
                    eval_failures.append(config)

                if do_splatting:
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
            val_steps=self.val_steps,
            title=self.description,
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
        "COLMAP versus VGGT over various number of images",
        {
            "seed": [42, 43, 44],
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["voxels"],
            "gt_eval": [True],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="sampling_mode"),
    ),
    Experiment(
        "num_images_pose_opt",
        "COLMAP versus VGGT over various number of images with pose optimization",
        {
            "seed": [42],  # , 43, 44
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["voxels"],
            "pose_opt": [True, False],
            "gt_eval": [True],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="sampling_mode,pose_opt"),
    ),
    Experiment(
        "num_images_30000",
        "COLMAP versus VGGT over various number of images for different step counts",
        {
            "seed": [42, 43, 44],
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["voxels"],
            "gt_eval": [True],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
            "num_steps": [30000],
        },
        PlotConfig(x_axis="num_images", split_param="sampling_mode,val_step"),
        val_steps=[7000, 15000, 30000],
    ),
    Experiment(
        "pose_opt",
        "COLMAP versus VGGT with pose optimization",
        {
            "seed": [42, 43, 44],
            "num_images": [100],
            "pose_opt": [True, False],
            "gt_eval": [True],
            "choice": ["vggt", "colmap"],
            "num_steps": [15000],
        },
        PlotConfig(x_axis="", split_param="choice,pose_opt,eval_opt,gt_eval"),
        val_steps=[15000],
    ),
    Experiment(
        "num_points",
        "COLMAP versus VGGT initialized with different numbers of points",
        {
            "seed": [42],  #
            "num_images": [100],
            "sampling_mode": ["voxels", "ba"],
            "num_points_per_image": [10, 50, 100, 200, 300, 500, 750, 1000, 2500, 5000, 10000, 25000, 50000],
            "image_mode": ["farthestpose"],
            "choice": ["vggt", "colmap"],
            "pose_opt": [True, False],
            "gt_eval": True,
        },
        PlotConfig(x_axis="num_points", split_param="sampling_mode,pose_opt"),
    ),
    Experiment(
        "sampling_mode",
        "A comparison of different VGGT point cloud sampling modes",
        {
            "seed": [42, 43, 44],
            "num_images": [100],
            "sampling_mode": ["voxels", "random", "confidence", "ba"],
            "num_points_per_image": [1100, 2200, 5000],
            "choice": ["vggt", "colmap"],
            "gt_eval": True,
        },
        PlotConfig(x_axis="", split_param="sampling_mode,num_points"),
    ),
    Experiment(
        "test",
        "Small test for functionality",
        {
            "seed": [42],  # , 43, 44
            "num_images": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sampling_mode": ["voxels"],
            "gt_eval": [True],
            "colmap_mode": ["default", "relaxed"],
            "image_mode": ["farthestpose"],
            "choice": ["colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="colmap_mode,image_mode"),
    ),
    Experiment(
        "colmap_mode",
        "Default versus Relaxed COLMAP arguments over various number of images",
        {
            "seed": [42, 43, 44],
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
        "Small test for gt evaluation",
        {
            "choice": ["gt", "vggt"],
        },
        PlotConfig(x_axis="num_images", split_param=""),
    ),
    Experiment(
        "depth",
        "COLMAP versus VGGT with depth loss",
        {
            "seed": [42, 43, 44],
            "num_images": [100],
            "sampling_mode": ["voxels", "ba"],
            "depth_loss": [True, False],
            "depth_conf": [True, False],
            "pose_opt": [True],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="", split_param="choice,depth_loss,depth_conf,sampling_mode"),
    ),
    Experiment(
        "camera_type",
        "A comparison of different camera types with bundle adjustment",
        {
            "seed": [42],
            "num_images": [80],
            "sampling_mode": ["ba"],
            "all_opt": [False],
            "camera_type": ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
            "choice": ["vggt", "colmap"],
        },
        PlotConfig(x_axis="num_images", split_param="camera_type"),
    ),
    Experiment(
        "camera_type_ext",
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
        "The effect of using ground truth extrinsics, intrinsics and points on VGGT and COLMAP results",
        {
            "seed": [42, 43, 44],
            "num_images": [100],
            "gt_eval": True,
            "use_gt_extrinsics": [True, False],
            "use_gt_intrinsics": [True, False],
            "use_gt_points": [True, False],
            # "pose_opt": [True, False],
            "choice": ["vggt", "colmap", "gt"],
            "num_steps": [15000],
        },
        PlotConfig(x_axis="", split_param="use_gt_extrinsics,use_gt_intrinsics,use_gt_points"),
        val_steps=[15000],
    ),
]

experiment_dict = {exp.name: exp for exp in experiments}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment to run")
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
    args = parser.parse_args()

    cuda_devices = args.cuda_visible_devices.split(",") if args.cuda_visible_devices else None

    for experiment in experiments:
        if experiment.name == args.experiment_name or args.experiment_name == "all":
            experiment = replace(experiment, include_gt=args.include_gt)
            if not args.plot_only:
                experiment.run(
                    args.dataset_name,
                    do_reconstruct=args.do_reconstruct,
                    do_splatting=not args.skip_splatting,
                    cuda_devices=cuda_devices,
                )
            experiment.plot(args.dataset_name)
