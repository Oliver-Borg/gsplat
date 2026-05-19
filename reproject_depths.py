from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import tqdm


from experiment_runner import datasets
from examples.datasets.nerf_synth import SimpleParser
from vggt.cam_utils import reproject_depth


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=datasets.keys(), required=True)
    args = parser.parse_args()

    dataset = datasets[args.dataset]
    assert (
        dataset.gt_train_data_dir is not None and dataset.gt_eval_data_dir is not None
    ), "Dataset must have gt_train_data_dir and gt_eval_data_dir"
    train_dir = dataset.gt_train_data_dir
    eval_dir = dataset.gt_eval_data_dir
    train_parser = SimpleParser(train_dir)
    eval_parser = SimpleParser(eval_dir, max_points=100_000_000, create_pointcloud=True)

    output_dir = Path(dataset.directory) / dataset.data_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Eval parser has actual points/depths. Train parser does not.
    # We want to reproject the eval depths to the train cameras.
    for cam_id in tqdm.tqdm(train_parser.camera_ids):
        c2w = train_parser.camtoworlds[cam_id]
        K = train_parser.Ks_dict[cam_id]
        w, h = train_parser.imsize_dict[cam_id]
        train_depth = reproject_depth(c2w, K, eval_parser.points, eval_parser.points_rgb, w, h)
        image_name = train_parser.image_names[cam_id]
        # data/nerf_synthetic/lego/test/r_0.png data/nerf_synthetic/lego/test/r_0_depth_0001.png
        depth_output_name = output_dir / f"{image_name[:-4]}_depth_0001.png"
        bg_mask = np.isnan(train_depth)
        train_depth[bg_mask] = 0.0
        train_depth = train_depth / 8.0
        train_depth = (255.0 - train_depth * 255.0).clip(0, 255).astype(np.uint8)
        train_depth[bg_mask] = 0
        cv2.imwrite(str(depth_output_name), train_depth)


if __name__ == "__main__":
    main()
