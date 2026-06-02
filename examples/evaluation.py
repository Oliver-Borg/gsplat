import argparse
import datetime
import glob
import json
import os
from pathlib import Path
import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as Rot
from typing import Dict, Tuple, List, Optional, Any, TypedDict, Union
import struct

try:
    from .datasets.colmap import Parser
    from .datasets.nerf_synth import SimpleParser, load_json_data
except ImportError:  # TODO Figure out a better way to do this
    from datasets.colmap import Parser
    from datasets.nerf_synth import SimpleParser, load_json_data


class EvalMetrics(TypedDict, total=False):
    mean_rre_deg: float
    mean_rte: float
    auc_1: float
    auc_5: float
    auc_10: float
    auc_20: float
    auc_30: float
    num_aligned: int
    real_num_points: int | None
    alignment_scale: float
    error: str
    all_rre: list[float]
    all_rte: list[float]
    mean_depth_l1: float | None
    mean_depth_absrel: float | None
    mean_depth_rmse: float | None
    all_depth_l1: list[float] | None
    all_depth_absrel: list[float] | None
    all_depth_rmse: list[float] | None


class EvalReport(TypedDict):
    timestamp: str
    pred_path: str
    gt_path: str
    metrics: EvalMetrics
    profiling: Dict[str, Any]


def umeyama_alignment(
    from_points: np.ndarray, to_points: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Computes optimal similarity transform: p = s * R * q + t.
    https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    q = from_points
    p = to_points

    if np.allclose(q, p):
        return 1.0, np.eye(3), np.zeros(3)

    # Translation
    n, m = q.shape
    q_mean = q.mean(axis=0)
    p_mean = p.mean(axis=0)
    q_centered = q - q_mean
    p_centered = p - p_mean
    # Computation of the covariance matrix

    H = np.dot(p_centered.T, q_centered) / n
    # First, calculate the SVD of the covariance matrix H,
    U, Sigma, Vt = np.linalg.svd(H)

    # Next, record if the orthogonal matrices contain a reflection,
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1

    # Finally, calculate our optimal rotation matrix R as
    rotation = np.dot(np.dot(U, S), Vt)

    if with_scale:
        var_x = np.var(q_centered, axis=0).sum()
        scale = np.trace(np.dot(np.diag(Sigma), S)) / var_x
    else:
        scale = 1.0

    translation = p_mean - scale * np.dot(rotation, q_mean)
    return float(scale), rotation, translation


def stochastic_umeyama_alignment(
    from_points: np.ndarray, to_points: np.ndarray, with_scale: bool = True, num_trials: int = 10, num_samples: int = 3
) -> tuple[float, np.ndarray, np.ndarray]:
    best_score = float("inf")
    best_transform = (1.0, np.eye(3), np.zeros(3))
    base_seed = 42
    np.random.seed(base_seed)
    for _ in range(num_trials):
        indices = np.random.choice(
            from_points.shape[0],
            size=max(min(from_points.shape[0] // num_samples, from_points.shape[0]), 3),
            replace=False,
        )
        s, R, t = umeyama_alignment(from_points[indices], to_points[indices], with_scale=with_scale)

        transformed = s * (R @ from_points.T).T + t
        score = np.mean(np.linalg.norm(transformed - to_points, axis=1))

        if score < best_score:
            best_score = score
            best_transform = (s, R, t)

    return best_transform


def load_parser_data(
    path: str,
) -> Tuple[Optional[Union[Parser, SimpleParser]], Optional[Dict], Optional[Dict], Optional[Dict], Optional[str]]:
    """Safe wrapper to initialize Parser and extract poses."""
    try:
        if path.endswith(".json"):
            parser = load_json_data(path)
        else:
            if path.endswith("cameras.bin") or path.endswith("cameras.txt"):
                path = os.path.dirname(path)
            # Parser expects data_dir. Normalize=False to keep raw scale for alignment.
            parser = Parser(data_dir=path, normalize=False)

        # Create dict {image_name: c2w} for metrics calculation
        poses = {name: c2w for name, c2w in zip(parser.image_names, parser.camtoworlds)}
        intrinsics = {
            name: parser.Ks_dict[parser.camera_ids[parser.image_names.index(name)]] for name in parser.image_names
        }
        imsizes = {
            name: parser.imsize_dict[parser.camera_ids[parser.image_names.index(name)]] for name in parser.image_names
        }
        return parser, poses, intrinsics, imsizes, None
    except Exception as e:
        return None, None, None, None, str(e)


def calculate_metrics(
    pred_poses: Dict[str, np.ndarray],
    gt_poses: Dict[str, np.ndarray],
    pred_parser: Parser | SimpleParser | None = None,
    gt_parser: Parser | SimpleParser | None = None,
    output_path: str | None = None,
) -> EvalMetrics:
    common_names = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))
    use_direction_vectors = False
    if len(common_names) < 3:
        use_direction_vectors = True

    p_centers = np.array([pred_poses[n][:3, 3] for n in common_names])
    g_centers = np.array([gt_poses[n][:3, 3] for n in common_names])

    if use_direction_vectors:
        # Add an extra set of points to p_centers and g_centers by adding the direction vector of each camera
        p_directions = np.array([pred_poses[n][:3, 2] for n in common_names])
        g_directions = np.array([gt_poses[n][:3, 2] for n in common_names])
        p_centers = np.concatenate([p_centers, p_centers + p_directions])
        g_centers = np.concatenate([g_centers, g_centers + g_directions])

    s, R_align, t_align = stochastic_umeyama_alignment(p_centers, g_centers)

    rre_list: List[float] = []
    rte_list: List[float] = []

    for name in common_names:
        p_c2w = pred_poses[name].copy()
        p_c2w[:3, 3] = s * R_align @ p_c2w[:3, 3] + t_align
        p_c2w[:3, :3] = R_align @ p_c2w[:3, :3]

        g_c2w = gt_poses[name]

        # RRE: Geodesic distance
        rel_rot = np.dot(p_c2w[:3, :3].T, g_c2w[:3, :3])
        cos_theta = (np.trace(rel_rot) - 1.0) / 2.0
        rre = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # RTE: Euclidean distance
        rte = np.linalg.norm(p_c2w[:3, 3] - g_c2w[:3, 3])

        rre_list.append(float(rre))
        rte_list.append(float(rte))

    depth_l1_list = []
    depth_absrel_list = []
    depth_rmse_list = []

    if pred_parser is not None and gt_parser is not None:
        if hasattr(pred_parser, "depths") and hasattr(gt_parser, "depths"):
            if output_path is not None:
                print("Writing depths to", (Path(output_path) / "depth_eval").absolute())
            for name in common_names:
                if name in gt_parser.depths and name in pred_parser.depths:
                    gt_depth = gt_parser.depths[name]
                    # Scale the predicted depth map by the alignment scale
                    pred_depth = pred_parser.depths[name] * s

                    # Mask out invalid (NaN or negative) depths
                    valid_mask = ~np.isnan(gt_depth) & ~np.isnan(pred_depth) & (gt_depth > 0) & (pred_depth > 0)
                    gt_depth = np.nan_to_num(gt_depth)
                    pred_depth = np.nan_to_num(pred_depth)

                    if np.any(valid_mask):
                        d_gt = gt_depth[valid_mask]
                        d_pr = pred_depth[valid_mask]

                        depth_l1_list.append(float(np.mean(np.abs(d_gt - d_pr))))
                        depth_absrel_list.append(float(np.mean(np.abs(d_gt - d_pr) / d_gt)))
                        depth_rmse_list.append(float(np.sqrt(np.mean((d_gt - d_pr) ** 2))))

                        # Normalise the depth maps and save them to output_path/depth_eval
                        if output_path is None:
                            continue
                        depth_output_path = os.path.join(output_path, "depth_eval")
                        os.makedirs(depth_output_path, exist_ok=True)
                        side_by_side = np.concatenate(
                            [gt_depth, pred_depth, np.abs(gt_depth - pred_depth)], axis=1
                        ) / np.max(gt_depth)
                        cv2.imwrite(
                            os.path.join(depth_output_path, f"depth_{name}"),
                            (side_by_side * 255).clip(0, 255).astype(np.uint8),
                        )

    return {
        "mean_rre_deg": round(float(np.mean(rre_list)), 4),
        "mean_rte": round(float(np.mean(rte_list)), 6),
        "auc_1": round(float(np.mean(np.array(rre_list) < 1)), 3),
        "auc_5": round(float(np.mean(np.array(rre_list) < 5)), 3),
        "auc_10": round(float(np.mean(np.array(rre_list) < 10)), 3),
        "auc_20": round(float(np.mean(np.array(rre_list) < 20)), 3),
        "auc_30": round(float(np.mean(np.array(rre_list) < 30)), 3),
        "num_aligned": len(common_names),
        "real_num_points": None if pred_parser is None else pred_parser.points.shape[0],
        "alignment_scale": round(s, 6),
        "all_rre": rre_list,
        "all_rte": rte_list,
        "mean_depth_l1": round(float(np.mean(depth_l1_list)), 4) if depth_l1_list else None,
        "mean_depth_absrel": round(float(np.mean(depth_absrel_list)), 4) if depth_absrel_list else None,
        "mean_depth_rmse": round(float(np.mean(depth_rmse_list)), 4) if depth_rmse_list else None,
        "all_depth_l1": depth_l1_list if depth_l1_list else None,
        "all_depth_absrel": depth_absrel_list if depth_absrel_list else None,
        "all_depth_rmse": depth_rmse_list if depth_rmse_list else None,
    }


def main(pred: str, gt: str, force: bool = False) -> None:
    failed_eval_json = Path(pred) / "failed_eval.json"
    try:
        if os.path.exists(failed_eval_json):
            os.remove(failed_eval_json)
        _main(pred, gt, force=force)
    except Exception as e:
        error_file = failed_eval_json
        with open(error_file, "w") as f:
            json.dump(str(e), f)
        raise e


def _main(pred: str, gt: str, force: bool = False) -> None:
    out_file = os.path.join(pred, "eval_results.json")
    if os.path.exists(out_file) and not force:
        print(f"Evaluation for {pred} already exists at {out_file}. Skipping.")
        return

    gt_parser, gt_poses, _, _, gt_err = load_parser_data(gt)
    pred_parser, pred_poses, _, _, pred_err = load_parser_data(pred)

    if gt_err:
        raise ValueError(gt_err)

    if pred_err:
        raise ValueError(pred_err)

    assert gt_poses is not None and pred_poses is not None

    if not gt_poses or not pred_poses:
        raise ValueError(f"Error: Missing or empty reconstruction in pred ({len(pred_poses)}) or gt ({len(gt_poses)})")

    metrics = calculate_metrics(pred_poses, gt_poses, pred_parser, gt_parser, output_path=pred)

    stat_json_path = os.path.join(pred, "stat.json")
    profiling = {}
    if os.path.exists(stat_json_path):
        with open(stat_json_path, "r") as f:
            profiling = json.load(f).get("profiling", {})

    report: EvalReport = {
        "timestamp": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
        "pred_path": pred,
        "gt_path": gt,
        "metrics": metrics,
        "profiling": profiling,
    }
    with open(out_file, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Evaluation for {pred} Complete. Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SfM reconstruction vs GT.")
    parser.add_argument("--pred-glob", required=True, type=str)
    parser.add_argument("--gt", required=True, type=str)
    parser.add_argument("--force", type=bool, default=False, help="Force re-evaluation")
    args = parser.parse_args()

    for pred in sorted(glob.glob(args.pred_glob)):
        main(pred, args.gt, force=args.force)
