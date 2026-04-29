import argparse
import datetime
import glob
import json
import os
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
    pred_poses: Dict[str, np.ndarray], gt_poses: Dict[str, np.ndarray], pred_parser: Parser | SimpleParser | None = None
) -> EvalMetrics:
    common_names = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))
    if len(common_names) < 3:
        return {"error": f"Only {len(common_names)} images matched. Need >= 3 for Umeyama."}

    p_centers = np.array([pred_poses[n][:3, 3] for n in common_names])
    g_centers = np.array([gt_poses[n][:3, 3] for n in common_names])

    s, R_align, t_align = umeyama_alignment(p_centers, g_centers)

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
    }


def main(pred: str, gt: str, force: bool = False) -> None:
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
        print(f"Error: Missing or empty reconstruction in pred ({len(pred_poses)}) or gt ({len(gt_poses)})")
        return

    metrics = calculate_metrics(pred_poses, gt_poses, pred_parser)

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
