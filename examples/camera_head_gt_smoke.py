"""Smoke test: fine-tune the VGGT-Omega camera head directly against ground-truth cameras (no rendering).

The GT cameras are aligned once to the head's initial prediction (Umeyama sim(3) on the camera centres, rotations
rotated by the same R) and are then fixed targets. Loss: Huber (smooth L1) on the 9-D pose encoding
(T, quaternion, fov_h, fov_w) of the Omega world frame. Every step the head runs on all frames with gradient.
Metrics (mean rre/rte, GT aligned to the current predicted cameras like `eval_rre`/`eval_rte`) are logged each step.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_finetune import CameraFinetuner  # noqa: E402
from datasets.colmap import Parser  # noqa: E402
from evaluation import calculate_metrics, stochastic_umeyama_alignment  # noqa: E402
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402


def gt_targets(ft: CameraFinetuner, gt_parser: Parser, names):
    """GT cameras aligned to the initial prediction, as pose encodings [S,9] of the Omega world frame."""
    gt_c2w = gt_parser.get_camera_positions(names)  # [S,4,4], GT parser frame
    ref = ft.ref_c2w.cpu().numpy()  # initial prediction, trainer frame
    s, R, t = stochastic_umeyama_alignment(gt_c2w[:, :3, 3], ref[:, :3, 3])
    al = np.tile(np.eye(4), (len(names), 1, 1))
    al[:, :3, 3] = s * gt_c2w[:, :3, 3] @ R.T + t
    al[:, :3, :3] = R @ gt_c2w[:, :3, :3]
    al = torch.from_numpy(al).float().to(ft.device)

    # trainer frame -> Omega world, w2c (inverse of CameraFinetuner._cameras)
    Rc = ft.R_M.t() @ al[:, :3, :3]
    pos = (al[:, :3, 3] - ft.t_M) @ ft.R_M / ft.scale
    Rw = Rc.transpose(1, 2)
    Tw = -(Rw @ pos[..., None])[..., 0]
    extr = torch.cat([Rw, Tw[..., None]], dim=-1)

    # GT focal at the network resolution (per-axis)
    K = []
    for n in names:
        i = gt_parser.image_names.index(n)
        cid = gt_parser.camera_ids[i]
        w, h = gt_parser.imsize_dict[cid]
        k = gt_parser.Ks_dict[cid].copy()
        k[0] *= ft.W / w
        k[1] *= ft.H / h
        K.append(k)
    K = torch.from_numpy(np.stack(K)).float().to(ft.device)

    enc = extri_intri_to_pose_encoding(extr[None], K[None], (ft.H, ft.W))[0]
    flip = ((enc[:, 3:7] * ft.ref_pose_enc[:, 3:7]).sum(-1) < 0).float() * -2 + 1  # keep the ref quaternion sign
    enc[:, 3:7] = enc[:, 3:7] * flip[:, None]
    return enc, gt_c2w


@torch.no_grad()
def metrics(ft: CameraFinetuner, gt_c2w, names):
    pred = ft.current_cameras()[0].cpu().numpy()
    m = calculate_metrics(
        {n: gt_c2w[i] for i, n in enumerate(names)}, {n: pred[ft.frame_of[n]] for i, n in enumerate(names)}
    )
    return m["mean_rre_deg"], m["mean_rte"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", required=True)
    ap.add_argument("--data_dir", required=True, help="trainer-frame COLMAP dir of the Omega reconstruction")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    cache = torch.load(a.cache_file, map_location="cpu", weights_only=False)
    names = list(cache["base_image_path_list"])
    train_parser = Parser(data_dir=a.data_dir, factor=1, normalize=True, test_every=8)
    gt_parser = Parser(data_dir=a.gt_dir, factor=2, normalize=False, test_every=8)
    cfg = SimpleNamespace(
        camft_cache_file=a.cache_file,
        camft_checkpoint=cache["omega_checkpoint"],
        camft_lora=a.lora,
        camft_lora_rank=a.lora_rank,
        camft_lora_alpha=float(a.lora_rank),
        camft_delay=0,
        camft_lr=a.lr,
    )
    ft = CameraFinetuner(cfg, train_parser, names, a.device)
    target, gt_c2w = gt_targets(ft, gt_parser, names)
    opt = ft.make_optimizer()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for step in range(a.steps + 1):
            if step > 0:
                loss = F.smooth_l1_loss(ft._pose_enc(), target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                enc = ft._pose_enc()
                parts = (enc - target).abs().mean(0)
                loss_now = F.smooth_l1_loss(enc, target).item()
            rre, rte = metrics(ft, gt_c2w, names)
            rec = {
                "step": step,
                "loss": loss_now,
                "rre": rre,
                "rte": rte,
                "l1_T": parts[:3].mean().item(),
                "l1_quat": parts[3:7].mean().item(),
                "l1_fov": parts[7:].mean().item(),
            }
            f.write(json.dumps(rec) + "\n")
            if step % 20 == 0:
                print(rec, flush=True)


if __name__ == "__main__":
    main()
