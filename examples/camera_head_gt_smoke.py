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
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_finetune import CameraFinetuner, gt_pose_targets  # noqa: E402
from datasets.colmap import Parser  # noqa: E402
from evaluation import calculate_metrics  # noqa: E402


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
    ap.add_argument("--snapshots", type=int, nargs="*", default=[], help="steps at which the cameras are saved")
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
    target, gt_c2w = gt_pose_targets(ft, gt_parser, names)
    opt = ft.make_optimizer()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)
    print(f"device={a.device} torch_threads={torch.get_num_threads()}", flush=True)
    c0 = ft.current_cameras()
    initial = (c0[0].cpu().numpy(), c0[1].cpu().numpy())
    snaps = {}
    t_start = time.perf_counter()
    with open(a.out, "w") as f:
        for step in range(a.steps + 1):
            t_train = 0.0
            if step > 0:
                sync()
                t0 = time.perf_counter()
                loss = F.smooth_l1_loss(ft._pose_enc(), target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sync()
                t_train = time.perf_counter() - t0
            t0 = time.perf_counter()
            with torch.no_grad():
                enc = ft._pose_enc()
                parts = (enc - target).abs().mean(0)
                loss_now = F.smooth_l1_loss(enc, target).item()
            rre, rte = metrics(ft, gt_c2w, names)
            t_eval = time.perf_counter() - t0
            rec = {
                "step": step,
                "loss": loss_now,
                "rre": rre,
                "rte": rte,
                "l1_T": parts[:3].mean().item(),
                "l1_quat": parts[3:7].mean().item(),
                "l1_fov": parts[7:].mean().item(),
                "t_train_s": t_train,  # forward + backward + optimiser step
                "t_eval_s": t_eval,  # re-predict cameras + loss + rre/rte (not part of training)
            }
            if step in a.snapshots:
                cs = ft.current_cameras()
                snaps[f"c2w_step{step}"], snaps[f"K_step{step}"] = cs[0].cpu().numpy(), cs[1].cpu().numpy()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if step % 50 == 0:
                print(rec, flush=True)
    print(f"total wall time {time.perf_counter() - t_start:.1f}s", flush=True)

    c1 = ft.current_cameras()
    np.savez(
        os.path.splitext(a.out)[0] + "_cameras.npz",
        names=np.array(names),
        frame_idx=np.array([ft.frame_of[n] for n in names]),
        init_c2w=initial[0],
        init_K=initial[1],
        final_c2w=c1[0].cpu().numpy(),
        final_K=c1[1].cpu().numpy(),
        gt_c2w=gt_c2w,
        **snaps,
    )

if __name__ == "__main__":
    main()
