"""Camera plot (vggt/plot_cameras.py `plot_extrinsics`, RTE-coloured): initial prediction vs fine-tuned cameras vs GT.

Run in the vggt env:
  env -u PYTHONPATH ~/.conda/envs/vggt/bin/python gsplat/experiments/<this folder>/make_camera_plot.py [step]
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.expanduser("~/work/git"))
from vggt.plot_cameras import CameraSeries, plot_extrinsics  # noqa: E402

step = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
d = Path(__file__).resolve().parent
arms = {"full head lr 1e-5": "full_lr1e-5_cameras.npz", "LoRA r16 lr 1e-4": "lora16_lr1e-4_cameras.npz", "pose MLP lr 1e-4": "mlp_lr1e-4_cameras.npz"}
series = []
cols = [(1.0, 0.0, 0.0, 0.5)] * 4
z0 = np.load(d / next(iter(arms.values())), allow_pickle=True)
names = [str(n) for n in z0["names"]]
idx = z0["frame_idx"]
H, W = 1039, 1559


def umeyama(src, dst):
    """Similarity (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / X.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def mk(label, c2w, K, gt, gtK):
    # GT (raw COLMAP frame) aligned to this series' predicted cameras, as in the eval metrics
    p = c2w[idx]
    s, R, t = umeyama(gt[:, :3, 3], p[:, :3, 3])
    gt = gt.copy()
    gt[:, :3, 3] = s * gt[:, :3, 3] @ R.T + t
    gt[:, :3, :3] = R @ gt[:, :3, :3]
    return CameraSeries(
        label, cols[0],
        {n: c2w[i] for n, i in zip(names, idx)}, {n: K[i] for n, i in zip(names, idx)}, {n: (W, H) for n in names},
        {n: gt[j] for j, n in enumerate(names)}, {n: gtK[i] for n, i in zip(names, idx)}, {n: (W, H) for n in names},
        names,
    )


gt = z0["gt_c2w"]
gtK = z0["init_K"]  # GT frustum drawn with the initial predicted intrinsics (shape only)
series.append(mk("initial_", z0["init_c2w"], z0["init_K"], gt, gtK))
for label, f in arms.items():
    z = np.load(d / f, allow_pickle=True)
    series.append(mk(label.replace(" ", "_") + "_", z[f"c2w_step{step}"], z[f"K_step{step}"], gt, gtK))

tmp = Path(tempfile.mkdtemp())
plot_extrinsics([("default", series)], tmp, use_error_colors=True, max_cols=4)
(d / "figures").mkdir(exist_ok=True)
shutil.copy(tmp / "camera_alignment.jpg", d / "figures" / f"cameras_step{step}.jpg")
print("saved", d / "figures" / f"cameras_step{step}.jpg")
