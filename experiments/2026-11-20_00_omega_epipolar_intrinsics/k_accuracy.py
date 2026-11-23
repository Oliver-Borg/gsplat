"""Focal accuracy of the training K in experiments 120-123, backfilled from files (no GPU).

GT: the dataset's COLMAP camera (single PINHOLE) scaled to the training image size. Used K per run type:
  per-frame Omega K  -> the PINHOLE cameras of the Omega reconstruction (data dir `sparse/`), 255 frames
  `meank` runs       -> the mean of the per-frame fx / fy (shared K)
  `intropt` runs     -> `train_fx` / `train_fy` of each val_step json (shared K with the optimised focal)
  COLMAP reference   -> the GT K itself (error 0 by construction)
Errors are relative, in %: (used - gt) / gt. Runs in the vggt env (pycolmap):  python k_accuracy.py
"""
import csv, glob, json, os, re
import numpy as np
import pycolmap

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
vggt = os.path.abspath(os.path.join(root, "../vggt"))
data_dir = os.path.join(vggt, "vggt_outputs/bonsai_2_n255_s42_omega_c0.0_p100000_random_farthestpose")
P = os.path.join(root, "results/vggt_outputs/bonsai_2_n255_s42_omega_c0.0_p100000_random_farthestpose_i255_gteval_")
S = "_tpo_evalall_steps30000"

gt = list(pycolmap.Reconstruction(os.path.join(vggt, "data/360_v2/bonsai/sparse/0")).cameras.values())
assert len(gt) == 1 and gt[0].model.name == "PINHOLE"
gt = gt[0]
om = list(pycolmap.Reconstruction(os.path.join(data_dir, "sparse")).cameras.values())
W, H = om[0].width, om[0].height
assert all((c.width, c.height) == (W, H) and c.model.name == "PINHOLE" for c in om)
gt_fx, gt_fy = gt.params[0] * W / gt.width, gt.params[1] * H / gt.height
fx = np.array([c.params[0] for c in om])
fy = np.array([c.params[1] for c in om])
print(f"GT K at {W}x{H}: fx {gt_fx:.3f} fy {gt_fy:.3f}; Omega frames: {len(om)}, fx mean {fx.mean():.3f}, fy mean {fy.mean():.3f}")


def pct(u, g):
    return (np.asarray(u) - g) / g * 100.0


def summ(ex, ey):
    return dict(fx_mean=ex.mean(), fx_abs_mean=np.abs(ex).mean(), fx_abs_max=np.abs(ex).max(),
                fy_mean=ey.mean(), fy_abs_mean=np.abs(ey).mean(), fy_abs_max=np.abs(ey).max())


rows = []
# per-frame Omega K (initial / fixed-Omega runs)
rows.append(("Omega per-frame K (fixed Omega cameras)", "per-frame", summ(pct(fx, gt_fx), pct(fy, gt_fy))))
# shared mean K
sm = np.array([fx.mean()]), np.array([fy.mean()])
rows.append(("shared mean K (all `meank` runs)", "shared", summ(pct(sm[0], gt_fx), pct(sm[1], gt_fy))))
rows.append(("COLMAP reference (GT K)", "GT", summ(np.zeros(1), np.zeros(1))))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "k_accuracy.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "kind", "fx_err_mean_pct", "fx_abs_err_mean_pct", "fx_abs_err_max_pct", "fy_err_mean_pct", "fy_abs_err_mean_pct", "fy_abs_err_max_pct"])
    for name, kind, d in rows:
        w.writerow([name, kind] + [f"{d[k]:.4f}" for k in ("fx_mean", "fx_abs_mean", "fx_abs_max", "fy_mean", "fy_abs_mean", "fy_abs_max")])
    # focal optimisation runs: the shared focal at every evaluation
    for g in sorted(glob.glob(P + "*intropt" + S)):
        g_name = os.path.basename(g).replace(os.path.basename(P), "")
        for jf in sorted(glob.glob(g + "/stats/val_step*.json"), key=lambda x: int(re.search(r"val_step(\d+)", x).group(1))):
            j = json.load(open(jf))
            step = int(re.search(r"val_step(\d+)", jf).group(1)) + 1
            ex, ey = pct([j["train_fx"]], gt_fx)[0], pct([j["train_fy"]], gt_fy)[0]
            w.writerow([f"{g_name} step {step}", "intropt", f"{ex:.4f}", f"{abs(ex):.4f}", f"{abs(ex):.4f}", f"{ey:.4f}", f"{abs(ey):.4f}", f"{abs(ey):.4f}"])
for name, kind, d in rows:
    print(f"{name}: fx err mean {d['fx_mean']:+.3f}% (|.| mean {d['fx_abs_mean']:.3f}%, max {d['fx_abs_max']:.3f}%), "
          f"fy err mean {d['fy_mean']:+.3f}% (|.| mean {d['fy_abs_mean']:.3f}%, max {d['fy_abs_max']:.3f}%)")
