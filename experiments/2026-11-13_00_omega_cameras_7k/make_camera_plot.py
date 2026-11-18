"""Camera plot of the five camera sets (vggt/plot_cameras.py `plot_extrinsics`, RTE-coloured, aligned to GT).
Uses each data dir's `aligned_cameras.json` (predicted cameras aligned to the GT world) and `gt_cameras.json`
(the GT arm has neither file, its cameras are the GT cameras).
Run in the vggt env:  env -u PYTHONPATH ~/.conda/envs/vggt/bin/python <this file>
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/work/git"))
from vggt.cam_utils import load_image_sizes_from_json, load_intrinsics_from_json, load_poses_from_json  # noqa: E402
from vggt.plot_cameras import CameraSeries, plot_extrinsics  # noqa: E402

V = Path(os.path.expanduser("~/work/git/vggt"))
N = sys.argv[1] if len(sys.argv) > 1 else "40"  # training images: 40 or 255
SEL = "minfarthestpose" if N == "40" else "farthestpose"
base = f"bonsai_2_n{N}_s42_omega"
dirs = {
    "GT_cams_": V / f"combined_outputs/{base}_combined_gtcams_vggtpcd_amglobal_default_c0.0_p100000_random_{SEL}_sharedcam",
    "Head_full_": V / f"vggt_outputs/{base}_c0.0_p100000_random_{SEL}_ftcams-full",
    "Head_LoRA_": V / f"vggt_outputs/{base}_c0.0_p100000_random_{SEL}_ftcams-lora",
    "Omega_": V / f"vggt_outputs/{base}_c0.0_p100000_random_{SEL}",
    "COLMAP_": V / f"combined_outputs/{base}_combined_colmapcams_vggtpcd_amglobal_default_c0.0_p100000_random_{SEL}_sharedcam",
}
gt_json = V / "vggt_outputs" / f"{base}_c0.0_p100000_random_{SEL}" / "gt_cameras.json"
gt_poses, gt_K, gt_sz = (f(gt_json) for f in (load_poses_from_json, load_intrinsics_from_json, load_image_sizes_from_json))
names = sorted(load_poses_from_json(dirs['Omega_'] / 'aligned_cameras.json'))  # the 40 training images
series = []
for label, d in dirs.items():
    j = d / "aligned_cameras.json"
    if not j.exists():  # GT arm: the cameras are the GT cameras, which have no aligned_cameras.json
        j = gt_json
    series.append(CameraSeries(label, (1.0, 0.0, 0.0, 0.5), load_poses_from_json(j), load_intrinsics_from_json(j),
                               load_image_sizes_from_json(j), gt_poses, gt_K, gt_sz, names))
tmp = Path(tempfile.mkdtemp())
plot_extrinsics([("default", series)], tmp, use_error_colors=True, max_cols=5)
out = Path(__file__).resolve().parent / "figures" / ("cameras.jpg" if N == "40" else "cameras_n255.jpg")
shutil.copy(tmp / "camera_alignment.jpg", out)
print("saved", out)
