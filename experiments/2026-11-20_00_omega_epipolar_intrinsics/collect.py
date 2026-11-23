import csv, glob, json, os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
P = "vggt_outputs/*_n255_*omega_c0.0*farthestpose_i255_gteval_"
S = "_tpo_evalall_steps30000"
ARMS = {  # key -> (label, glob)
    "mlp_epi_k": ("3RGS MLP, epipolar+photometric, shared mean K", P + "cft-0.0001-mlp-d0-epi2x3000k10_meank" + S),
    "mlp_photo_k": ("3RGS MLP, photometric, shared mean K", P + "cft-0.0001-mlp-d0_meank" + S),
    "def_epi_k": ("default module, epipolar+photometric, shared mean K", P + "cft-0.0001-pdef-d0-epi2x3000k10_meank" + S),
    "def_photo_k": ("default module, photometric, shared mean K", P + "cft-0.0001-pdef-d0_meank" + S),
    "mlp_epi_i": ("3RGS MLP, epipolar+photometric, shared K, focal optimised", P + "cft-0.0001-mlp-d0-epi2x3000k10_intropt" + S),
    "mlp_photo_i": ("3RGS MLP, photometric, shared K, focal optimised", P + "cft-0.0001-mlp-d0_intropt" + S),
    "def_epi_i": ("default module, epipolar+photometric, shared K, focal optimised", P + "cft-0.0001-pdef-d0-epi2x3000k10_intropt" + S),
    "def_photo_i": ("default module, photometric, shared K, focal optimised", P + "cft-0.0001-pdef-d0_intropt" + S),
    "fixed_perframe": ("Omega cameras (fixed), per-frame K", P + "tpo_evalall_steps30000"),
    "fixed_meank": ("Omega cameras (fixed), shared mean K", P + "meank" + S),
    "fixed_intropt": ("Omega cameras (fixed), shared K, focal optimised", P + "intropt" + S),
    "colmap": ("COLMAP cameras + point cloud (fixed)", "gt_outputs/bonsai_2_n255_s42_i292_tpo_steps30000"),
}
SECTIONS = {
    "meank": ["mlp_epi_k", "mlp_photo_k", "def_epi_k", "def_photo_k", "fixed_perframe", "fixed_meank", "colmap"],
    "intropt": ["mlp_epi_i", "mlp_photo_i", "def_epi_i", "def_photo_i", "mlp_epi_k", "mlp_photo_k", "def_epi_k", "def_photo_k", "fixed_intropt", "fixed_meank", "fixed_perframe", "colmap"],
}
F = ("psnr", "ssim", "lpips", "num_GS", "eval_rre", "eval_rte", "tpo_psnr", "tpo_ssim", "tpo_lpips", "train_fx", "train_fy")
rows = []
for sec, keys in SECTIONS.items():
    for k in keys:
        arm, g = ARMS[k]
        d = glob.glob(os.path.join(root, "results", g)); assert len(d) == 1, (arm, d); d = d[0]
        drift = {}
        if os.path.exists(d + "/stats/ft_drift.jsonl"):
            for l in open(d + "/stats/ft_drift.jsonl"):
                j = json.loads(l); drift[j["step"]] = j
        for f in sorted(glob.glob(d + "/stats/val_step*.json"), key=lambda f: int(re.search(r"val_step(\d+)", f).group(1))):
            j = json.load(open(f)); s = int(re.search(r"val_step(\d+)", f).group(1)) + 1
            dr = drift.get(s, {})
            rows.append({"section": sec, "arm": arm, "run_dir": os.path.basename(d), "step": s, "num_eval_images": j["num_eval_images"],
                         **{x: j.get(x, "") for x in F}, "train_time_s": j.get("train_time_s", ""),
                         "drift_rot_deg_mean": dr.get("rot_deg_mean", ""), "drift_pos_mean": dr.get("pos_mean", "")})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
