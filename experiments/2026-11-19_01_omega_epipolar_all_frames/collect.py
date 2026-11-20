import csv, glob, json, os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
P = "vggt_outputs/*_n255_*omega_c0.0*farthestpose_i255_gteval_"
S = "_tpo_evalall_steps30000"
ARMS = {  # key -> (label, glob)
    "head_epi": ("head full lr 1e-5, epipolar+photometric", P + "cft-1e-05-full-d0-epi2x3000k10" + S),
    "head_photo": ("head full lr 1e-5, photometric", P + "cft-1e-05-full-d0" + S),
    "lora_epi": ("head LoRA r16 lr 1e-4, epipolar+photometric", P + "cft-0.0001-lora16-d0-epi2x3000k10" + S),
    "lora_photo": ("head LoRA r16 lr 1e-4, photometric", P + "cft-0.0001-lora16-d0" + S),
    "mlp_epi": ("3RGS MLP lr 1e-4, epipolar+photometric", P + "cft-0.0001-mlp-d0-epi2x3000k10" + S),
    "mlp_photo": ("3RGS MLP lr 1e-4, photometric", P + "cft-0.0001-mlp-d0" + S),
    "def_epi": ("default module lr 1e-4, epipolar+photometric", P + "cft-0.0001-pdef-d0-epi2x3000k10" + S),
    "def_photo": ("default module lr 1e-4, photometric", P + "cft-0.0001-pdef-d0" + S),
    "fixed": ("Omega cameras (fixed)", P[:-1] + "_tpo_evalall_steps30000"),
    "colmap": ("COLMAP cameras + point cloud (fixed)", "gt_outputs/bonsai_2_n255_s42_i292_tpo_steps30000"),
}
SECTIONS = {
    "fixed": ["head_epi", "lora_epi", "mlp_epi", "def_epi", "fixed", "colmap"],
    "modules": ["def_epi", "def_photo", "mlp_epi", "mlp_photo", "fixed", "colmap"],
    "heads": ["head_epi", "head_photo", "lora_epi", "lora_photo", "fixed", "colmap"],
}
F = ("psnr", "ssim", "lpips", "num_GS", "eval_rre", "eval_rte", "tpo_psnr", "tpo_ssim", "tpo_lpips")
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
