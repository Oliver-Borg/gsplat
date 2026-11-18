import csv, glob, json, os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
runs = {
    "1. GT cameras (fixed)": "combined_outputs/*omega_combined_gtcams*evalall_steps15000",
    "2. Head full lr 1e-5, GT loss": "vggt_outputs/*cft-1e-05-full-d0-gtloss_evalall_steps15000",
    "3. Head LoRA r16 lr 1e-4, GT loss": "vggt_outputs/*cft-0.0001-lora16-d0-gtloss_evalall_steps15000",
    "4. Omega cameras (fixed)": "vggt_outputs/*omega*minfarthestpose_i40_gteval_evalall_steps15000",
    "5. COLMAP cameras (fixed)": "combined_outputs/*omega_combined_colmapcams*evalall_steps15000",
    "6. Pose residual MLP (3RGS) lr 1e-4, GT loss": "vggt_outputs/*cft-0.0001-mlp-d0-gtloss_evalall_steps15000",
}
rows = []
for arm, g in runs.items():
    d = glob.glob(os.path.join(root, "results", g))
    assert len(d) == 1, (arm, d)
    d = d[0]
    for f in sorted(glob.glob(d + "/stats/val_step*.json"), key=lambda f: int(re.search(r"val_step(\d+)", f).group(1))):
        j = json.load(open(f))
        s = int(re.search(r"val_step(\d+)", f).group(1)) + 1
        rows.append({"arm": arm, "run_dir": os.path.basename(d), "step": s, "num_eval_images": j["num_eval_images"],
                     **{k: j[k] for k in ("psnr", "ssim", "lpips", "num_GS", "eval_rre", "eval_rte")},
                     "train_time_s": j.get("train_time_s", ""), "train_steps": j.get("train_steps", "")})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
