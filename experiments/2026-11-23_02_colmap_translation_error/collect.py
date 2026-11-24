import csv, glob, json, os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
TOK = "trand"
rows = []
for d in sorted(glob.glob(os.path.join(root, "results/gt_outputs/bonsai_2_n255_s42_i292_*steps7000"))):
    name = os.path.basename(d)
    m = re.search(TOK + r"([0-9.]+)", name)
    if not m and not re.search(r"i292_steps7000", name):
        continue
    bound = float(m.group(1)) if m else 0.0
    for f in sorted(glob.glob(d + "/stats/val_step*.json"), key=lambda f: int(re.search(r"val_step(\d+)", f).group(1))):
        j = json.load(open(f)); s = int(re.search(r"val_step(\d+)", f).group(1)) + 1
        rows.append({"run_dir": name, "bound": bound, "step": s, "num_eval_images": j["num_eval_images"],
                     **{k: j[k] for k in ("psnr", "ssim", "lpips", "num_GS")},
                     **{k: j.get(k, 0.0) for k in ("train_pose_scale", "train_rot_err_mean_deg", "train_rot_err_max_deg", "train_trans_err_mean_pct", "train_trans_err_max_pct")},
                     "train_time_s": j.get("train_time_s", "")})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows,", len({r["run_dir"] for r in rows}), "runs")
