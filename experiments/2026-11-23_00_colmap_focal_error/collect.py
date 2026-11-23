import csv, glob, json, os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
rows = []
for d in sorted(glob.glob(os.path.join(root, "results/gt_outputs/bonsai_2_n255_s42_i292_*steps7000"))):
    name = os.path.basename(d)
    m = re.search(r"kfix([+-][0-9.]+)", name); r = re.search(r"krand([0-9.]+)", name)
    kind = "fixed" if m else "random" if r else "baseline"
    bound = float(m.group(1)) if m else float(r.group(1)) if r else 0.0
    for f in sorted(glob.glob(d + "/stats/val_step*.json"), key=lambda f: int(re.search(r"val_step(\d+)", f).group(1))):
        j = json.load(open(f)); s = int(re.search(r"val_step(\d+)", f).group(1)) + 1
        ch = lambda k: j.get(k, 0.0)
        rows.append({"run_dir": name, "kind": kind, "setting_pct": bound, "step": s, "num_eval_images": j["num_eval_images"],
                     **{k: j[k] for k in ("psnr", "ssim", "lpips", "num_GS")},
                     "fx_change_mean_pct": ch("train_k_fx_change_mean_pct"), "fy_change_mean_pct": ch("train_k_fy_change_mean_pct"),
                     "fx_change_abs_mean_pct": ch("train_k_fx_change_abs_mean_pct"), "fy_change_abs_mean_pct": ch("train_k_fy_change_abs_mean_pct"),
                     "train_time_s": j.get("train_time_s", "")})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows,", len({r["run_dir"] for r in rows}), "runs")
