import csv, glob, json, os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
E_ = "_evalall_steps15000"
runs = [  # (experiment, arm, glob)
    ("loss", "GT cameras (fixed)", "combined_outputs/*omega_combined_gtcams*evalall_steps15000"),
    ("loss", "Omega cameras (fixed)", "vggt_outputs/*omega*minfarthestpose_i40_gteval_evalall_steps15000"),
    ("loss", "default, Huber enc", "vggt_outputs/*cft-0.0001-pdef-d0-gtloss" + E_),
    ("loss", "default, L1 c2w", "vggt_outputs/*cft-0.0001-pdef-d0-gtl1c2w" + E_),
    ("loss", "3RGS MLP, Huber enc", "vggt_outputs/*cft-0.0001-mlp-d0-gtloss" + E_),
    ("loss", "3RGS MLP, L1 c2w", "vggt_outputs/*cft-0.0001-mlp-d0-gtl1c2w" + E_),
    ("gtk", "GT cameras (fixed)", "combined_outputs/*omega_combined_gtcams*evalall_steps15000"),
    ("gtk", "Omega poses (fixed), GT K", "vggt_outputs/*cft-0.0001-fixed-d0-gtk" + E_),
    ("gtk", "default, Huber enc", "vggt_outputs/*cft-0.0001-pdef-d0-gtloss-gtk" + E_),
    ("gtk", "default, L1 c2w", "vggt_outputs/*cft-0.0001-pdef-d0-gtl1c2w-gtk" + E_),
    ("gtk", "3RGS MLP, Huber enc", "vggt_outputs/*cft-0.0001-mlp-d0-gtloss-gtk" + E_),
    ("gtk", "3RGS MLP, L1 c2w", "vggt_outputs/*cft-0.0001-mlp-d0-gtl1c2w-gtk" + E_),
]
rows = []
for ex, arm, g in runs:
    d = glob.glob(os.path.join(root, "results", g))
    assert len(d) == 1, (arm, d)
    d = d[0]
    for f in sorted(glob.glob(d + "/stats/val_step*.json"), key=lambda f: int(re.search(r"val_step(\d+)", f).group(1))):
        j = json.load(open(f))
        s = int(re.search(r"val_step(\d+)", f).group(1)) + 1
        rows.append({"experiment": ex, "arm": arm, "run_dir": os.path.basename(d), "step": s, "num_eval_images": j["num_eval_images"],
                     **{k: j[k] for k in ("psnr", "ssim", "lpips", "num_GS", "eval_rre", "eval_rte")},
                     "train_time_s": j.get("train_time_s", ""), "train_steps": j.get("train_steps", "")})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
