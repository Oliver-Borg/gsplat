"""Collects the eval metrics of the omega_camera_finetune runs into metrics.csv and copies the camera drift logs."""
import csv, glob, json, os, re, shutil

R = "/home/ojab2/work/git/gsplat/results/vggt_outputs/"
base = "bonsai_2_n40_s42_omega_c0.0_p100000_random_minfarthestpose_i40_gteval_"
arms = [
    ("1. MCMC baseline", "steps15000"),
    ("2. full, lr 1e-4", "cft-0.0001-full-d0_steps15000"),
    ("3. full, lr 1e-5", "cft-1e-05-full-d0_steps15000"),
    ("4. full, lr 1e-6", "cft-1e-06-full-d0_steps15000"),
    ("5. LoRA r16, lr 1e-4", "cft-0.0001-lora16-d0_steps15000"),
    ("6. full, lr 1e-4, from step 5000", "cft-0.0001-full-d5000_steps15000"),
    ("7. LoRA r16, lr 1e-4, from step 5000", "cft-0.0001-lora16-d5000_steps15000"),
]
rows = []
for label, suf in arms:
    d = R + base + suf + "/stats"
    for f in glob.glob(d + "/val_step*.json"):
        step = int(re.search(r"val_step(\d+)", f).group(1)) + 1
        v = json.load(open(f))
        rows.append(dict(arm=label, run=suf, step=step, psnr=v["psnr"], ssim=v["ssim"], lpips=v["lpips"],
                         eval_rre=v["eval_rre"], eval_rte=v["eval_rte"], num_GS=v["num_GS"]))
    if os.path.exists(d + "/ft_drift.jsonl"):
        shutil.copy(d + "/ft_drift.jsonl", "drift_" + suf.split("_steps")[0] + ".jsonl")
rows.sort(key=lambda r: (r["arm"], r["step"]))
with open("metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["arm"][:38].ljust(38), r["step"], *(f"{r[k]:.3g}" for k in ("psnr", "ssim", "lpips", "eval_rre", "eval_rte")), r["num_GS"])
