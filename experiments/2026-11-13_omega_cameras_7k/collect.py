import csv, glob, json, os, sys
N = sys.argv[1] if len(sys.argv) > 1 else "40"  # number of training images: 40 or 255
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
runs40 = {
    "1. GT cameras": "combined_outputs/*omega_combined_gtcams*steps7000",
    "2. Fine-tuned head (full, lr 1e-5)": "vggt_outputs/*omega*ftcams-full_steps7000",
    "3. Fine-tuned head (LoRA r16, lr 1e-4)": "vggt_outputs/*omega*ftcams-lora_steps7000",
    "4. Omega cameras": "vggt_outputs/*omega*minfarthestpose_i40_gteval_steps7000",
    "5. COLMAP cameras": "combined_outputs/*omega_combined_colmapcams*steps7000",
}
runs255 = {
    "1. GT cameras": "combined_outputs/*n255*omega_combined_gtcams*steps7000",
    "2. Fine-tuned head (full, lr 1e-5)": "vggt_outputs/*n255*omega*ftcams-full_steps7000",
    "3. Fine-tuned head (LoRA r16, lr 1e-4)": "vggt_outputs/*n255*omega*ftcams-lora_steps7000",
    "4. Omega cameras": "vggt_outputs/*n255*omega*farthestpose_i255_gteval_steps7000",
    "5. COLMAP cameras": "combined_outputs/*n255*omega_combined_colmapcams*steps7000",
}
runs = runs40 if N == "40" else runs255
rows = []
for arm, g in runs.items():
    d = glob.glob(os.path.join(root, "results", g))[0]
    for f in sorted(glob.glob(d + "/stats/val_step*.json")):
        j = json.load(open(f))
        rows.append({"arm": arm, "run_dir": os.path.basename(d), "step": int(f[-9:-5]) + 1 if f[-9:-5] != "0000" else 1,
                     **{k: j[k] for k in ("psnr", "ssim", "lpips", "num_GS", "eval_rre", "eval_rte")}})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.csv" if N == "40" else "metrics_n255.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
