import csv, glob, json, os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
E_ = "_evalall_steps15000"
ARMS = {  # key -> (label, glob)
    "gt": ("GT cameras (fixed)", "combined_outputs/*omega_combined_gtcams*evalall_steps15000"),
    "om": ("Omega cameras (fixed)", "vggt_outputs/*omega*minfarthestpose_i40_gteval_evalall_steps15000"),
    "co": ("COLMAP cameras (fixed)", "combined_outputs/*omega_combined_colmapcams*evalall_steps15000"),
    "def0": ("default module, photometric, delay 0", "vggt_outputs/*cft-0.0001-pdef-d0" + E_),
    "def7": ("default module, photometric, delay 7500", "vggt_outputs/*cft-0.0001-pdef-d7500" + E_),
    "3rgs0": ("3RGS MLP, photometric, delay 0", "vggt_outputs/*cft-0.0001-mlp-d0" + E_),
    "3rgs7": ("3RGS MLP, photometric, delay 7500", "vggt_outputs/*cft-0.0001-mlp-d7500" + E_),
    "full0": ("head full lr 1e-5, photometric, delay 0", "vggt_outputs/*cft-1e-05-full-d0" + E_),
    "full7": ("head full lr 1e-5, photometric, delay 7500", "vggt_outputs/*cft-1e-05-full-d7500" + E_),
    "lora0": ("head LoRA r16 lr 1e-4, photometric, delay 0", "vggt_outputs/*cft-0.0001-lora16-d0" + E_),
    "lora7": ("head LoRA r16 lr 1e-4, photometric, delay 7500", "vggt_outputs/*cft-0.0001-lora16-d7500" + E_),
}
SECTIONS = {  # section -> arms (a run can appear in several sections)
    "fixed": ["def7", "3rgs7", "full7", "lora7", "gt", "om", "co"],
    "modules": ["def0", "def7", "3rgs0", "3rgs7", "om"],
    "heads": ["full0", "full7", "lora0", "lora7", "om"],
}
rows = []
for sec, keys in SECTIONS.items():
    for k in keys:
        arm, g = ARMS[k]
        d = glob.glob(os.path.join(root, "results", g)); assert len(d) == 1, (arm, d); d = d[0]
        for f in sorted(glob.glob(d + "/stats/val_step*.json"), key=lambda f: int(re.search(r"val_step(\d+)", f).group(1))):
            j = json.load(open(f)); s = int(re.search(r"val_step(\d+)", f).group(1)) + 1
            rows.append({"section": sec, "arm": arm, "run_dir": os.path.basename(d), "step": s, "num_eval_images": j["num_eval_images"],
                         **{x: j[x] for x in ("psnr", "ssim", "lpips", "num_GS", "eval_rre", "eval_rte")},
                         "train_time_s": j.get("train_time_s", ""), "train_steps": j.get("train_steps", "")})
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
