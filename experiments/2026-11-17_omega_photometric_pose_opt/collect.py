import csv, glob, json, os, re
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
E_ = "_evalall_steps15000"
ARMS = {  # key -> (label, glob)
    "gt": ("GT cameras (fixed)", "combined_outputs/*omega_combined_gtcams*evalall_steps15000"),
    "om": ("Omega cameras (fixed)", "vggt_outputs/*omega*minfarthestpose_i40_gteval_evalall_steps15000"),
    "co": ("COLMAP cameras (fixed)", "combined_outputs/*omega_combined_colmapcams*evalall_steps15000"),
    "p_def": ("default module, photometric", "vggt_outputs/*cft-0.0001-pdef-d0" + E_),
    "p_3rgs": ("3RGS MLP, photometric", "vggt_outputs/*cft-0.0001-mlp-d0" + E_),
    "p_full": ("head full lr 1e-5, photometric", "vggt_outputs/*cft-1e-05-full-d0" + E_),
    "p_lora": ("head LoRA r16 lr 1e-4, photometric", "vggt_outputs/*cft-0.0001-lora16-d0" + E_),
    "g_def": ("default module, GT loss (Huber enc)", "vggt_outputs/*cft-0.0001-pdef-d0-gtloss" + E_),
    "g_3rgs": ("3RGS MLP, GT loss (Huber enc)", "vggt_outputs/*cft-0.0001-mlp-d0-gtloss" + E_),
    "g_full": ("head full lr 1e-5, GT loss", "vggt_outputs/*cft-1e-05-full-d0-gtloss" + E_),
    "g_lora": ("head LoRA r16 lr 1e-4, GT loss", "vggt_outputs/*cft-0.0001-lora16-d0-gtloss" + E_),
}
SECTIONS = {  # section -> arms (a run can appear in several sections)
    "fixed": ["p_def", "p_3rgs", "p_full", "p_lora", "gt", "om", "co"],
    "modules": ["p_def", "p_3rgs", "g_def", "g_3rgs", "om"],
    "heads": ["p_full", "p_lora", "g_full", "g_lora", "om"],
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
