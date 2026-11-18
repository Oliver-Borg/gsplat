import json, os, glob
S="/tmp/claude-6786/-home-ojab2-work-git-vggt/bd27083c-05fc-4582-b862-58359a843fe4/scratchpad/diag_rerun"
runs=["lr1e-4","lr1e-5","lr1e-6","freezecam","freezedepth","lora"]
for r in runs:
    d=f"{S}/{r}/stats"
    if not os.path.exists(d): print(r,"MISSING"); continue
    print("##",r)
    print("| step | PSNR | SSIM | LPIPS | eval_rre | eval_rte |"); print("|---|---|---|---|---|---|")
    for st in ("0000","0019","0039","0079"):
        f=f"{d}/val_step{st}.json"
        if os.path.exists(f):
            v=json.load(open(f)); print(f"| {int(st)+1} | {v['psnr']:.2f} | {v['ssim']:.3f} | {v['lpips']:.3f} | {v['eval_rre']:.2f} | {v['eval_rte']:.3f} |")
    p=json.load(open(f"{d}/profile_vggt_ft.json")); print("step_total_ms %.0f, peak_mem_gb %.1f"%(p["step_total_ms"],p["peak_mem_gb"]))
    f=f"{d}/ft_drift.jsonl"
    if os.path.exists(f):
        L=[json.loads(l) for l in open(f)]
        print("| step | rot deg mean/max | pos mean/max | focal rel mean/max |"); print("|---|---|---|---|")
        for x in L:
            if x["step"] in (5,20,40,60,80): print(f"| {x['step']} | {x['rot_deg_mean']:.2f}/{x['rot_deg_max']:.2f} | {x['pos_mean']:.4f}/{x['pos_max']:.4f} | {x['focal_rel_mean']:.4f}/{x['focal_rel_max']:.4f} |")
