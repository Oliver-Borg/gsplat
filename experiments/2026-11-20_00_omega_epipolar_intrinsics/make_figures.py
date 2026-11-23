import csv, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(d, "../..")
rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
for ex in ("meank", "intropt_3rgs", "intropt_default", "intropt_omega"):
    R = [r for r in rows if r["section"] == ex]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.4)); ax = ax.flatten()
    for arm in dict.fromkeys(r["arm"] for r in R):
        r = [x for x in R if x["arm"] == arm]
        for a, k in zip(ax, ["psnr", "ssim", "lpips", "eval_rre", "eval_rte", "drift_rot_deg_mean"]):
            pts = [(int(x["step"]), float(x[k])) for x in r if x[k] != ""]
            if pts: a.plot(*zip(*pts), marker="o", label=arm)
    for a, t in zip(ax, ["PSNR", "SSIM", "LPIPS", "eval_rre (deg)", "eval_rte", "drift mean rotation (deg)"]):
        a.set_title(t, fontsize=10); a.set_xlabel("step")
    ax[3].set_yscale("log"); ax[4].set_yscale("log"); ax[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(d, f"figures/metrics_{ex}.png"), dpi=110); plt.close()
