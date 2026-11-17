import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
arms = list(dict.fromkeys(r["arm"] for r in rows))
fig, ax = plt.subplots(2, 3, figsize=(12, 6.4)); ax = ax.flatten()
for arm in arms:
    r = [x for x in rows if x["arm"] == arm]
    for a, k in zip(ax, ["psnr", "ssim", "lpips", "eval_rre", "eval_rte", "train_time_s"]):
        pts = [(int(x["step"]), float(x[k])) for x in r if x[k] != "" and (k != "train_time_s" or int(x["step"]) > 1)]
        a.plot(*zip(*pts), marker="o", label=arm)
for a, t in zip(ax, ["PSNR", "SSIM", "LPIPS", "eval_rre (deg)", "eval_rte", "cumulative train time (s)"]):
    a.set_title(t, fontsize=10); a.set_xlabel("step")
ax[3].set_yscale("log"); ax[4].set_yscale("log"); ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(d, "figures/metrics.png"), dpi=110)
