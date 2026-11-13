import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
arms = list(dict.fromkeys(r["arm"] for r in rows))
fig, ax = plt.subplots(1, 5, figsize=(17, 3.2))
for arm in arms:
    r = [x for x in rows if x["arm"] == arm]
    for a, k in zip(ax, ["psnr", "ssim", "lpips", "eval_rre", "eval_rte"]):
        a.plot([int(x["step"]) for x in r], [float(x[k]) for x in r], marker="o", label=arm)
for a, t in zip(ax, ["PSNR", "SSIM", "LPIPS", "eval_rre (deg)", "eval_rte"]):
    a.set_title(t, fontsize=10); a.set_xlabel("step")
ax[3].set_yscale("log"); ax[4].set_yscale("log"); ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(d, "figures/metrics.png"), dpi=110)
