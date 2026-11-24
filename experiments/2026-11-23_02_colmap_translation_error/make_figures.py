import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = os.path.dirname(os.path.abspath(__file__))
R = sorted([r for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))) if r["step"] == "7000"], key=lambda r: float(r["bound"]))
x = [float(r["train_trans_err_mean_pct"]) for r in R]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, (k, t) in zip(ax, [("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS")]):
    a.plot(x, [float(r[k]) for r in R], marker="o")
    a.set_xscale("symlog", linthresh=0.05); a.set_xticks([0, 0.1, 1, 5, 10]); a.set_xticklabels(["0", "0.1", "1", "5", "10"]); a.minorticks_off()
    a.set_title(t, fontsize=10); a.set_xlabel("mean camera-centre error of the training cameras (% of scene scale)")
plt.tight_layout(); plt.savefig(os.path.join(d, "figures/metrics_vs_error.png"), dpi=110); plt.close()

# quality metrics together with the training-camera rotation (RRE) and translation (RTE) error vs the nominal bound
b = [float(r["bound"]) for r in R]
fig, ax = plt.subplots(2, 3, figsize=(12, 7))
for a, (k, t) in zip(ax[0], [("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS")]):
    a.plot(b, [float(r[k]) for r in R], marker="o"); a.set_title(t, fontsize=10)
for a, (k, t) in zip(ax[1], [("rot", "RRE of the training cameras (deg)"), ("trans", "RTE of the training cameras (% of scene scale)")]):
    kk = "train_rot_err_%s_deg" if k == "rot" else "train_trans_err_%s_pct"
    a.plot(b, [float(r[kk % "mean"]) for r in R], marker="o", label="mean")
    a.plot(b, [float(r[kk % "max"]) for r in R], marker="s", label="max"); a.set_title(t, fontsize=10); a.legend(fontsize=8)
ax[1][2].axis("off")
for a in ax.flat[:5]:
    a.set_xscale("symlog", linthresh=0.05); a.set_xticks([0, 0.1, 1, 5]); a.set_xticklabels([f"{v:g}" for v in [0, 0.1, 1, 5]]); a.minorticks_off(); a.set_xlabel("nominal translation bound x (% of scene scale)")
plt.tight_layout(); plt.savefig(os.path.join(d, "figures/metrics_with_rre_rte.png"), dpi=110); plt.close()
