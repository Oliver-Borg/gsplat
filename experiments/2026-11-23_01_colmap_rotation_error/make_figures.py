import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = os.path.dirname(os.path.abspath(__file__))
R = sorted([r for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))) if r["step"] == "7000"], key=lambda r: float(r["bound"]))
x = [float(r["train_rot_err_mean_deg"]) for r in R]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, (k, t) in zip(ax, [("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS")]):
    a.plot(x, [float(r[k]) for r in R], marker="o")
    a.set_xscale("symlog", linthresh=0.005); a.set_xticks([0, 0.01, 0.1, 1]); a.set_xticklabels(["0", "0.01", "0.1", "1"]); a.minorticks_off()
    a.set_title(t, fontsize=10); a.set_xlabel("mean rotation error of the training cameras (deg)")
plt.tight_layout(); plt.savefig(os.path.join(d, "figures/metrics_vs_error.png"), dpi=110); plt.close()
