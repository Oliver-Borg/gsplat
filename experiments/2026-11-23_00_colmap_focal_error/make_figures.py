import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = os.path.dirname(os.path.abspath(__file__))
R = [r for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))) if r["step"] == "7000"]
base = [r for r in R if r["kind"] == "baseline"][0]
fixed = sorted([r for r in R if r["kind"] == "fixed"] + [base], key=lambda r: float(r["setting_pct"]))
rand = sorted([r for r in R if r["kind"] == "random"], key=lambda r: float(r["setting_pct"]))
M = [("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS")]
# 1: fixed focal change, signed
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, (k, t) in zip(ax, M):
    a.plot([float(r["setting_pct"]) for r in fixed], [float(r[k]) for r in fixed], marker="o")
    a.axvline(0, color="grey", lw=0.5); a.set_title(t, fontsize=10); a.set_xlabel("fixed focal change of every training view (%)")
plt.tight_layout(); plt.savefig(os.path.join(d, "figures/metrics_fixed_signed.png"), dpi=110); plt.close()
# 2: fixed (negative / positive) and random vs mean absolute focal error (fx and fy averaged), symlog x
mean_abs = lambda r: 0.5 * (float(r["fx_change_abs_mean_pct"]) + float(r["fy_change_abs_mean_pct"]))
series = [("fixed, negative change", [base] + [r for r in fixed if float(r["setting_pct"]) < 0][::-1]),
          ("fixed, positive change", [base] + [r for r in fixed if float(r["setting_pct"]) > 0]),
          ("random per view (up to +-x%)", [base] + rand)]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, (k, t) in zip(ax, M):
    for lab, rr in series:
        a.plot([mean_abs(r) for r in rr], [float(r[k]) for r in rr], marker="o", label=lab)
    a.set_xscale("symlog", linthresh=0.05); a.set_xticks([0, 0.1, 1, 5, 10]); a.set_xticklabels(["0", "0.1", "1", "5", "10"]); a.minorticks_off(); a.set_title(t, fontsize=10); a.set_xlabel("mean absolute focal error of the training views (%)")
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(d, "figures/metrics_vs_abs_error.png"), dpi=110); plt.close()
