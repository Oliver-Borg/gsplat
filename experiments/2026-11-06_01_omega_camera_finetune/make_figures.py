"""Figures of the log from metrics.csv and drift_*.jsonl."""
import csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["1. MCMC baseline", "2. full, lr 1e-4", "3. full, lr 1e-5", "4. full, lr 1e-6", "5. LoRA r16, lr 1e-4",
        "6. full, lr 1e-4, from step 5000", "7. LoRA r16, lr 1e-4, from step 5000"]
DRIFT = {ARMS[1]: "cft-0.0001-full-d0", ARMS[2]: "cft-1e-05-full-d0", ARMS[3]: "cft-1e-06-full-d0",
         ARMS[4]: "cft-0.0001-lora16-d0", ARMS[5]: "cft-0.0001-full-d5000", ARMS[6]: "cft-0.0001-lora16-d5000"}
COL = dict(zip(ARMS, ["#3b3a36", "#2a78d6", "#eb6834", "#1baf7a", "#008300", "#2a78d6", "#008300"]))
LS = dict(zip(ARMS, ["-", "-", "-", "-", "-", "--", "--"]))
MARK = dict(zip(ARMS, ["o", "s", "^", "D", "v", "s", "v"]))
INK, GRID = "#3b3a36", "#e3e2dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
rows = list(csv.DictReader(open("metrics.csv")))
for r in rows:
    for k in ("step", "psnr", "ssim", "lpips", "eval_rre", "eval_rte"):
        r[k] = float(r[k])


def metric(ax, key, ylabel, log=False):
    for a in ARMS:
        rr = sorted([r for r in rows if r["arm"] == a], key=lambda r: r["step"])
        ax.plot([r["step"] for r in rr], [r[key] for r in rr], color=COL[a], ls=LS[a], lw=2, marker=MARK[a], ms=6,
                mec="white", mew=1.2, label=a)
    ax.set_xlabel("training step"); ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.set_xticks([1, 3000, 7000, 10000, 15000]); ax.set_xticklabels(["1", "3k", "7k", "10k", "15k"])


def legend(fig, ax, ncol=3):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False)


fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
metric(axs[0], "eval_rre", "eval_rre (log)", True); metric(axs[1], "eval_rte", "eval_rte (log)", True)
axs[0].set_title("eval_rre", loc="left"); axs[1].set_title("eval_rte", loc="left")
legend(fig, axs[0]); fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig("figures/camera_error.png", dpi=110); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
for ax, k, t in zip(axs, ("psnr", "ssim", "lpips"), ("PSNR (dB)", "SSIM", "LPIPS")):
    metric(ax, k, t); ax.set_title(t, loc="left")
legend(fig, axs[0]); fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig("figures/quality.png", dpi=110); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8))
for ax, key, t in zip(axs, ("rot_deg_mean", "pos_mean", "focal_rel_mean"),
                      ("rotation change, mean (deg)", "position change, mean (trainer units)", "focal change, mean (relative)")):
    for a, f in DRIFT.items():
        L = [json.loads(l) for l in open(f"drift_{f}.jsonl")]
        ax.plot([x["step"] for x in L], [max(x[key], 1e-5) for x in L], color=COL[a], ls=LS[a], lw=1.8, label=a)
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_title(t, loc="left", fontsize=9)
legend(fig, axs[0], 2); fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig("figures/camera_drift.png", dpi=110); plt.close(fig)
