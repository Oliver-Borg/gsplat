"""Figures of the experiment log from diagnostics_metrics.csv, diagnostics*/drift_*.jsonl and the numbers below."""
import csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = ["lr1e-4", "lr1e-5", "lr1e-6", "freezecam", "freezedepth", "lora"]
LABEL = {"lr1e-4": "full, lr 1e-4", "lr1e-5": "full, lr 1e-5", "lr1e-6": "full, lr 1e-6",
         "freezecam": "camera head frozen", "freezedepth": "depth head frozen", "lora": "LoRA r16, lr 1e-4"}
COL = dict(zip(RUNS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))  # categorical slots 1-6
MARK = dict(zip(RUNS, ["o", "s", "^", "D", "v", "P"]))
INK, GRID = "#3b3a36", "#e3e2dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})

rows = list(csv.DictReader(open("diagnostics_metrics.csv")))
for r in rows:
    for k in ("step", "psnr", "ssim", "lpips", "eval_rre", "eval_rte", "step_total_ms"):
        r[k] = float(r[k])


def series(tag, run, key):
    rr = sorted([r for r in rows if r["set"] == tag and r["run"] == run], key=lambda r: r["step"])
    return [r["step"] for r in rr], [r[key] for r in rr]


def style_legend(fig, handles_from_ax):
    h, l = handles_from_ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))


def two_sets(ax, key, ylabel, log=False):
    for run in RUNS:
        x, y = series("first", run, key)
        ax.plot(x, y, color=COL[run], lw=1.2, ls="--", alpha=0.65)
        x, y = series("rerun", run, key)
        ax.plot(x, y, color=COL[run], lw=2, marker=MARK[run], ms=6, mec="white", mew=1.2, label=LABEL[run])
    ax.set_xlabel("training step"); ax.set_ylabel(ylabel)
    ax.set_xticks([1, 20, 40, 80])
    if log:
        ax.set_yscale("log")


# 1. camera error
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
two_sets(axs[0], "eval_rre", "eval_rre (log)", log=True)
two_sets(axs[1], "eval_rte", "eval_rte (log)", log=True)
axs[0].set_title("eval_rre", loc="left"); axs[1].set_title("eval_rte", loc="left")
style_legend(fig, axs[0]); fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("figures/camera_error.png", dpi=110); plt.close(fig)

# 2. quality
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, k, t in zip(axs, ("psnr", "ssim", "lpips"), ("PSNR (dB)", "SSIM", "LPIPS")):
    two_sets(ax, k, t); ax.set_title(t, loc="left")
style_legend(fig, axs[0]); fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("figures/quality.png", dpi=110); plt.close(fig)

# 3. camera drift
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, key, t in zip(axs, ("rot_deg_mean", "pos_mean", "focal_rel_mean"),
                      ("rotation change, mean (deg)", "position change, mean (trainer units)", "focal change, mean (relative)")):
    for run in RUNS:
        for tag, d, kw in (("first", "diagnostics", dict(ls="--", lw=1.2, alpha=0.65)), ("rerun", "diagnostics_rerun", dict(lw=2))):
            L = [json.loads(l) for l in open(f"{d}/drift_{run}.jsonl")]
            ax.plot([x["step"] for x in L], [max(x[key], 1e-5) for x in L], color=COL[run], **kw,
                    **({"label": LABEL[run]} if tag == "rerun" else {}))
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_title(t, loc="left", fontsize=9)
style_legend(fig, axs[0]); fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("figures/camera_drift.png", dpi=110); plt.close(fig)

# 4. timings
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.2, 1, 1]})
secs = ["ft_forward", "sampling", "rasterize", "loss", "rasterize_backward", "heads_backward", "optimizer"]
full = [554.3, 3.4, 7.8, 1.3, 23.9, 1350.9, 82.5]
part = [45.5, 8.9, 14.4, 0.9, 24.3, 40.9, 121.2]
y = range(len(secs))
ax = axs[0]
ax.barh([i - 0.2 for i in y], full, height=0.36, color="#2a78d6", label="all 40 frames with grad (total 2024 ms)")
ax.barh([i + 0.2 for i in y], part, height=0.36, color="#eb6834", label="partial grad, 1 frame (total 256 ms)")
ax.set_yticks(list(y)); ax.set_yticklabels(secs); ax.invert_yaxis(); ax.set_xlabel("ms / step")
ax.set_title("step profile (smoke runs)", loc="left"); ax.legend(frameon=False, fontsize=7, loc="center right"); ax.grid(axis="y", visible=False)
bench = [("camera fwd no_grad, 40 fr", 16.3), ("camera fwd+bwd, 40 fr", 26.5), ("camera fwd+bwd, 1 fr", 22.4),
         ("depth fwd no_grad, 40 fr", 803.3), ("depth fwd+bwd, 1 fr", 65.2), ("depth fwd+bwd, 8 fr", 389.4)]
ax = axs[1]
ax.barh(range(len(bench)), [b[1] for b in bench], height=0.5, color="#1baf7a")
ax.set_yticks(range(len(bench))); ax.set_yticklabels([b[0] for b in bench]); ax.invert_yaxis()
ax.set_xlabel("ms"); ax.set_title("head benchmark", loc="left"); ax.grid(axis="y", visible=False)
ax = axs[2]
import numpy as np
ys = np.arange(len(RUNS))
for off, tag, c, lab in ((-0.2, "first", "#9ec5f4", "first run"), (0.2, "rerun", "#2a78d6", "rerun")):
    v = [series(tag, r, "step_total_ms")[1][0] for r in RUNS]
    ax.barh(ys + off, v, height=0.36, color=c, label=lab)
ax.set_yticks(ys); ax.set_yticklabels([LABEL[r] for r in RUNS]); ax.invert_yaxis()
ax.set_xlim(0, 1550); ax.set_xlabel("ms / step"); ax.set_title("diagnostic step time", loc="left"); ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("figures/timing.png", dpi=110); plt.close(fig)
