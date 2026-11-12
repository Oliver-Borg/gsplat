import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
d = os.path.dirname(os.path.abspath(__file__))
arms = {"full head, lr 1e-5": "full_lr1e-5.jsonl", "LoRA r16, lr 1e-4": "lora16_lr1e-4.jsonl"}
# reference, same code path (examples/colmap_reference_metrics.py): COLMAP run on the same 40 images
COLMAP = {"rre": 0.5092, "rte": 0.05524}
INIT = {"rre": 0.4134, "rte": 0.01296}  # Omega initial prediction (step 0)
rows = {lab: [json.loads(l) for l in open(os.path.join(d, f))] for lab, f in arms.items()}
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
for lab, r in rows.items():
    for a, k in zip(ax, ["loss", "rre", "rte"]):
        a.plot([x["step"] for x in r], [x[k] for x in r], label=lab)
    t = np.array([x["t_train_s"] for x in r[1:]]); ax[3].plot(range(1, len(t) + 1), t, lw=0.7, label=lab)
for a, k in zip(ax[1:3], ["rre", "rte"]):
    a.axhline(COLMAP[k], color="k", ls="--", lw=1, label="COLMAP (same 40 images)")
    a.axhline(INIT[k], color="gray", ls=":", lw=1, label="Omega initial")
for a, t in zip(ax, ["Huber loss (pose encoding)", "rre (deg, vs GT)", "rte (vs GT)", "train time per step (s, CPU)"]):
    a.set_title(t, fontsize=10); a.set_xlabel("step")
ax[0].set_yscale("log"); ax[1].set_yscale("log"); ax[2].set_yscale("log")
ax[1].legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(d, "figures/gt_smoke.png"), dpi=110)
