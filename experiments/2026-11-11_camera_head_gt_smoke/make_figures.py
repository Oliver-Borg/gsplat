import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = os.path.dirname(os.path.abspath(__file__))
arms = {"full head, lr 1e-5": "full_lr1e-5.jsonl", "LoRA r16, lr 1e-4": "lora16_lr1e-4.jsonl"}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for lab, f in arms.items():
    r = [json.loads(l) for l in open(os.path.join(d, f))]
    for a, k in zip(ax, ["loss", "rre", "rte"]):
        a.plot([x["step"] for x in r], [x[k] for x in r], label=lab)
for a, t in zip(ax, ["Huber loss (pose encoding)", "rre (deg, vs GT)", "rte (vs GT)"]):
    a.set_title(t, fontsize=10); a.set_xlabel("step"); a.set_yscale("log" if t.startswith("Huber") else "linear")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(d, "figures/gt_smoke.png"), dpi=110)
