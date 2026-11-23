"""Render figure: ground truth and the render of every run (0% baseline, fixed and random focal changes) for three test views.
Panels are cut from the evaluation images of each run (`renders/val_step6999_<idx>.png`: 5 panels side by side, panel 0 = ground
truth, panel 1 = render). 3 columns = test views 0, 2, 5."""
import csv, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
d = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(d, "../..")
R = [r for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))) if r["step"] == "7000"]
order = {"baseline": 0, "fixed": 1, "random": 2}
R.sort(key=lambda r: (order[r["kind"]], float(r["setting_pct"])))
lab = lambda r: "0% (COLMAP cameras)" if r["kind"] == "baseline" else (f"fixed {float(r['setting_pct']):+g}%" if r["kind"] == "fixed" else f"random up to +-{float(r['setting_pct']):g}%")
views = [0, 2, 5]
W = 520


def panel(run_dir, idx, p):
    im = Image.open(os.path.join(root, "results/gt_outputs", run_dir, "renders", f"val_step6999_{idx:04d}.png"))
    pw = im.width // 5
    c = im.crop((p * pw, 0, (p + 1) * pw, im.height))
    return c.resize((W, int(c.height * W / c.width)))


rows = [("Ground truth", None)] + [(lab(r), r) for r in R]
fig, ax = plt.subplots(len(rows), 3, figsize=(10, 2.1 * len(rows)))
for i, (name, r) in enumerate(rows):
    for j, v in enumerate(views):
        a = ax[i, j]
        a.imshow(panel(R[0]["run_dir"] if r is None else r["run_dir"], v, 0 if r is None else 1)); a.set_xticks([]); a.set_yticks([])
        if j == 0:
            a.set_ylabel(name if r is None else f"{name}\nPSNR {float(r['psnr']):.2f}\nLPIPS {float(r['lpips']):.3f}\nSSIM {float(r['ssim']):.3f}", fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
plt.tight_layout()
plt.savefig(os.path.join(d, "figures/renders.jpg"), dpi=100, pil_kwargs={"quality": 82})
