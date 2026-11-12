"""rre/rte of a COLMAP reconstruction vs GT, same code path as camera_head_gt_smoke.py (Parser normalize=True, GT aligned to predictions)."""
import sys, json
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from datasets.colmap import Parser
from evaluation import calculate_metrics

pred_dir, gt_dir = sys.argv[1], sys.argv[2]
p = Parser(data_dir=pred_dir, factor=1, normalize=True, test_every=8)
g = Parser(data_dir=gt_dir, factor=2, normalize=False, test_every=8)
names = sorted(set(p.image_names) & set(g.image_names))
m = calculate_metrics({n: g.get_camera_positions([n])[0] for n in names}, {n: p.get_camera_positions([n])[0] for n in names})
print(json.dumps({"n": len(names), "rre": m["mean_rre_deg"], "rte": m["mean_rte"], "scale": m["alignment_scale"]}))
