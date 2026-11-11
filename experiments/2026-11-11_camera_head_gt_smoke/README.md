# Omega camera head fine-tuned directly against GT cameras (smoke test, no rendering)

## Question
Smoke test: fine-tune only the VGGT-Omega camera head against ground-truth camera poses, with the GT aligned to the
head's first (initial) output. No 3DGS training, no rendering.

## Setup
- Scene: Mip-NeRF 360 `bonsai`, 40 training images, Omega 512 checkpoint, reconstruction
  `bonsai_2_n40_s42_omega_c0.0_p100000_random_minfarthestpose` (cache `bonsai_2_cache_n40_s42_omega_minfarthestpose/raw_preds.pt`).
- Cached final-layer camera+register tokens of all 40 frames; the camera head runs on all 40 frames with gradient every step.
- Targets: GT cameras (`data/360_v2/bonsai`, factor 2) aligned once at step 0 to the initial predicted cameras with
  `stochastic_umeyama_alignment` on camera centres (scale s, R, t; GT rotations multiplied by R); then fixed.
  Targets are encoded as the 9-D Omega pose encoding (T, quaternion, fov_h, fov_w) in the Omega world frame
  (quaternion sign chosen to match the initial prediction). fov comes from the GT intrinsics scaled to the network
  resolution (624x416) per axis. Filled gap (not decided by the user): fov is part of the loss.
- Loss: Huber (`smooth_l1`) on the pose encoding, mean over all 40 x 9 values.
- Optimiser: Adam, constant lr, 200 steps. Filled gaps: steps and lrs (full head lr 1e-5; LoRA r16, alpha 16, lr 1e-4).
- Metrics per step: mean rre (deg) / rte, GT aligned to the current predicted cameras (same `calculate_metrics` call as
  the trainer's `eval_rre`/`eval_rte`), trainer frame, on the 40 training frames. Step 0 equals the Omega baseline
  (rre 0.4134, rte 0.01296).
- Device: CPU (all four GPUs had other users' jobs at launch), `CUDA_VISIBLE_DEVICES=` unset-GPU; about 1.7 s per step on a
  loaded 32-core machine (load average ~48). Seed 42, no determinism check.
- Script: `examples/camera_head_gt_smoke.py`.

## Launch
```bash
cd ~/work/git/gsplat
C="--device cpu --cache_file ../vggt/vggt_outputs/bonsai_2_cache_n40_s42_omega_minfarthestpose/raw_preds.pt \
   --data_dir ../vggt/vggt_outputs/bonsai_2_n40_s42_omega_c0.0_p100000_random_minfarthestpose \
   --gt_dir ../vggt/data/360_v2/bonsai --steps 200"
CUDA_VISIBLE_DEVICES= env -u PYTHONPATH $HOME/.conda/envs/gsplat/bin/python examples/camera_head_gt_smoke.py $C --lr 1e-5 --out full_lr1e-5.jsonl
CUDA_VISIBLE_DEVICES= env -u PYTHONPATH $HOME/.conda/envs/gsplat/bin/python examples/camera_head_gt_smoke.py $C --lora --lr 1e-4 --out lora16_lr1e-4.jsonl
```

## Run metadata
- gsplat: commit `55e6920d` + uncommitted `examples/camera_head_gt_smoke.py` (committed with this log); vggt: `add5873`.
- Env: conda `gsplat`, shell `PYTHONPATH` unset; `vggt_omega` imported from `~/work/git/vggt-omega` (`48b23c8`).
- Raw per-step data: `full_lr1e-5.jsonl`, `lora16_lr1e-4.jsonl`; figure script `make_figures.py`.

## Results
Per-step loss, rre and rte (mean over the 40 train frames):

![gt smoke](figures/gt_smoke.png)

| Step | Full lr 1e-5: loss / rre / rte | LoRA r16 lr 1e-4: loss / rre / rte |
|---|---|---|
| 0 | 1.05e-5 / 0.4134 / 0.01296 | 1.05e-5 / 0.4134 / 0.01296 |
| 50 | 1.46e-6 / 0.1482 / 0.00661 | 2.23e-6 / 0.1989 / 0.00775 |
| 100 | 1.21e-6 / 0.1320 / 0.00587 | 1.76e-6 / 0.1596 / 0.00709 |
| 200 | 8.99e-7 / 0.1064 / 0.00486 | 1.37e-6 / 0.1277 / 0.00627 |

## Failed / missing runs, confounds
- No failed runs. Only the 40 training frames are in the loss and in the metrics (no held-out views).
- Single seed, one lr per arm, 200 steps, CPU; the GT alignment is fitted on the same 40 centres that are evaluated.

## Insights
To be filled by the user.
