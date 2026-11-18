# 3R-GS reproduction (Mip-NeRF 360 bonsai and counter)

## Question
Does the ported 3R-GS code (https://github.com/zsh523/3rgs, arXiv 2504.04294) reproduce the paper's pose-optimisation results? Scenes: `bonsai` (not in the paper's tables) and `counter` (Tables 1 and 2).

## Setup
- Code: copy of the 3rgs `src/` in `examples/rgs3/` (`trainer.py`, `datasets/`, `utils/`); only the imports were changed (`from datasets.` -> `from rgs3.datasets.`, `from utils.` -> `from rgs3.utils.`). Run as a separate trainer, not part of `simple_trainer.py`.
- Data: the authors' MASt3R outputs (`MipNeRF360.tar.gz`, per scene `mast3r/*.npy`, `pointcloud.ply`, `images_{train,test}.txt`, `pose_gt_{train,test}.npy`), extracted to `~/work/data/3rgs_mast3r/extracted/MipNeRF360/`. Images are not in the archive; the local copies in `~/work/git/vggt/data/360_v2/{bonsai,counter}/images` (+ `sparse`) are used, linked from `~/work/data/3rgs_{bonsai,counter}`; `images_2` is created by the dataset parser (factor 2). bonsai: 255 train / 37 test views; counter: 240 images in the split lists.
- 30000 steps, `--data_factor 2`, seed 42 (trainer default), one run per cell.
- Arms (commands from the repo's `scripts/3dgs_train.sh` and `scripts/3rgs_train.sh`):
  - 3DGS baseline: `default` strategy, `--pose_opt_type sfm --no-use_corres_epipolar_loss` (cfg: pose_opt lr 1e-5, init_type sfm).
  - 3R-GS: `mcmc` strategy, `--pose_opt_type mlp --use_corres_epipolar_loss` (MLP pose module, epipolar loss weight 2.0).
- Launch (`queue.sh` bonsai, `queue_counter.sh` counter; both wait for a 0%-utilisation GPU, `*.sh` is gitignored), from `~/work/git/gsplat/examples`, `CUDA_VISIBLE_DEVICES=0`, `env -u PYTHONPATH ~/.conda/envs/gsplat/bin/python`:
```
python -m rgs3.trainer default --data_dir ~/work/data/3rgs_<scene> --data_factor 2 --result_dir ~/work/git/gsplat/results/3rgs_repro/3dgs/<scene> --pose_opt_type sfm --no-use_corres_epipolar_loss
python -m rgs3.trainer mcmc    --data_dir ~/work/data/3rgs_<scene> --data_factor 2 --result_dir ~/work/git/gsplat/results/3rgs_repro/mcmc-mlp-epi/<scene> --pose_opt_type mlp --use_corres_epipolar_loss
```
- Metrics (from the 3rgs trainer's eval): PSNR/SSIM/LPIPS (alex) on test views after test-time pose optimisation; ATE RMSE and mean per-frame rotation error on the train views (evo, Procrustes with scale, from `stats/plot/stats_<step>.json`).

## Run metadata
- gsplat commit `4a403345`; uncommitted at run time: `examples/rgs3/` (untracked).
- Env `gsplat`: gsplat 1.5.3, torch 2.10.0+cu130, numpy 1.26.4, kornia 0.8.2, evo 1.11.0 (`~/.evo/settings.json` plot_backend set to Agg); `plyfile` installed with `--no-deps`.
- Hardware: GPU 0 (L40S) shared; another user's process (jsk62, ~29.6 GiB) was resident at 0% utilisation during the runs.
- Run directories: `results/3rgs_repro/{3dgs,mcmc-mlp-epi}/{bonsai,counter}` (gitignored). Logs: `queue.log`, `queue_counter.log`.
- Launch times: counter queue 15:32 (3DGS ends 15:49, 3R-GS ends 16:07, 2026-11-18); bonsai ran earlier the same day.

## Results (ours)
Step 29999:

| Scene | Run | PSNR | SSIM | LPIPS | #GS | ATE RMSE | Mean rot. err (deg) | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| bonsai | 3DGS | 14.12 | 0.520 | 0.616 | 1,908,780 | 0.0567 | 2.025 | 834 |
| bonsai | 3R-GS | 32.28 | 0.944 | 0.127 | 1,000,000 | 0.0077 | 0.178 | 880 |
| counter | 3DGS | 21.87 | 0.784 | 0.299 | 1,691,753 | 0.0142 | 0.265 | 850 |
| counter | 3R-GS | 29.29 | 0.912 | 0.150 | 1,000,000 | 0.0039 | 0.060 | 956 |

Step 6999:

| Scene | Run | PSNR | SSIM | LPIPS | #GS | ATE RMSE | Mean rot. err (deg) | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| bonsai | 3DGS | 14.86 | 0.538 | 0.645 | 1,000,220 | 0.0521 | 2.055 | 109 |
| bonsai | 3R-GS | 28.24 | 0.875 | 0.230 | 1,000,000 | 0.0097 | 0.172 | 167 |
| counter | 3DGS | 21.14 | 0.757 | 0.377 | 896,410 | 0.0129 | 0.275 | 110 |
| counter | 3R-GS | 26.76 | 0.848 | 0.255 | 1,000,000 | 0.0045 | 0.069 | 186 |

## Paper numbers (as stated in arXiv 2504.04294)
bonsai is not in the paper's tables.

| Scene | Run | PSNR | SSIM | LPIPS | ATE | Rot. err (deg) |
|---|---|---|---|---|---|---|
| counter | 3DGS (Table 1 / 2) | 27.57 | 0.862 | 0.209 | 0.011 | 0.25 |
| counter | 3R-GS "Ours" (Table 1 / 2) | 28.80 | 0.897 | 0.157 | 0.003 | 0.05 |

## Failed / missing runs, fixes, confounds
- No failed runs. A 60-step smoke test of the unmodified code passed before the runs (train, eval, trajectory render).
- Fix before running: evo plot backend (`TkAgg` unavailable) set to `Agg`.
- Confounds: MASt3R data from the authors' archive (not regenerated); local images instead of the authors' images; gsplat 1.5.3 in the `gsplat` env, not the commit pinned by the 3rgs repo (`ec3e715`); single seed; shared GPU (times not comparable to an idle GPU); bonsai has no paper reference.
- The 3DGS baseline on bonsai (PSNR 14.12) and counter (21.87) is below the paper's counter 3DGS number (27.57); not investigated.

## Insights
To be filled by the user.
