# 3DGS with five sets of cameras and the same Omega point cloud (bonsai, 7000 steps)

## Question
Train the same 3DGS scene for 7000 steps with each of five camera sets (cameras fixed during training): GT, camera head
fine-tuned against GT (full head and LoRA), Omega initial prediction, COLMAP.

## Setup
- Scene: Mip-NeRF 360 `bonsai`, 40 training images (`minfarthestpose`, seed 42), 30 evaluation images, `eval_data_factor=2`,
  `data_factor=1`. Experiment `omega_cameras_7k` (group 102) in `experiment_runner.py`.
- Trainer: `simple_trainer.py mcmc`, 7000 steps, MCMC densification with `cap_max=1_000_000`, no pose optimiser, no
  camera-head training, cameras fixed. Evaluation at steps 1, 3000, 7000 (trainer's fixed eval steps). PLY saved at 7000.
- Points: the same 100K random points of the VGGT-Omega reconstruction in every arm (`sampling_mode=random`).
  Intrinsics come from each arm's own camera source.
- Evaluation: `gt_eval`; GT test cameras are aligned to the (fixed) training cameras with Umeyama and `eval_rre`/`eval_rte`
  are measured against GT. They are therefore constant over steps within an arm.

| Arm | Cameras | How built |
|---|---|---|
| 1 | GT | `choice=combined`, `camera_src=gt`, `pcd_src=vggt` (Omega): GT cameras copied into the Omega reconstruction, points aligned to the cameras (global Umeyama) |
| 2 | Camera head fine-tuned against GT, full head lr 1e-5, step 1000 | `ft_cams=full`: `vggt/make_ft_cameras.py` writes a copy of the Omega reconstruction with the cameras (poses and fov-derived intrinsics) from `../2026-11-11_camera_head_gt_smoke/full_lr1e-5_cameras.npz` |
| 3 | Camera head fine-tuned against GT, LoRA r16 lr 1e-4, step 1000 | `ft_cams=lora`, `lora16_lr1e-4_cameras.npz` |
| 4 | Omega initial prediction | plain Omega reconstruction (`choice=vggt`, `model=omega`) |
| 5 | COLMAP (same 40 images, shared camera, `default` mode) | `choice=combined`, `camera_src=colmap`, `pcd_src=vggt` |

## Launch
Reconstruction (CPU only, cached Omega and COLMAP reconstructions reused), then the trainers through a queue script that
waited for an idle GPU:
```bash
cd ~/work/git/gsplat
env -u PYTHONPATH PATH=$HOME/.conda/envs/gsplat/bin:$PATH python experiment_runner.py \
  --experiment_names omega_cameras_7k --dataset_name bonsai --dataset_collections \
  --do_reconstruct --skip_plotting --cuda_visible_devices 0
python experiments/2026-11-13_omega_cameras_7k/collect.py && python experiments/2026-11-13_omega_cameras_7k/make_figures.py
```

## Run metadata
- gsplat commit `0116ceee` (`ft_cams`, `omega_cameras_7k`); the `val_steps` of the experiment definition (listed 1, 1000, 3000, 5000, 7000 at run time,
  the trainer evaluates at 1, 3000, 7000) was corrected afterwards to `[1, 3000, 7000]` (committed with this log).
  vggt commit `34fc5bc` (contains `make_ft_cameras.py`).
- Envs: conda `gsplat` (trainer), `vggt` (reconstruction and camera variants), `PYTHONPATH` unset. Hardware: one GPU (index 0), launched 12 Nov 18:54, finished 19:09 (other users' jobs on the other GPUs).
- Run directories (`results/`, gitignored):
  - `combined_outputs/bonsai_2_n40_s42_omega_combined_gtcams_vggtpcd_amglobal_default_c0.0_p100000_random_minfarthestpose_sharedcam_i40_gteval_steps7000`
  - `vggt_outputs/bonsai_2_n40_s42_omega_c0.0_p100000_random_minfarthestpose_i40_gteval_ftcams-full_steps7000`
  - `vggt_outputs/bonsai_2_n40_s42_omega_c0.0_p100000_random_minfarthestpose_i40_gteval_ftcams-lora_steps7000`
  - `vggt_outputs/bonsai_2_n40_s42_omega_c0.0_p100000_random_minfarthestpose_i40_gteval_steps7000`
  - `combined_outputs/bonsai_2_n40_s42_omega_combined_colmapcams_vggtpcd_amglobal_default_c0.0_p100000_random_minfarthestpose_sharedcam_i40_gteval_steps7000`
- Raw numbers: `metrics.csv` (from `results/*/stats/val_step*.json`).

## Results
![metrics](figures/metrics.png)

PSNR / SSIM / LPIPS at the evaluation steps, and the camera errors (constant over steps):

| Arm | Step 1 | Step 3000 | Step 7000 | eval_rre (deg) | eval_rte |
|---|---|---|---|---|---|
| 1. GT | 10.95 / 0.246 / 0.789 | 25.42 / 0.840 / 0.245 | 27.37 / 0.886 / 0.165 | 0.0000 | 0.00000 |
| 2. Full head lr 1e-5 | 10.95 / 0.246 / 0.789 | 24.65 / 0.799 / 0.272 | 25.80 / 0.821 / 0.195 | 0.0560 | 0.00235 |
| 3. LoRA r16 lr 1e-4 | 10.95 / 0.246 / 0.789 | 24.33 / 0.777 / 0.285 | 25.05 / 0.790 / 0.209 | 0.0775 | 0.00262 |
| 4. Omega initial | 10.95 / 0.246 / 0.789 | 19.90 / 0.631 / 0.370 | 19.76 / 0.617 / 0.307 | 0.4131 | 0.01296 |
| 5. COLMAP | 10.95 / 0.246 / 0.789 | 20.88 / 0.642 / 0.297 | 20.86 / 0.631 / 0.226 | 0.5092 | 0.05524 |

Gaussian count: 100,000 at step 1, 322,497 at step 3000, 1,000,000 at step 7000 in every arm.

## Failed / missing runs, confounds
- All five runs completed; no failed or missing runs.
- Single scene, single seed, one run per arm, no determinism check.
- The camera errors `eval_rre`/`eval_rte` are properties of the fixed cameras (they do not change during training).
- Camera sets differ in more than pose accuracy: each arm uses its own intrinsics (GT: the GT camera intrinsics as copied by `combine_clouds`; heads: fov of the head; COLMAP: shared camera focal; Omega: predicted fov).
- The point cloud is the Omega cloud in every arm; in arms 1 and 5 it is aligned to the cameras with a global Umeyama sim(3)
  (`combine_clouds`, mean RRE/RTE between the COLMAP and Omega cameras after alignment 0.72 deg / 0.256), in arms 2 and 3 the
  cameras were mapped into the Omega reconstruction frame with an exact similarity fitted on the initial cameras.
- The arm 2 and 3 cameras were fine-tuned against the same GT cameras used for the evaluation (on the 40 training frames).
- Variant 2/3 camera check: scoring the written variant against GT gave rre 0.056 / rte 0.002351 (full head), against 0.0574 / 0.002353 in the smoke test.
- The three evaluation steps are the trainer's fixed ones (the experiment definition listed 1000 and 5000 too; they were never evaluated).

## Insights
To be filled by the user.
