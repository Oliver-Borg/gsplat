# Learned point sampler vs MCMC densification (bonsai, 30k steps, 3 seeds)

## Question

How does a gaussian set sampled from the full VGGT point cloud by a learned MLP (no densification) compare to normal
MCMC densification starting from 100K random VGGT points, with the gaussian limit fixed at 1M? Third arm: the same
sampler with the MLP frozen (`sampler_lr=0`).

## Setup

- Scene: Mip-NeRF 360 `bonsai`, 40 training images, VGGT reconstruction (`choice=vggt`, `sampling_mode=random`,
  `num_points_value=100000`, `gt_eval=True`), evaluation on 30 images (`max_eval_images`), `eval_data_factor=2`.
- Trainer: `examples/simple_trainer.py mcmc`, `max_steps=30000`, MCMC `cap_max=1_000_000`, eval at steps
  1, 3000, 7000, 10000, 15000, 20000, 25000, 30000. PLY exported at the last step.
- Seeds: 42, 43, 44 (seed is part of the run directory name; each seed has its own VGGT reconstruction).
- Experiment name in `experiment_runner.py`: `sampler_vs_densification` (group 99).

| Arm | Series label | Differs by |
|---|---|---|
| 1 | Densification (100K start, 1M cap) | MCMC densification, 100K random VGGT points at init, no sampler |
| 2 | Learned sampler (1M of 10M pool) | `--init_type vggt_full`, pool = full VGGT cloud (cap 10M), 1M sampled, `sampler_every=10`, `sampler_lr=1e-3`, densification disabled |
| 3 | Random sampler ablation (lr=0) | as arm 2 with `sampler_lr=0.0` (frozen MLP, Gumbel-noisy sampling) |

Sampler settings (arms 2, 3): per-point MLP, 3 hidden layers x 512, ReLU, input = positional embedding of normalised
xyz (`sampler_num_freqs=10`), one logit per point, softmax over the pool, Gumbel-top-k, straight-through mask
(sigmoid surrogate, `sampler_temperature=1.0`) multiplied into opacity. Evaluation and PLY export use the
deterministic top-k.

Pool size actually used (`sparse/points_full.npz`, `sampler_pool_size` cap was 10M): 7,169,120 points for each of
seeds 42, 43, 44.

## Launch

```bash
cd ~/work/git/gsplat
env -u PYTHONPATH PATH=$HOME/.conda/envs/gsplat/bin:$PATH python experiment_runner.py \
  --experiment_names sampler_vs_densification --dataset_name bonsai --dataset_collections \
  --do_reconstruct --skip_plotting --cuda_visible_devices 1,2
# plots
env -u PYTHONPATH PATH=$HOME/.conda/envs/gsplat/bin:$PATH python experiment_runner.py \
  --experiment_names sampler_vs_densification --dataset_name bonsai --dataset_collections --plot_only
```

## Run metadata

- gsplat: commit `634a3fad` + uncommitted changes (`examples/simple_trainer.py`, `examples/datasets/colmap.py`,
  `experiment_runner.py`, new `examples/point_sampler.py`).
- vggt: commit `8cb67c5` + uncommitted changes (`demo_colmap.py`, `reconstruct.py`, `reconstruct_args.py`,
  `plot_metrics.py`).
- Envs: conda `gsplat` (Python 3.10, trainer/runner) and `vggt` (reconstruction), shell `PYTHONPATH` unset.
- Hardware: NVIDIA L40S. Trainers on GPUs 1 and 2; the VGGT reconstruction step ran on GPU 3.
- Run directories (`results/vggt_outputs/`, gitignored), `s` in {42, 43, 44}:
  - `bonsai_2_n40_s{s}_c0.0_p100000_random_minfarthestpose_i40_gteval_steps30000`
  - `bonsai_2_n40_s{s}_c0.0_p100000_random_fullpcd_minfarthestpose_i40_gteval_sampler1000000p10000000e10_steps30000`
  - `bonsai_2_n40_s{s}_c0.0_p100000_random_fullpcd_minfarthestpose_i40_gteval_sampler1000000p10000000e10_slr0.0_steps30000`
- Plots: `latest_plots/99_sampler_vs_densification_bonsai_latest_*` (full, renders, depth_conf, table).

## Results

Mean over seeds 42, 43, 44 (PSNR dB / SSIM / LPIPS), from `mean_metrics.csv`:

| Step | 1. Densification | 2. Learned sampler | 3. Random sampler (lr=0) |
|---|---|---|---|
| 1 | 10.62 / 0.213 / 0.782 | 11.40 / 0.160 / 0.881 | 10.40 / 0.134 / 0.900 |
| 3000 | 18.22 / 0.598 / 0.460 | 17.30 / 0.584 / 0.602 | 14.35 / 0.456 / 0.695 |
| 7000 | 18.02 / 0.574 / 0.394 | 17.84 / 0.599 / 0.571 | 15.90 / 0.533 / 0.601 |
| 10000 | 17.91 / 0.565 / 0.386 | 17.55 / 0.600 / 0.605 | 16.23 / 0.542 / 0.569 |
| 15000 | 17.79 / 0.558 / 0.384 | 17.49 / 0.592 / 0.617 | 16.38 / 0.544 / 0.543 |
| 20000 | 17.74 / 0.554 / 0.384 | 17.47 / 0.591 / 0.621 | 16.48 / 0.546 / 0.530 |
| 25000 | 17.72 / 0.552 / 0.384 | 17.37 / 0.591 / 0.622 | 16.49 / 0.546 / 0.524 |
| 30000 | 17.69 / 0.551 / 0.384 | 16.68 / 0.569 / 0.644 | 16.53 / 0.545 / 0.518 |

Final PSNR (step 30000) per seed:

| Arm | s42 | s43 | s44 |
|---|---|---|---|
| 1. Densification | 17.59 | 17.60 | 17.88 |
| 2. Learned sampler | 16.69 | 18.19 | 15.15 |
| 3. Random sampler (lr=0) | 16.48 | 16.55 | 16.54 |

Learned-sampler PSNR per seed (s42 / s43 / s44) at step 7000: 18.10 / 18.20 / 17.23; at step 15000:
16.11 / 18.08 / 18.27.

Gaussian count (`num_GS`): arm 1 is 100,000 at step 1, 322,497 at step 3000, 1,000,000 from step 7000; arms 2 and 3
are 1,000,000 at every step.

Per-seed, per-step values for all metrics: `per_seed_metrics.csv`. Figure (mean line, shaded spread across seeds):
`metrics_over_steps.jpg`.

## Failed / missing runs, fixes, confounds

- All 9 runs completed; no failed or missing runs.
- Earlier, a 15k-step single-seed (42) version of the same three arms was run with the same code; its results are in
  the `..._steps15000` run directories and are not part of this table.
- Fixes applied before these runs (during the 15k-step runs): evaluation mask taken from exact top-k indices instead
  of a score threshold (a bf16-tie version gave 1,003,946 selected points instead of 1,000,000); `sampler` and
  `sampler_lr` regex params added to `plot_metrics.py` so arms are not merged when plotting.
- Code is uncommitted in both repos (see metadata).
- Single scene (bonsai), 3 seeds, no determinism check; the learned-sampler arm varies across seeds (see per-seed
  table).
- No MLP diagnostics were logged (score entropy, mask overlap between resamples, etc.); only the eval metrics above.
- Arms 2 and 3 use a pool of 7.17M points (below the 10M cap); arm 1 initialises from 100K points.

## Insights

To be filled by the user.
