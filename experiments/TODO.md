# To do later (user's list)

- **Counter scene for experiments 120/121** (`omega_epi_meank`, `omega_epi_intropt`): currently run on bonsai only. For counter the Omega
  reconstruction on the 210 train frames (every 8th of 240 is test) has to run on a GPU with >= 25 GiB (an L4 with 22 GiB runs out of memory in fp32,
  tested 2026-11-20; `dev-gpu-2` = `gxp-l40s-2` has 4x L40S 45 GiB), then the runner generates the track pairs (`--knn 10`, generate once before launching parallel
  runs) and runs the arms:
  `python experiment_runner.py --experiment_names omega_epi_meank omega_epi_intropt --dataset_name counter --dataset_collections --do_reconstruct --skip_plotting --cuda_visible_devices ... --wait_gpu_idle`
- **Camera head arms (full head lr 1e-5, LoRA r16 lr 1e-4)** were dropped from experiments 120/121 because of their training time (about 2-3 h per run
  on L40S at 255 frames); re-add them later.
- Open "To investigate" item in `2026-11-18_01_3rgs_reproduction`: our 3DGS baseline numbers vs the paper's reported 3DGS numbers.
