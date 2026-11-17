#!/bin/bash
# Waits for one idle GPU (no process of anyone on it, <500 MiB used, idle on two consecutive checks), runs the pose-MLP smoke test
# on it (GPU timing, 1000 steps), then the omega_camft_gt_15k experiment: the runner skips arms 1 to 5 (results exist) and
# runs arm 6 (pose residual MLP with GT loss).
free_gpus() {
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | while IFS=', ' read -r idx uuid mem; do
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
    if [ "$n" -eq 0 ] && [ "$mem" -lt 500 ]; then echo $idx; fi
  done
}
prev=""
[ -n "$GPU" ] && a=$GPU   # GPU=<index> skips the wait (explicitly allowed by the user)
while [ -z "$GPU" ]; do
  a=$(free_gpus | head -1)
  if [ -n "$a" ] && [ "$a" = "$prev" ]; then break; fi
  prev="$a"; sleep 60
done
echo "$(date) launching on GPU $a"
cd ~/work/git/gsplat
S=experiments/2026-11-11_camera_head_gt_smoke
CUDA_VISIBLE_DEVICES=$a env -u PYTHONPATH $HOME/.conda/envs/gsplat/bin/python examples/camera_head_gt_smoke.py --device cuda \
  --cache_file ../vggt/vggt_outputs/bonsai_2_cache_n40_s42_omega_minfarthestpose/raw_preds.pt \
  --data_dir ../vggt/vggt_outputs/bonsai_2_n40_s42_omega_c0.0_p100000_random_minfarthestpose --gt_dir ../vggt/data/360_v2/bonsai \
  --steps 1000 --snapshots 200 1000 --mlp --lr 1e-4 --out $S/gpu_mlp.jsonl > $S/gpu_mlp.log 2>&1
echo "$(date) gpu smoke exit $?"
env -u PYTHONPATH PATH=$HOME/.conda/envs/gsplat/bin:$PATH python experiment_runner.py \
  --experiment_names omega_camft_gt_15k --dataset_name bonsai --dataset_collections \
  --do_reconstruct --skip_plotting --cuda_visible_devices "$a"
echo "$(date) runner exit $?"
