#!/bin/bash
# Waits for one idle GPU (no process of anyone on it, <500 MiB used, idle on two consecutive checks), runs a 60-step smoke test
# of the GT-loss camera-head training on it, then the experiment (all five runs one after the other on that single GPU,
# so the step times are comparable).
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
export CUDA_VISIBLE_DEVICES=$a
PY=$HOME/.conda/envs/gsplat/bin/python
SM=${TMPDIR:-/tmp}/omega_camft_gt_smoke_$$
if [ -z "$SKIP_SMOKE" ]; then
env -u PYTHONPATH $PY examples/simple_trainer.py mcmc --data_dir ../vggt/vggt_outputs/bonsai_2_n40_s42_omega_c0.0_p100000_random_minfarthestpose \
  --data_factor 1 --eval_data_factor 2 --result-dir $SM --max_train_cameras 40 --max_steps 60 --eval_steps 1 60 --disable_viewer \
  --gt_train_data_dir ../vggt/data/360_v2/bonsai --gt_eval_data_dir ../vggt/data/360_v2/bonsai \
  --camft --camft_cache_file $(realpath ../vggt/vggt_outputs/bonsai_2_cache_n40_s42_omega_minfarthestpose/raw_preds.pt) \
  --camft_checkpoint 512 --camft_lr 1e-5 --camft_gt_loss --eval_all_frames --camft_drift_every 20
rc=$?
echo "$(date) smoke exit $rc ($SM)"
[ $rc -eq 0 ] || exit $rc
fi
env -u PYTHONPATH PATH=$HOME/.conda/envs/gsplat/bin:$PATH python experiment_runner.py \
  --experiment_names omega_camft_gt_15k --dataset_name bonsai --dataset_collections \
  --do_reconstruct --skip_plotting --cuda_visible_devices "$a"
echo "$(date) runner exit $?"
