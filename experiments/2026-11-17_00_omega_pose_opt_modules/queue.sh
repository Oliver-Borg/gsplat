#!/bin/bash
# Waits for a GPU at 0% utilisation (on two consecutive checks) with at least MINFREE MiB (default 6000) of free memory; sharing the GPU with
# other users' idle processes is allowed by the user. Then runs omega_pose_opt_loss and omega_pose_opt_gtk one after the other on it.
# GPU=<index> skips the wait. Earlier runs peaked at 1.5 to 2.3 GiB (torch max allocated) for the fixed / pose-module arms.
MINFREE=${MINFREE:-6000}
free_gpus() {
  nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=', ' read -r idx tot used util; do
    if [ "$util" -eq 0 ] && [ $((tot - used)) -ge "$MINFREE" ]; then echo $idx; fi
  done
}
prev=""
[ -n "$GPU" ] && a=$GPU
while [ -z "$GPU" ]; do
  a=$(free_gpus | head -1)
  if [ -n "$a" ] && [ "$a" = "$prev" ]; then break; fi
  prev="$a"; sleep 60
done
echo "$(date) launching on GPU $a"
cd ~/work/git/gsplat
env -u PYTHONPATH PATH=$HOME/.conda/envs/gsplat/bin:$PATH python experiment_runner.py \
  --experiment_names omega_pose_opt_loss omega_pose_opt_gtk --dataset_name bonsai --dataset_collections \
  --do_reconstruct --skip_plotting --cuda_visible_devices "$a"
echo "$(date) runner exit $?"
