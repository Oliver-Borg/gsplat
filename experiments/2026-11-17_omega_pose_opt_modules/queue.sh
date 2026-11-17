#!/bin/bash
# Waits for one idle GPU (no process of anyone on it, <500 MiB used, idle on two consecutive checks), then runs the two experiments
# omega_pose_opt_loss and omega_pose_opt_gtk one after the other on it. GPU=<index> skips the wait (only when the user allows it).
free_gpus() {
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | while IFS=', ' read -r idx uuid mem; do
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
    if [ "$n" -eq 0 ] && [ "$mem" -lt 500 ]; then echo $idx; fi
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
