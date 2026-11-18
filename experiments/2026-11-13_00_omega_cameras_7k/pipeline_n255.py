"""n=255 pipeline: (1) Omega reconstruction on an idle GPU, (2) camera head fine-tuning on CPU (full, LoRA), (3) camera variants,
(4) the 3DGS runs through experiment_runner on an idle GPU (at most one GPU at a time)."""
import os, subprocess, sys, time, shutil
sys.path.insert(0, os.path.expanduser("~/work/git/gsplat"))
os.chdir(os.path.expanduser("~/work/git/gsplat"))
import experiment_runner as er
ReconstructArgs = er.ReconstructArgs

def log(*a): print(time.strftime("%a %d %b %H:%M:%S"), *a, flush=True)

def free_gpu():
    q = subprocess.run("nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip().splitlines()
    apps = subprocess.run("nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader", shell=True, capture_output=True, text=True).stdout
    for line in q:
        idx, uuid, mem = [x.strip() for x in line.split(",")]
        if uuid not in apps and int(mem) < 500:
            return idx
    return None

def wait_gpu():
    prev = None
    while True:
        g = free_gpu()
        if g is not None and g == prev:
            return g
        prev = g
        time.sleep(60)

cfgs = {c.series_label_override.split(".")[0]: c for c in er.experiment_dict["omega_cameras_7k_full"].get_configs("bonsai")}
omega = cfgs["4"]
os.environ.pop("PYTHONPATH", None)

g = wait_gpu(); log("Omega reconstruction on GPU", g)
os.environ["CUDA_VISIBLE_DEVICES"] = g
rc = omega.reconstruct(); log("omega reconstruct rc", rc, "reconstructed", omega.is_reconstructed)
os.environ.pop("CUDA_VISIBLE_DEVICES")
if not omega.is_reconstructed:
    log("Omega reconstruction failed, stopping"); sys.exit(1)

cache = os.path.abspath(os.path.join("../vggt", ReconstructArgs(**omega._reconstruct_args).cache_dir, "raw_preds.pt"))
data = os.path.abspath(omega.data_dir)
out = os.path.join(er._FT_DIR, "n255"); os.makedirs(out, exist_ok=True)
py = os.path.expanduser("~/.conda/envs/gsplat/bin/python")
base = [py, "examples/camera_head_gt_smoke.py", "--device", "cpu", "--cache_file", cache, "--data_dir", data,
        "--gt_dir", "../vggt/data/360_v2/bonsai", "--steps", "1000", "--snapshots", "200", "1000"]
env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
for name, extra in (("full_lr1e-5", ["--lr", "1e-5"]), ("lora16_lr1e-4", ["--lora", "--lr", "1e-4"])):
    log("fine-tuning", name, "on CPU")
    with open(os.path.join(out, name + ".log"), "w") as f:
        rc = subprocess.run(base + extra + ["--out", os.path.join(out, name + ".jsonl")], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    log(name, "rc", rc)
    if rc:
        sys.exit(1)

for k in ("1", "2", "3", "5"):
    log("reconstruct arm", k, "rc", cfgs[k].reconstruct(), "reconstructed", cfgs[k].is_reconstructed)

g = wait_gpu(); log("3DGS runs on GPU", g)
rc = subprocess.run([py, "experiment_runner.py", "--experiment_names", "omega_cameras_7k_full", "--dataset_name", "bonsai",
                     "--dataset_collections", "--do_reconstruct", "--skip_plotting", "--cuda_visible_devices", g]).returncode
log("runner exit", rc)
