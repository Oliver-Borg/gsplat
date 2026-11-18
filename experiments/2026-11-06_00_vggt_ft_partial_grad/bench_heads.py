import sys, time, torch
sys.path.insert(0, "/home/ojab2/work/git/vggt")
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
dev = "cuda"
w = torch.load("/home/ojab2/.cache/torch/hub/checkpoints/model.pt", map_location="cpu")
sub = lambda p: {k[len(p) + 1:]: v for k, v in w.items() if k.startswith(p + ".")}
cam = CameraHead(dim_in=2048); cam.load_state_dict(sub("camera_head")); cam.to(dev).eval()
dpt = DPTHead(dim_in=2048, output_dim=2, activation="exp", conf_activation="expp1"); dpt.load_state_dict(sub("depth_head")); dpt.to(dev).eval()
S, N, PSI = 40, 5 + 37 * 37, 5
toks = [torch.randn(1, S, N, 2048, device=dev) * 0.5 if i in (4, 11, 17, 23) else None for i in range(24)]
imgs = torch.rand(1, S, 3, 518, 518, device=dev)
def timeit(f, n=8, warm=3):
    for _ in range(warm): f()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    t = torch.tensor(ts); return f"{t.mean():8.1f} ms (std {t.std():.1f}, min {t.min():.1f})"
def cam_fwd(s):
    def f():
        with torch.no_grad(): cam([None]*23 + [toks[-1][:, :s]])
    return f
def cam_fb(s):
    def f():
        out = cam([None]*23 + [toks[-1][:, :s]])[-1]; out.sum().backward(); cam.zero_grad(set_to_none=True)
    return f
def dpt_fwd(s, chunk=8):
    def f():
        with torch.no_grad():
            for a in range(0, s, chunk):
                b = min(a + chunk, s); dpt([None if t is None else t[:, a:b] for t in toks], imgs[:, a:b], PSI, frames_chunk_size=None)
    return f
def dpt_fb(s):
    def f():
        p, c = dpt([None if t is None else t[:, :s] for t in toks], imgs[:, :s], PSI, frames_chunk_size=None)
        (p.sum() + c.sum()).backward(); dpt.zero_grad(set_to_none=True)
    return f
for p in list(cam.parameters()) + list(dpt.parameters()): p.requires_grad_(True)
print("camera params %.1fM, depth params %.1fM" % (sum(p.numel() for p in cam.parameters())/1e6, sum(p.numel() for p in dpt.parameters())/1e6))
print("camera head fwd no_grad,  40 frames:", timeit(cam_fwd(40)))
print("camera head fwd+bwd,      40 frames:", timeit(cam_fb(40)))
print("camera head fwd+bwd,       1 frame :", timeit(cam_fb(1)))
print("depth head fwd no_grad,   40 frames (chunks of 8):", timeit(dpt_fwd(40), n=4, warm=1))
print("depth head fwd+bwd,        1 frame :", timeit(dpt_fb(1)))
print("depth head fwd+bwd,        8 frames:", timeit(dpt_fb(8), n=4, warm=1))
print("peak mem GB", torch.cuda.max_memory_allocated()/1e9)
