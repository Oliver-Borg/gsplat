"""Fine-tunes the VGGT camera and depth (and optionally point) heads on top of cached backbone tokens.

The aggregator runs once; its tokens are cached. Every training step the heads are re-evaluated on all frames, the
depth map is unprojected with the predicted cameras to get the Gaussian means, and the predicted cameras are used
to render, so the 3DGS loss reaches the heads through both the means and the cameras.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

_VGGT_ROOT = os.path.abspath(
    os.environ.get("VGGT_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "vggt"))
)
if _VGGT_ROOT not in sys.path:
    sys.path.insert(0, _VGGT_ROOT)

# The VGGT class itself needs einops (track head), so the parts are assembled here instead.
from vggt.heads.camera_head import CameraHead  # noqa: E402
from vggt.heads.dpt_head import DPTHead  # noqa: E402
from vggt.models.aggregator import Aggregator  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

VGGT_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_RES = 518
DPT_LAYERS = (4, 11, 17, 23)  # intermediate layers read by the DPT heads, the last one also feeds the camera head
NUM_LAYERS = 24


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        w = base.weight
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=w.device, dtype=w.dtype))
        self.scale = alpha / rank

    @property
    def in_features(self) -> int:  # read by attention modules that reach into their projections
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


class LoRAConv2d(nn.Module):
    """Low-rank update of a convolution: a kxk conv down to `rank` channels followed by a 1x1 conv back up."""

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        assert base.groups == 1
        self.base = base
        self.down = nn.Conv2d(
            base.in_channels,
            rank,
            base.kernel_size,
            base.stride,
            base.padding,
            base.dilation,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.up(self.down(x)) * self.scale


def apply_lora(module: nn.Module, rank: int, alpha: float) -> List[nn.Parameter]:
    """Freezes `module` and wraps every Linear / Conv2d with a LoRA update. Returns the trainable parameters."""
    for p in module.parameters():
        p.requires_grad_(False)

    def wrap(parent: nn.Module):
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, rank, alpha))
            elif isinstance(child, nn.Conv2d):
                setattr(parent, name, LoRAConv2d(child, rank, alpha))
            else:
                wrap(child)

    wrap(module)
    params: List[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, LoRALinear):
            params += [m.A, m.B]
        elif isinstance(m, LoRAConv2d):
            params += [m.down.weight, m.up.weight]
    for p in params:
        p.requires_grad_(True)
    return params


class StepProfiler:
    """Per-section GPU timings. `mark(name)` closes the section `name`, which started at the previous mark."""

    def __init__(self, enabled: bool, warmup: int = 20):
        self.enabled = enabled
        self.warmup = warmup
        self.step = 0
        self.events: List[Tuple[str, torch.cuda.Event]] = []
        self.sums: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self.last: Dict[str, float] = {}
        self.peak_mem_gb = 0.0

    def mark(self, name: str):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.append((name, ev))

    def end_step(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.last = {}
        for (_, a), (name, b) in zip(self.events[:-1], self.events[1:]):
            ms = a.elapsed_time(b)
            self.last[name] = self.last.get(name, 0.0) + ms
        if self.step >= self.warmup:
            for name, ms in self.last.items():
                self.sums[name] += ms
                self.counts[name] += 1
            self.peak_mem_gb = max(self.peak_mem_gb, torch.cuda.max_memory_allocated() / 1024**3)
        self.events = []
        self.step += 1

    def summary(self) -> Dict[str, float]:
        out = {f"{k}_ms": self.sums[k] / max(self.counts[k], 1) for k in self.sums}
        out["step_total_ms"] = sum(out.values())
        out["peak_mem_gb"] = self.peak_mem_gb
        out["steps_measured"] = max(self.counts.values()) if self.counts else 0
        return out

    def dump(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)


class VGGTFinetuner(nn.Module):
    def __init__(self, cfg, parser, used_names: List[str], device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        assert cfg.vggt_cache_file and os.path.exists(cfg.vggt_cache_file), (
            f"vggt_cache_file {cfg.vggt_cache_file!r} missing, it is written by the VGGT reconstruction"
        )
        heads = [h for h in cfg.vggt_heads.split(",") if h]
        assert set(heads) <= {"camera", "depth", "point"} and "camera" in heads and "depth" in heads, heads
        self.use_point_head = "point" in heads
        if self.use_point_head:
            assert cfg.vggt_means_source == "point", "the point head only gets gradients with vggt_means_source=point"
        if cfg.vggt_means_source == "point":
            assert self.use_point_head, "vggt_means_source=point needs 'point' in vggt_heads"

        cache = torch.load(cfg.vggt_cache_file, map_location="cpu", weights_only=False)
        self.register_buffer("images", cache["images"].float()[None].to(device), persistent=False)  # [1,S,3,H,W]
        self.S, _, self.H, self.W = self.images.shape[1:]
        assert self.H == VGGT_RES and self.W == VGGT_RES, self.images.shape
        masks = np.asarray(cache["masks"])[..., 0] == 1  # [S,H,W]
        coords = np.asarray(cache["original_coords"])  # [S,6] x1,y1,x2,y2,w,h
        names = list(cache["base_image_path_list"])
        assert sorted(names) == sorted(used_names), "cached VGGT frames differ from the training images"
        self.frame_of = {n: i for i, n in enumerate(names)}
        self.train_frames = torch.tensor([self.frame_of[n] for n in used_names], device=device)

        # Model: frozen backbone (run once), trainable heads
        weights = torch.hub.load_state_dict_from_url(VGGT_WEIGHTS_URL, map_location="cpu")

        def sub(prefix: str) -> Dict[str, Tensor]:
            return {k[len(prefix) + 1 :]: v for k, v in weights.items() if k.startswith(prefix + ".")}

        aggregator = Aggregator(img_size=VGGT_RES, patch_size=14, embed_dim=1024)
        aggregator.load_state_dict(sub("aggregator"))
        aggregator = aggregator.to(device).eval()
        self.camera_head = CameraHead(dim_in=2048)
        self.camera_head.load_state_dict(sub("camera_head"))
        self.depth_head = DPTHead(dim_in=2048, output_dim=2, activation="exp", conf_activation="expp1")
        self.depth_head.load_state_dict(sub("depth_head"))
        self.point_head = None
        if self.use_point_head:
            self.point_head = DPTHead(dim_in=2048, output_dim=4, activation="inv_log", conf_activation="expp1")
            self.point_head.load_state_dict(sub("point_head"))
        del weights
        self.camera_head.to(device).eval()
        self.depth_head.to(device).eval()
        if self.point_head is not None:
            self.point_head.to(device).eval()

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            tokens, self.patch_start_idx = aggregator(self.images)
        # Only the layers the heads read are kept
        self.tokens: List[Optional[Tensor]] = [
            tokens[i].detach() if i in DPT_LAYERS else None for i in range(NUM_LAYERS)
        ]
        del tokens, aggregator
        torch.cuda.empty_cache()

        # Trainable parameters
        trainable_heads = [self.camera_head, self.depth_head] + ([self.point_head] if self.point_head else [])
        self.params: List[nn.Parameter] = []
        for head in trainable_heads:
            if (head is self.camera_head and cfg.vggt_freeze_camera) or (
                head is self.depth_head and cfg.vggt_freeze_depth
            ):
                for p in head.parameters():
                    p.requires_grad_(False)
                continue
            if cfg.vggt_lora:
                self.params += apply_lora(head, cfg.vggt_lora_rank, cfg.vggt_lora_alpha)
            else:
                for p in head.parameters():
                    p.requires_grad_(True)
                self.params += list(head.parameters())
        print(
            f"VGGT fine-tuning: heads={heads} lora={cfg.vggt_lora} trainable params="
            f"{sum(p.numel() for p in self.params) / 1e6:.2f}M"
        )

        # Fixed similarity VGGT world -> trainer world (the parser normalisation of the initial reconstruction)
        probe = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        out = parser.transform_world_points(probe)
        lin = (out[1:] - out[0]).T  # columns = images of the basis vectors
        scale = float(np.linalg.norm(lin[:, 0]))
        self.scale = scale
        self.register_buffer("R_M", torch.from_numpy(lin / scale).float().to(device), persistent=False)
        self.register_buffer("t_M", torch.from_numpy(out[0]).float().to(device), persistent=False)

        # Image sizes of the trainer images, per frame, and the matching 518px -> original resolution ratio
        sizes = coords[:, -2:].astype(np.float64)  # (w, h)
        self.register_buffer("img_wh", torch.from_numpy(sizes).float().to(device), persistent=False)
        self.register_buffer("ratio", torch.from_numpy(sizes.max(1) / VGGT_RES).float().to(device), persistent=False)

        # Reference (frozen) predictions, the pixel pool and the initial point cloud
        with torch.no_grad():
            ref = self._predict(grad=False)
        valid = torch.from_numpy(masks).to(device).reshape(-1) & torch.isfinite(ref["depth"].reshape(-1))
        valid &= ref["depth"].reshape(-1) > 1e-6
        idx = valid.nonzero().squeeze(-1)
        if cfg.vggt_pool_size > 0 and len(idx) > cfg.vggt_pool_size:
            g = torch.Generator(device="cpu").manual_seed(42)
            keep = torch.randperm(len(idx), generator=g)[: cfg.vggt_pool_size].sort().values.to(device)
            idx = idx[keep]
        self.valid_idx = idx
        self.num_candidates = len(idx)
        self.ref_depth = ref["depth"].reshape(-1)[idx].clone()
        self.ref_pose_enc = ref["pose_enc"].clone()
        self.rgb = (self.images[0].permute(0, 2, 3, 1).reshape(-1, 3)[idx] * 255).round().clamp(0, 255)
        print(f"VGGT fine-tuning: {self.num_candidates} candidate pixels from {self.S} frames")

        # Sanity check: the initial predicted cameras reproduce the parser's cameras (same frame and image size)
        with torch.no_grad():
            c2w, K = self._cameras(ref["pose_enc"])
        self.ref_c2w, self.ref_K = c2w.clone(), K.clone()
        pn = {n: i for i, n in enumerate(parser.image_names)}
        order = [pn[n] for n in names]
        ref_c2w = torch.from_numpy(parser.camtoworlds[order]).float().to(device)
        ref_K = torch.from_numpy(np.stack([parser.Ks_dict[parser.camera_ids[i]] for i in order])).float().to(device)
        pos_err = (c2w[:, :3, 3] - ref_c2w[:, :3, 3]).norm(dim=-1).max().item()
        rot_err = (c2w[:, :3, :3] - ref_c2w[:, :3, :3]).abs().max().item()
        k_err = (K - ref_K).abs().max().item()
        print(f"VGGT fine-tuning: init camera check pos={pos_err:.2e} rot={rot_err:.2e} K={k_err:.2e}")
        # The reconstruction ran the heads on bf16 activations, these run in fp32: ~0.2% differences are expected
        assert (
            pos_err < 1e-2 * max(1.0, parser.scene_scale)
            and rot_err < 1e-2
            and k_err < 1e-2 * ref_K[:, 0, 0].max().item()
        ), (
            "predicted cameras do not reproduce the COLMAP cameras",
            pos_err,
            rot_err,
            k_err,
        )
        self.initial_points = ref["means"].reshape(-1, 3)[idx]

        # Partial-gradient mode: per-frame slices of the (sorted) candidate pool and the cached no-grad outputs
        self.train_frames_list = [self.frame_of[n] for n in used_names]
        self.partial = cfg.vggt_partial_grad
        if self.partial:
            assert cfg.vggt_means_source == "depth", "vggt_partial_grad is implemented for vggt_means_source=depth"
            assert cfg.batch_size == 1, "vggt_partial_grad differentiates the single frame of the rendered view"
            hw = self.H * self.W
            self.frame_offsets = torch.searchsorted(idx, torch.arange(self.S + 1, device=device) * hw).tolist()
            self.c_means = self.initial_points.clone()
            self.c_conf = ref["conf"].reshape(-1)[idx].clone()
            self.c_depth = ref["depth"].reshape(-1)[idx].clone()
            self.train_calls = 0

    # ---- prediction -------------------------------------------------------------------------------------------

    def _amp(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.vggt_head_amp)

    def _dpt(self, head: DPTHead, grad: bool, frames: Optional[Tuple[int, int]] = None) -> Tuple[Tensor, Tensor]:
        preds, confs = [], []
        chunk = self.cfg.vggt_frame_chunk
        lo, hi = frames if frames is not None else (0, self.S)
        for a in range(lo, hi, chunk):
            b = min(a + chunk, hi)

            def run(a=a, b=b):
                toks = [None if t is None else t[:, a:b].float() for t in self.tokens]
                with self._amp():
                    p, c = head(toks, self.images[:, a:b], self.patch_start_idx, frames_chunk_size=None)
                return p.float(), c.float()

            if grad and self.cfg.vggt_checkpoint and hi - lo > 1:
                p, c = checkpoint(run, use_reentrant=False)
            else:
                p, c = run()
            preds.append(p)
            confs.append(c)
        return torch.cat(preds, 1), torch.cat(confs, 1)  # [1,S,H,W,C], [1,S,H,W]

    def _cameras(self, pose_enc: Tensor) -> Tuple[Tensor, Tensor]:
        """Predicted cameras in the trainer frame: c2w [S,4,4] and K [S,3,3] at the trainer image resolution."""
        extr, intr = pose_encoding_to_extri_intri(pose_enc[None], (VGGT_RES, VGGT_RES))
        extr, intr = extr[0], intr[0]
        Rw, tw = extr[:, :3, :3], extr[:, :3, 3]
        Rc = Rw.transpose(1, 2)
        pos = -(Rc @ tw[..., None])[..., 0]
        c2w = torch.eye(4, device=pose_enc.device).repeat(self.S, 1, 1)
        c2w[:, :3, :3] = self.R_M @ Rc
        c2w[:, :3, 3] = self.scale * pos @ self.R_M.t() + self.t_M
        K = torch.zeros(self.S, 3, 3, device=pose_enc.device)
        K[:, 0, 0] = intr[:, 0, 0] * self.ratio
        K[:, 1, 1] = intr[:, 1, 1] * self.ratio
        K[:, 0, 2] = self.img_wh[:, 0] / 2
        K[:, 1, 2] = self.img_wh[:, 1] / 2
        K[:, 2, 2] = 1.0
        return c2w, K

    def _unproject(self, depth: Tensor, pose_enc: Tensor) -> Tensor:
        """depth [S,H,W] -> world points [S,H,W,3] in the trainer frame (VGGT's integer pixel grid)."""
        extr, intr = pose_encoding_to_extri_intri(pose_enc[None], (VGGT_RES, VGGT_RES))
        extr, intr = extr[0], intr[0]
        v, u = torch.meshgrid(
            torch.arange(self.H, device=depth.device, dtype=depth.dtype),
            torch.arange(self.W, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        fx, fy = intr[:, 0, 0][:, None, None], intr[:, 1, 1][:, None, None]
        cx, cy = intr[:, 0, 2][:, None, None], intr[:, 1, 2][:, None, None]
        cam = torch.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dim=-1)  # [S,H,W,3]
        R, t = extr[:, :3, :3], extr[:, :3, 3]
        world = torch.einsum("shwj,sji->shwi", cam - t[:, None, None], R)  # R^T (cam - t)
        return self.scale * world @ self.R_M.t() + self.t_M

    def _pose_enc(self, toks) -> Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.vggt_freeze_camera):
            return self.camera_head(toks)[-1][0]  # [S,9]

    def _predict(self, grad: bool) -> Dict[str, Tensor]:
        toks = [None] * (NUM_LAYERS - 1) + [self.tokens[-1].float()]
        pose_enc = self._pose_enc(toks)
        depth_p, conf = self._dpt(self.depth_head, grad and not self.cfg.vggt_freeze_depth)
        depth, conf = depth_p[0, ..., 0], conf[0]  # [S,H,W]
        out = {"pose_enc": pose_enc, "depth": depth, "conf": conf}
        if self.cfg.vggt_means_source == "point":
            pts, pconf = self._dpt(self.point_head, grad)
            # Points are predicted in the VGGT frame, like the unprojected depth
            out["means"] = self.scale * pts[0, ..., :3] @ self.R_M.t() + self.t_M
            out["conf"] = pconf[0]
        else:
            out["means"] = self._unproject(depth, pose_enc)
        return out

    @torch.no_grad()
    def current_cameras(self) -> Tuple[Tensor, Tensor]:
        """c2w [S,4,4] and K [S,3,3] of the current camera head, in the trainer frame."""
        toks = [None] * (NUM_LAYERS - 1) + [self.tokens[-1].float()]
        return self._cameras(self.camera_head(toks)[-1][0])

    @torch.no_grad()
    def _refresh_cache(self):
        o = self._predict(grad=False)
        idx = self.valid_idx
        self.c_means = o["means"].reshape(-1, 3)[idx].clone()
        self.c_conf = o["conf"].reshape(-1)[idx].clone()
        self.c_depth = o["depth"].reshape(-1)[idx].clone()

    def _forward_partial(self, frame: Optional[int]) -> Dict[str, Tensor]:
        """Camera head on all frames with gradient, depth head with gradient on `frame` only. The other frames'
        means/confidences come from the cache (initial prediction, replaced by a frame's latest output whenever it
        was the gradient frame, and by a full no-grad re-evaluation every `vggt_refresh_every` steps if > 0)."""
        toks = [None] * (NUM_LAYERS - 1) + [self.tokens[-1].float()]
        training = frame is not None and torch.is_grad_enabled()
        if not training:
            with torch.no_grad():
                pose_enc = self.camera_head(toks)[-1][0]
            c2w, K = self._cameras(pose_enc)
            return {"means": self.c_means, "conf": self.c_conf, "depth": self.c_depth, "pose_enc": pose_enc,
                    "c2w": c2w, "K": K, "lo": 0, "hi": self.num_candidates}
        self.train_calls += 1
        if self.cfg.vggt_refresh_every > 0 and self.train_calls % self.cfg.vggt_refresh_every == 0:
            self._refresh_cache()
        pose_enc = self._pose_enc(toks)  # [S,9], all frames
        c2w, K = self._cameras(pose_enc)
        depth_p, conf = self._dpt(self.depth_head, not self.cfg.vggt_freeze_depth, (frame, frame + 1))
        depth, conf = depth_p[0, ..., 0], conf[0]  # [1,H,W]
        means = self._unproject(depth, pose_enc[frame : frame + 1])  # [1,H,W,3]
        lo, hi = self.frame_offsets[frame], self.frame_offsets[frame + 1]
        loc = self.valid_idx[lo:hi] - frame * self.H * self.W
        new_means, new_conf, new_depth = means.reshape(-1, 3)[loc], conf.reshape(-1)[loc], depth.reshape(-1)[loc]
        out = {
            "means": torch.cat([self.c_means[:lo], new_means, self.c_means[hi:]]),
            "conf": torch.cat([self.c_conf[:lo], new_conf, self.c_conf[hi:]]),
            "depth": torch.cat([self.c_depth[:lo], new_depth, self.c_depth[hi:]]),
            "pose_enc": pose_enc, "c2w": c2w, "K": K, "lo": lo, "hi": hi,
        }
        with torch.no_grad():
            self.c_means[lo:hi], self.c_conf[lo:hi], self.c_depth[lo:hi] = new_means, new_conf, new_depth
        return out

    @torch.no_grad()
    def drift(self) -> Dict[str, float]:
        """Change of the predicted cameras (all frames) relative to the initial prediction, trainer frame."""
        c2w, K = self.current_cameras()
        Rrel = c2w[:, :3, :3] @ self.ref_c2w[:, :3, :3].transpose(1, 2)
        cos = ((Rrel.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)
        rot = torch.rad2deg(torch.acos(cos))
        pos = (c2w[:, :3, 3] - self.ref_c2w[:, :3, 3]).norm(dim=-1)
        foc = (K[:, 0, 0] / self.ref_K[:, 0, 0] - 1).abs()
        return {
            "rot_deg_mean": rot.mean().item(), "rot_deg_max": rot.max().item(),
            "pos_mean": pos.mean().item(), "pos_max": pos.max().item(),
            "focal_rel_mean": foc.mean().item(), "focal_rel_max": foc.max().item(),
        }

    def forward(self, frame: Optional[int] = None) -> Dict[str, Tensor]:
        """One evaluation of the heads on all frames, restricted to the candidate pixels. In partial-gradient mode
        `frame` is the frame whose depth gets gradients (None: evaluation from the cache)."""
        if self.partial:
            return self._forward_partial(frame)
        o = self._predict(grad=torch.is_grad_enabled())
        c2w, K = self._cameras(o["pose_enc"])
        idx = self.valid_idx
        conf = o["conf"].reshape(-1)[idx]
        return {
            "means": o["means"].reshape(-1, 3)[idx],
            "conf": conf,
            "depth": o["depth"].reshape(-1)[idx],
            "pose_enc": o["pose_enc"],
            "c2w": c2w,
            "K": K,
        }

    # ---- sampling scores and extra losses -----------------------------------------------------------------------

    def sampler_scores(self, out: Dict[str, Tensor]) -> Tensor:
        """Per-candidate scores for the Gumbel-top-k sampling; conf-based scores keep the gradient to the heads."""
        if self.cfg.vggt_conf_mode in ("sampler", "both"):
            return torch.log(out["conf"].clamp_min(1e-6))
        return torch.zeros_like(out["conf"])

    def extra_losses(self, out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Optional terms, all relative to the frozen initial prediction. Empty with the default weights of 0."""
        cfg = self.cfg
        losses: Dict[str, Tensor] = {}
        lo, hi = out.get("lo", 0), out.get("hi", len(self.ref_depth))  # only the gradient frame in partial mode
        if cfg.vggt_reg_depth > 0 or cfg.vggt_conf_mode in ("loss", "both"):
            rel_err = (out["depth"][lo:hi] - self.ref_depth[lo:hi]).abs() / self.ref_depth[lo:hi]
            if cfg.vggt_reg_depth > 0:
                losses["reg_depth"] = cfg.vggt_reg_depth * rel_err.mean()
            if cfg.vggt_conf_mode in ("loss", "both"):
                conf = out["conf"][lo:hi].clamp_min(1.0)
                # VGGT training form: gamma * err * conf - alpha * log(conf)
                losses["conf"] = cfg.vggt_conf_loss_weight * (
                    cfg.vggt_conf_gamma * rel_err * conf - cfg.vggt_conf_alpha * torch.log(conf)
                ).mean()
        if cfg.vggt_reg_pose > 0:
            losses["reg_pose"] = cfg.vggt_reg_pose * (out["pose_enc"] - self.ref_pose_enc).abs().mean()
        return losses

    def make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.params, lr=self.cfg.vggt_lr)

    def trainable_state_dict(self) -> Dict[str, Dict[str, Tensor]]:
        sd = {"camera_head": self.camera_head.state_dict(), "depth_head": self.depth_head.state_dict()}
        if self.point_head is not None:
            sd["point_head"] = self.point_head.state_dict()
        if self.partial:
            sd["partial_cache"] = {"means": self.c_means, "conf": self.c_conf, "depth": self.c_depth}
        return sd
