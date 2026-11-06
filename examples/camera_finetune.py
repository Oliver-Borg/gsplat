"""Fine-tunes the VGGT-Omega camera head during 3DGS training (pose optimisation by fine-tuning).

The aggregator ran once in the reconstruction (`omega_infer.py`); its final-layer camera and register tokens are cached.
Every step the camera head re-predicts the cameras of all frames from those tokens, so the 3DGS loss reaches the head
through the rendered view's camera. Before `camft_delay` the frozen initial prediction is used and the head is not
trained. The Gaussians (MCMC densification) are untouched by the head.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

OMEGA_ROOT = os.environ.get("VGGT_OMEGA_ROOT", os.path.expanduser("~/work/git/vggt-omega"))
if OMEGA_ROOT not in sys.path:
    sys.path.append(OMEGA_ROOT)

from vggt_omega.models.heads import CameraHead  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

from vggt_finetune import apply_lora  # noqa: E402

OMEGA_CHECKPOINTS = {
    "512": "~/work/checkpoints/vggt-omega/vggt_omega_1b_512.pt",
    "416": "~/work/checkpoints/vggt-omega/vggt_omega_1b_416_reproduce.pt",
}


class CameraFinetuner(nn.Module):
    def __init__(self, cfg, parser, used_names: List[str], device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        assert cfg.camft_cache_file and os.path.exists(cfg.camft_cache_file), (
            f"camft_cache_file {cfg.camft_cache_file!r} missing, it is written by the VGGT-Omega reconstruction"
        )
        cache = torch.load(cfg.camft_cache_file, map_location="cpu", weights_only=False)
        assert cache.get("model") == "omega", "camft needs a raw_preds.pt written by omega_infer.py"
        assert cache["omega_checkpoint"] == cfg.camft_checkpoint, (cache["omega_checkpoint"], cfg.camft_checkpoint)
        self.H, self.W = (int(v) for v in cache["net_hw"])
        coords = np.asarray(cache["original_coords"])  # [S,6] x1,y1,x2,y2,w,h
        names = list(cache["base_image_path_list"])
        assert sorted(names) == sorted(used_names), "cached Omega frames differ from the training images"
        self.frame_of = {n: i for i, n in enumerate(names)}
        self.train_frames_list = [self.frame_of[n] for n in used_names]
        tokens = cache["camera_and_register_tokens"].float()[None].to(device)  # [1,S,T,C]
        self.register_buffer("tokens", tokens, persistent=False)
        self.S, self.patch_start, dim = tokens.shape[1], tokens.shape[2], tokens.shape[3]

        # Camera head weights from the Omega checkpoint
        path = os.path.expanduser(OMEGA_CHECKPOINTS[cfg.camft_checkpoint])
        assert os.path.exists(path), f"VGGT-Omega checkpoint {path} not found"
        try:
            state = torch.load(path, map_location="cpu", mmap=True)
        except Exception:
            state = torch.load(path, map_location="cpu")
        prefix = "camera_head."
        self.head = CameraHead(dim_in=dim)
        self.head.load_state_dict({k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)})
        del state
        self.head.to(device).eval()

        if cfg.camft_lora:
            for p in self.head.parameters():
                p.requires_grad_(False)
            self.params: List[nn.Parameter] = apply_lora(self.head, cfg.camft_lora_rank, cfg.camft_lora_alpha)
        else:
            self.params = list(self.head.parameters())
            for p in self.params:
                p.requires_grad_(True)
        print(
            f"Camera fine-tuning: lora={cfg.camft_lora} trainable params={sum(p.numel() for p in self.params) / 1e6:.2f}M"
            f" delay={cfg.camft_delay} lr={cfg.camft_lr}"
        )

        # Fixed similarity Omega world -> trainer world (the parser normalisation of the initial reconstruction)
        probe = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        out = parser.transform_world_points(probe)
        lin = (out[1:] - out[0]).T
        scale = float(np.linalg.norm(lin[:, 0]))
        self.scale = scale
        self.register_buffer("R_M", torch.from_numpy(lin / scale).float().to(device), persistent=False)
        self.register_buffer("t_M", torch.from_numpy(out[0]).float().to(device), persistent=False)

        # Trainer image sizes (w, h) per frame and the per-axis ratio to the network resolution
        sizes = coords[:, -2:].astype(np.float64)
        self.register_buffer("img_wh", torch.from_numpy(sizes).float().to(device), persistent=False)
        ratio = np.stack([sizes[:, 0] / self.W, sizes[:, 1] / self.H], axis=1)
        self.register_buffer("ratio", torch.from_numpy(ratio).float().to(device), persistent=False)

        # Frozen initial prediction
        with torch.no_grad():
            pose_enc = self._pose_enc()
            self.ref_pose_enc = pose_enc.clone()
            self.ref_c2w, self.ref_K = self._cameras(pose_enc)
        cached = cache["pose_enc"].float().to(device)
        enc_err = (pose_enc - cached).abs().max().item()
        pn = {n: i for i, n in enumerate(parser.image_names)}
        order = [pn[n] for n in names]
        pc2w = torch.from_numpy(parser.camtoworlds[order]).float().to(device)
        pK = torch.from_numpy(np.stack([parser.Ks_dict[parser.camera_ids[i]] for i in order])).float().to(device)
        pos_err = (self.ref_c2w[:, :3, 3] - pc2w[:, :3, 3]).norm(dim=-1).max().item()
        rot_err = (self.ref_c2w[:, :3, :3] - pc2w[:, :3, :3]).abs().max().item()
        k_err = (self.ref_K - pK).abs().max().item()
        print(
            f"Camera fine-tuning: init check pose_enc={enc_err:.2e} pos={pos_err:.2e} rot={rot_err:.2e} K={k_err:.2e}"
        )
        assert (
            pos_err < 1e-2 * max(1.0, parser.scene_scale) and rot_err < 1e-2 and k_err < 1e-2 * pK[:, 0, 0].max().item()
        ), ("predicted cameras do not reproduce the COLMAP cameras", pos_err, rot_err, k_err)

    def _pose_enc(self) -> Tensor:
        return self.head([self.tokens], patch_token_start=self.patch_start)[0]  # [S,9]

    def _cameras(self, pose_enc: Tensor) -> Tuple[Tensor, Tensor]:
        """Predicted cameras in the trainer frame: c2w [S,4,4] and K [S,3,3] at the trainer image resolution."""
        extr, intr = encoding_to_camera(pose_enc[None], (self.H, self.W))
        extr, intr = extr[0], intr[0]
        Rw, tw = extr[:, :3, :3], extr[:, :3, 3]
        Rc = Rw.transpose(1, 2)
        pos = -(Rc @ tw[..., None])[..., 0]
        c2w = torch.eye(4, device=pose_enc.device).repeat(self.S, 1, 1)
        c2w[:, :3, :3] = self.R_M @ Rc
        c2w[:, :3, 3] = self.scale * pos @ self.R_M.t() + self.t_M
        K = torch.zeros(self.S, 3, 3, device=pose_enc.device)
        K[:, 0, 0] = intr[:, 0, 0] * self.ratio[:, 0]
        K[:, 1, 1] = intr[:, 1, 1] * self.ratio[:, 1]
        K[:, 0, 2] = self.img_wh[:, 0] / 2
        K[:, 1, 2] = self.img_wh[:, 1] / 2
        K[:, 2, 2] = 1.0
        return c2w, K

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Cameras of all frames from the (trainable) head."""
        return self._cameras(self._pose_enc())

    @torch.no_grad()
    def current_cameras(self) -> Tuple[Tensor, Tensor]:
        return self._cameras(self._pose_enc())

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
            "rot_deg_mean": rot.mean().item(),
            "rot_deg_max": rot.max().item(),
            "pos_mean": pos.mean().item(),
            "pos_max": pos.max().item(),
            "focal_rel_mean": foc.mean().item(),
            "focal_rel_max": foc.max().item(),
        }

    def make_optimizer(self) -> torch.optim.Optimizer:
        """Adam over the trainable head parameters; it only moves them once they receive gradients (after the delay)."""
        return torch.optim.Adam(self.params, lr=self.cfg.camft_lr)

    def trainable_state_dict(self) -> Dict[str, Tensor]:
        return self.head.state_dict()
