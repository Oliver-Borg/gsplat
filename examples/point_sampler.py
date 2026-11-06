"""Learned sampler that selects a subset of a fixed point pool to be the active Gaussians."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch import Tensor


def positional_embedding(x: Tensor, num_freqs: int) -> Tensor:
    """NeRF-style embedding: [x, sin(2^k pi x), cos(2^k pi x)] -> [N, 3 + 6 * num_freqs]."""
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * math.pi
    xf = x[..., None, :] * freqs[:, None]  # [N, F, 3]
    return torch.cat([x, torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)], dim=-1)


def gumbel_topk_mask(
    scores: Tensor, k: int, temperature: float = 1.0, noise: bool = True, u: Optional[Tensor] = None
) -> Tensor:
    """Gumbel-top-k (weighted sampling without replacement) as a straight-through mask [n].

    Forward: exactly k ones (also when scores tie). Backward: gradient of sigmoid((perturbed - kth) / temperature).
    `u` optionally fixes the uniform noise (e.g. for repeatable evaluation); noise=False takes the plain top-k.
    """
    n = scores.shape[0]
    if k >= n:
        return torch.ones_like(scores)
    perturbed = scores
    if noise:
        if u is None:
            u = torch.rand_like(scores)
        u = u.clamp(1e-10, 1 - 1e-10)
        perturbed = scores - torch.log(-torch.log(u))
    top = torch.topk(perturbed.detach(), k)
    kth = top.values[-1]
    hard = torch.zeros_like(scores)
    hard[top.indices] = 1.0  # exactly k, even when scores tie
    soft = torch.sigmoid((perturbed - kth) / temperature)
    return hard + soft - soft.detach()


class PointSampler(nn.Module):
    """Per-point MLP f(PE(xyz)) -> logit, with differentiable weighted sampling over a pool.

    The pool coordinates are normalised to [-1, 1] by its bounding box before the embedding. Sampling is
    Gumbel-top-k, i.e. weighted sampling without replacement from softmax(logits). The returned mask is hard
    in the forward pass and uses a sigmoid relaxation around the k-th largest perturbed score in the
    backward pass (straight-through).
    """

    def __init__(
        self,
        points: Tensor,
        num_freqs: int = 10,
        hidden_dim: int = 512,
        num_hidden_layers: int = 3,
        temperature: float = 1.0,
        chunk_size: int = 1 << 20,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.chunk_size = chunk_size
        lo, hi = points.min(0).values, points.max(0).values
        self.register_buffer("centre", (lo + hi) / 2)
        self.register_buffer("half_extent", ((hi - lo) / 2).max().clamp_min(1e-8))
        self.register_buffer("points", points.clone())

        in_dim = 3 + 6 * num_freqs
        layers: list[nn.Module] = []
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def _chunk_logits(self, x: Tensor) -> Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.mlp(positional_embedding(x, self.num_freqs)).squeeze(-1).float()

    def logits(self) -> Tensor:
        """Evaluated in chunks; activations are recomputed in backward so memory is independent of the pool."""
        x = (self.points - self.centre) / self.half_extent
        out = []
        for chunk in x.split(self.chunk_size):
            if torch.is_grad_enabled():
                out.append(checkpoint(self._chunk_logits, chunk, use_reentrant=False))
            else:
                out.append(self._chunk_logits(chunk))
        return torch.cat(out)  # [n]

    def forward(self, k: int, noise: bool = True) -> Tuple[Tensor, Tensor]:
        """Returns (mask [n], log_probs [n]). With noise=False the top-k of the scores is taken."""
        log_probs = torch.log_softmax(self.logits(), dim=0)
        return gumbel_topk_mask(log_probs, k, self.temperature, noise), log_probs
