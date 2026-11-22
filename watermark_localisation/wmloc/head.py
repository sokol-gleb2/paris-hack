"""Localisation head + IoU-oriented loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def _block(cin, cout, k=3, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2), nn.GroupNorm(8, cout), nn.GELU())


class LocHead(nn.Module):
    """Decoder feature maps (+ agreement map, + image) -> (B,128,128) logits."""

    def __init__(self, in_channels: list[int], extra_ch: int = 0, mid: int = 64,
                 image_branch: bool = False, dropout: float = 0.1):
        super().__init__()
        self.lat = nn.ModuleList([_block(c, mid, k=1) for c in in_channels])
        self.image_branch = image_branch
        img_ch = 32 if image_branch else 0
        if image_branch:   # 512 -> 128, keeps fine edges
            self.img = nn.Sequential(_block(3, 16, 3, 2), _block(16, 32, 3, 2))
        cin = mid * len(in_channels) + extra_ch + img_ch
        self.fuse = nn.Sequential(_block(cin, 128), nn.Dropout2d(dropout), _block(128, 64),
                                  nn.Conv2d(64, 1, 1))
        self.extra_ch = extra_ch

    def forward(self, feats, extra=None, image=None):
        size = (C.GRID, C.GRID)
        xs = [F.interpolate(l(f.float()), size=size, mode="bilinear", align_corners=False)
              for l, f in zip(self.lat, feats)]
        if self.extra_ch:
            xs.append(F.interpolate(extra.float(), size=size, mode="bilinear", align_corners=False))
        if self.image_branch:
            xs.append(self.img(image))
        return self.fuse(torch.cat(xs, 1))[:, 0]


# ------------------------------------------------------------ losses
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jac = 1.0 - inter / union
    if gt_sorted.numel() > 1:
        jac[1:] = jac[1:] - jac[:-1]
    return jac


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    grad = _lovasz_grad(labels[perm])
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_symmetric(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-image Lovász hinge on both classes. logits, y: (B,128,128)."""
    logits = logits.float().flatten(1)
    y = y.float().flatten(1)
    losses = [0.5 * (_lovasz_hinge_flat(l, t) + _lovasz_hinge_flat(-l, 1 - t))
              for l, t in zip(logits, y)]
    return torch.stack(losses).mean()


def seg_loss(logits: torch.Tensor, y: torch.Tensor, w_lovasz: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits.float(), y.float())
    return bce + w_lovasz * lovasz_symmetric(logits, y)
