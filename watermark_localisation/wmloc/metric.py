"""Exact task metric and mask downsampling."""
from __future__ import annotations

import numpy as np
import torch

from . import config as C


def downsample_mask(mask512: np.ndarray) -> np.ndarray:
    """512x512 0/1 -> 128x128: cell is 1 if >= 50% of its 16 pixels are 1.
    Prefer the starter kit's own function if it exists; check they agree (Step 0)."""
    m = (np.asarray(mask512) > 0).astype(np.float32)
    frac = m.reshape(C.GRID, C.CELL, C.GRID, C.CELL).mean(axis=(1, 3))
    return (frac >= 0.5).astype(np.uint8)


def miou_image(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-image mIoU over classes {0,1}; empty union counts as IoU 1 (task rule)."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    total = 0.0
    for p, g in ((~pred, ~gt), (pred, gt)):
        union = np.logical_or(p, g).sum()
        total += 1.0 if union == 0 else np.logical_and(p, g).sum() / union
    return total / 2.0


def score(preds, gts) -> float:
    return float(np.mean([miou_image(p, g) for p, g in zip(preds, gts)]))


def miou_torch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Batched version: pred, gt bool/0-1 tensors (B,128,128) -> (B,) mIoU."""
    pred = pred.bool().flatten(1)
    gt = gt.bool().flatten(1)
    out = []
    for p, g in ((~pred, ~gt), (pred, gt)):
        inter = (p & g).sum(1).float()
        union = (p | g).sum(1).float()
        out.append(torch.where(union == 0, torch.ones_like(union), inter / union.clamp(min=1)))
    return (out[0] + out[1]) / 2


SIZE_BINS = ((0.0, 0.05, "small <5%"), (0.05, 0.30, "medium 5-30%"), (0.30, 1.01, "large >30%"))


def breakdown(preds, gts) -> dict[str, tuple[int, float]]:
    """Mean mIoU by edit type and by true watermarked area -> {group: (count, mean)}.
    A change can help big rectangles and hurt small objects while the average barely moves."""
    from .data import edit_type
    groups: dict[str, list[float]] = {}
    for p, g in zip(preds, gts):
        g = np.asarray(g)
        s = miou_image(p, g)
        a = float(g.mean())
        keys = ["type " + edit_type(g)] + ["size " + name for lo, hi, name in SIZE_BINS if lo <= a < hi]
        for k in keys:
            groups.setdefault(k, []).append(s)
    return {k: (len(v), float(np.mean(v))) for k, v in sorted(groups.items())}
