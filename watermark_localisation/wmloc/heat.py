"""Known-message heatmaps: decode overlapping crops and score how well each matches the message."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C


def crop_boxes(size: int, overlap: float = 0.5) -> list[tuple[int, int]]:
    """Top-left corners (y, x) of size x size crops covering the image, overlapping by `overlap`."""
    assert size % C.CELL == 0 and size <= C.IMG, size
    step = max(C.CELL, int(size * (1 - overlap)) // C.CELL * C.CELL)
    pos = list(range(0, C.IMG - size + 1, step))
    if pos[-1] != C.IMG - size:
        pos.append(C.IMG - size)
    return [(y, x) for y in pos for x in pos]


def match_gap(scores: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """scores (B,n): decoder scores, > 0 means bit 1. bits (n,): the known message. -> (B,) gaps.

    match = mean over bits of sign * s, with sign = +1 for bit 1 and -1 for bit 0, and
    s = tanh(score / 4) (bounded, so a few huge scores do not dominate).
    Over ALL shuffles of the message, every sign meets every score equally often, so the
    average shuffled match is exactly mean(sign) * mean(s): no sampling needed.
    gap = match - average shuffled match; about 0 means "no better than a reordered message".
    """
    sign = bits.float().to(scores.device) * 2 - 1
    s = torch.tanh(scores.float() / 4.0)
    return (s * sign).mean(1) - s.mean(1) * sign.mean()


@torch.no_grad()
def crop_gaps(adapter, method: str, img01: torch.Tensor, bits: torch.Tensor, size: int, batch: int = 64):
    """Decode every crop of one size of img01 (1,3,512,512) -> (boxes, gaps (n_crops,))."""
    from .decoders import to_scores
    boxes = crop_boxes(size)
    gaps = []
    for i in range(0, len(boxes), batch):
        crops = torch.cat([img01[..., y:y + size, x:x + size] for y, x in boxes[i:i + batch]])
        _, _, out = adapter(crops)
        gaps.append(match_gap(to_scores(out, C.NET_OUTPUT[method], len(bits)), bits))
    return boxes, torch.cat(gaps).float().cpu()


@torch.no_grad()
def tile_agreement(adapter, img01: torch.Tensor, bits: torch.Tensor, size: int = 256,
                   overlap: float = 0.75, batch: int = 32) -> torch.Tensor:
    """Agreement map from tiles at the decoder's native size (no downscaling): each tile's
    per-position agreement (e.g. 8x8 for a 256-px tile = 32 px per position, vs 64 px for the
    whole image) is pasted back onto the 128 grid; overlapping tiles are averaged."""
    boxes = crop_boxes(size, overlap)
    acc = torch.zeros(C.GRID, C.GRID)
    cnt = torch.zeros(C.GRID, C.GRID)
    k = size // C.CELL
    for i in range(0, len(boxes), batch):
        part = boxes[i:i + batch]
        crops = torch.cat([img01[..., y:y + size, x:x + size] for y, x in part])
        _, agree, _ = adapter(crops, bits[None].expand(len(part), -1))
        up = F.interpolate(agree.float(), size=(k, k), mode="bilinear", align_corners=False)[:, 0].cpu()
        for (y, x), a in zip(part, up):
            cy, cx = y // C.CELL, x // C.CELL
            acc[cy:cy + k, cx:cx + k] += a
            cnt[cy:cy + k, cx:cx + k] += 1
    return acc / cnt.clamp(min=1)


def paste(boxes, gaps: torch.Tensor, size: int) -> torch.Tensor:
    """Put each crop's gap on the 128x128 cells it covers; overlapping crops are averaged."""
    acc = torch.zeros(C.GRID, C.GRID)
    cnt = torch.zeros(C.GRID, C.GRID)
    k = size // C.CELL
    for (y, x), g in zip(boxes, gaps.tolist()):
        cy, cx = y // C.CELL, x // C.CELL
        acc[cy:cy + k, cx:cx + k] += g
        cnt[cy:cy + k, cx:cx + k] += 1
    return acc / cnt.clamp(min=1)


def box_map(box) -> torch.Tensor:
    """[x1, y1, x2, y2] in 0..1 (or None) -> (128,128) map, 1 inside the box."""
    m = torch.zeros(C.GRID, C.GRID)
    if box is not None:
        x1, y1, x2, y2 = (min(max(int(round(float(v) * C.GRID)), 0), C.GRID) for v in box)
        m[y1:y2, x1:x2] = 1
    return m


# ------------------------------------------------------------ stored maps -> head inputs
def path(method: str):
    return C.CACHE_DIR / f"heat_{method}.pt"


def load(method: str) -> dict:
    """{'maps': {name: (K,128,128) fp16}, 'mean', 'std': (K,), 'channels': [...]} from Step 3b."""
    p = path(method)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run scripts.step3b_heatmaps for {method} first")
    return torch.load(p, weights_only=False)


def get(store: dict, name: str) -> torch.Tensor:
    """Normalised heatmaps of one image: (K,128,128) float."""
    m = store["maps"][name].float()
    return (m - store["mean"][:, None, None]) / store["std"][:, None, None]


def combine(agree, maps):
    """Extra head inputs at 128x128: agreement map (B,1,h,w) and/or heatmaps (B,K,128,128)."""
    parts = []
    if agree is not None:
        parts.append(F.interpolate(agree.float(), size=(C.GRID, C.GRID), mode="bilinear",
                                   align_corners=False))
    if maps is not None:
        parts.append(maps.float().to(parts[0].device) if parts else maps.float())
    return torch.cat(parts, 1) if parts else None


# ------------------------------------------------------------ the "heat" model
def cells(maps: torch.Tensor) -> torch.Tensor:
    """(K,128,128) -> (128*128, K): one row per cell."""
    return maps.float().permute(1, 2, 0).reshape(-1, maps.shape[0])


def fit_logistic(X, y, l2: float = 1e-4):
    """Tiny logistic regression: X (N,K) cell features, y (N,) 0/1 -> (mean, std, weights)."""
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    Z = (X - mu) / sd
    w = torch.zeros(X.shape[1] + 1, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Z @ w[:-1] + w[-1], y) + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return mu, sd, w.detach()


def predict_logistic(model, X) -> torch.Tensor:
    mu, sd, w = model
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.sigmoid(((X - mu) / sd) @ w[:-1] + w[-1])
