"""Data records, loading adapter, folds.

Everything downstream uses `Record`. Only `load_split` touches the starter kit,
so that is the one function to adapt.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from . import config as C


@dataclass
class Record:
    name: str                       # image name exactly as the submission needs it
    image: np.ndarray               # uint8, (512, 512, 3), RGB, exactly as stored
    method: str                     # one of C.METHODS
    bits: np.ndarray                # uint8 0/1, (n_bits,)
    mask128: Optional[np.ndarray]   # uint8 0/1, (128, 128); None for test
    mask512: Optional[np.ndarray]   # uint8 0/1, (512, 512); None for test


# ---------------------------------------------------------------- converters
def parse_bits(msg) -> np.ndarray:
    """Accepts '0101..', '0 1 0 1', '[0, 1, ...]', list, np array or tensor."""
    if isinstance(msg, torch.Tensor):
        msg = msg.detach().cpu().numpy()
    if isinstance(msg, (list, tuple, np.ndarray)):
        arr = np.asarray(msg).astype(np.float32).reshape(-1)
        return (arr > 0.5).astype(np.uint8)
    s = "".join(ch for ch in str(msg) if ch in "01")
    if not s:
        raise ValueError(f"cannot parse message: {msg!r}")
    return np.frombuffer(s.encode(), dtype=np.uint8) - ord("0")


def to_uint8_hwc(img) -> np.ndarray:
    """Tensor/array in CHW or HWC, uint8 or float in [0,1] -> uint8 HWC.
    A float image that came from an 8-bit PNG is restored exactly by round(x*255)."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if hasattr(img, "convert"):              # PIL image
        img = np.asarray(img.convert("RGB"))
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(np.round(img.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return np.ascontiguousarray(img)


def to_mask(m) -> Optional[np.ndarray]:
    if m is None:
        return None
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    if hasattr(m, "convert"):
        m = np.asarray(m.convert("L"))
    m = np.asarray(m).squeeze()
    return (m > (127 if m.max() > 1 else 0.5)).astype(np.uint8)


def image_to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> float32 (3,H,W) in [0,1]. The ONLY conversion allowed."""
    return torch.from_numpy(img_u8).permute(2, 0, 1).float().div_(255.0)


# ---------------------------------------------------------------- loading
def load_split(split: str) -> list[Record]:
    """Reads the unzipped release directly (same layout as the starter kit's
    LocalisationDataset). Not imported from the kit because its package is also
    called `wmloc` and would clash with this one.
        <DATA_DIR>/<split>/images/<name>.png, metadata.json,
        train/masks/<name>.png (255 = wm), train/masks_128.npz (names, masks)"""
    import json
    from PIL import Image
    d = C.DATA_DIR / split
    meta = json.loads((d / "metadata.json").read_text())
    m128 = {}
    if (d / "masks_128.npz").exists():
        z = np.load(d / "masks_128.npz")
        m128 = dict(zip(z["names"].tolist(), z["masks"]))
    recs = []
    for m in meta:
        name = m["image_name"]
        mp = d / "masks" / f"{name}.png"
        recs.append(Record(
            name=name,
            image=np.asarray(Image.open(d / "images" / f"{name}.png").convert("RGB")),
            method=m["method"],
            bits=parse_bits(m["message"]),
            mask128=to_mask(m128[name]) if name in m128 else None,
            mask512=to_mask(Image.open(mp)) if mp.exists() else None,
        ))
    return recs


def records_from_dataset(ds, keys: dict) -> list[Record]:
    """Generic converter. `keys` maps our fields to the dataset's dict keys, e.g.
    {'name':'image_name','image':'image','method':'method','message':'message',
     'mask128':'mask_128','mask512':'mask_512'}"""
    recs = []
    for i in range(len(ds)):
        it = ds[i]
        get = (lambda k: it.get(keys[k]) if keys.get(k) else None) if isinstance(it, dict) \
            else (lambda k: getattr(it, keys[k], None) if keys.get(k) else None)
        recs.append(Record(
            name=str(get("name")),
            image=to_uint8_hwc(get("image")),
            method=str(get("method")),
            bits=parse_bits(get("message")),
            mask128=to_mask(get("mask128")),
            mask512=to_mask(get("mask512")),
        ))
    return recs


def cached_split(split: str) -> list[Record]:
    """Load once, then reuse a torch file (fast restarts)."""
    path = C.CACHE_DIR / f"records_{split}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    recs = load_split(split)
    torch.save(recs, path)
    return recs


def by_method(recs: list[Record], method: str) -> list[Record]:
    return [r for r in recs if r.method == method]


# ---------------------------------------------------------------- folds
def make_folds(recs: list[Record], n_folds: int = C.N_FOLDS, seed: int = C.SEED,
               groups: dict[str, str] | None = None) -> dict[str, int]:
    """Fold id per image name, stratified by method and by watermarked-area quintile.
    Images in the same group (edits of one original photo; Step 0b writes cache/groups.json)
    always share a fold. Without groups this gives exactly the same folds as before."""
    if groups is None:
        path = C.CACHE_DIR / "groups.json"
        groups = json.loads(path.read_text()) if path.exists() else {}
    rng = random.Random(seed)
    folds: dict[str, int] = {}
    for m in sorted({r.method for r in recs}):
        units: dict[str, list[Record]] = {}
        for r in recs:
            if r.method == m:
                units.setdefault(groups.get(r.name, r.name), []).append(r)
        ordered = sorted(units.values(), key=lambda u: float(np.mean(
            [r.mask128.mean() if r.mask128 is not None else 0.0 for r in u])))
        for start in range(0, len(ordered), n_folds):      # each block of n spans all folds
            block = ordered[start:start + n_folds]
            ids = list(range(n_folds))
            rng.shuffle(ids)
            for unit, f in zip(block, ids):
                for r in unit:
                    folds[r.name] = f
    return folds


# ---------------------------------------------------------------- mask stats
def edit_type(mask128: np.ndarray) -> str:
    """Rough edit-type label from mask shape: 'rect' (outpainting), 'holes'
    (removal/addition), 'object' (transplantation), 'empty'."""
    from scipy import ndimage
    if mask128.sum() == 0:
        return "empty"
    ys, xs = np.nonzero(mask128)
    box = mask128[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    fill = box.mean()
    if fill > 0.97:
        return "rect"
    filled = ndimage.binary_fill_holes(box)
    if fill > 0.5 and (filled.sum() - box.sum()) > 0.02 * box.size:
        return "holes"
    if fill > 0.85:
        return "holes"   # rectangle with a notch cut from an edge
    return "object"
