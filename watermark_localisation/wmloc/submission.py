"""Build, validate and save the submission file.

Format (task description): torch .pt file holding
    {"image_names": names, "masks": masks}
names[i] matches masks[i]; masks is a tensor [num_images, 128, 128] of 0/1.
Rules: every test image exactly once, no duplicates, only 0/1, file <= 20 MB.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from . import config as C

MAX_BYTES = 20 * 1000 * 1000   # 20 MB; stay under 20 MiB too


def build(names: list[str], masks) -> dict:
    masks = torch.as_tensor(np.asarray(masks)).to(torch.uint8)
    return {"image_names": [str(n) for n in names], "masks": masks}


def validate(sub: dict, expected_names: list[str] | None = None) -> None:
    names, masks = sub["image_names"], sub["masks"]
    assert isinstance(masks, torch.Tensor), "masks must be a torch tensor"
    assert masks.ndim == 3 and tuple(masks.shape[1:]) == (C.GRID, C.GRID), \
        f"masks shape {tuple(masks.shape)} != [N,{C.GRID},{C.GRID}]"
    assert len(names) == masks.shape[0], "names and masks differ in length"
    assert len(set(names)) == len(names), "duplicate image names"
    vals = torch.unique(masks)
    assert set(vals.tolist()) <= {0, 1}, f"mask values must be 0/1, got {vals.tolist()}"
    if expected_names is not None:
        missing = set(expected_names) - set(names)
        extra = set(names) - set(expected_names)
        assert not missing, f"{len(missing)} test images missing, e.g. {sorted(missing)[:3]}"
        assert not extra, f"{len(extra)} unknown names, e.g. {sorted(extra)[:3]}"
    # every image should have at least one watermarked cell (all images are watermarked)
    empty = int((masks.flatten(1).sum(1) == 0).sum())
    if empty:
        print(f"WARNING: {empty} masks are all zero (right only if the true mask is empty; else mIoU <= 0.5)")


def save(sub: dict, path: str | Path, expected_names: list[str] | None = None) -> Path:
    validate(sub, expected_names)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sub, path)
    size = os.path.getsize(path)
    assert size <= MAX_BYTES, f"file is {size/1e6:.1f} MB > 20 MB"
    back = torch.load(path, weights_only=False)      # round-trip check
    validate(back, expected_names)
    print(f"saved {path}  ({len(sub['image_names'])} masks, {size/1e6:.2f} MB, "
          f"mean watermarked share {sub['masks'].float().mean():.3f})")
    return path


TASK_ID = "34-watermark-localisation-image"   # from starter kit submission_template.py


def submit(path: str | Path) -> None:
    """Same POST as the starter kit's submission_template.submit.
    Needs env vars: WMLOC_API_KEY (team key), WMLOC_BASE_URL (leaderboard server)."""
    import requests
    base = os.environ.get("WMLOC_BASE_URL", "http://34.147.229.55")
    key = os.environ.get("WMLOC_API_KEY")
    assert key, "export WMLOC_API_KEY=<your team API key> first"
    with open(path, "rb") as f:
        resp = requests.post(f"{base}/submit/{TASK_ID}", headers={"X-API-Key": key},
                             files={"file": (Path(path).name, f, "application/octet-stream")},
                             timeout=(10, 120))
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    print(f"HTTP {resp.status_code}: {body}")
    resp.raise_for_status()
