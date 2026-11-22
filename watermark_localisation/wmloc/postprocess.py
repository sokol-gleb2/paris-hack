"""Probability map -> final 0/1 mask, and per-method tuning on OOF predictions."""
from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage

from . import metric

THRS = [round(float(t), 2) for t in np.arange(0.20, 0.801, 0.05)]
MIN_CELLS = [0, 8, 32, 128]
MAX_HOLE = [0, 8, 32]
RECT_FILL = [0.0, 0.95, 0.90, 0.85]
KEEP_CONF = 0.9          # small blobs with a peak probability this high are never removed
CLEANUP_MARGIN = 0.005
ALLOW_EMPTY = False      # Step 0b: 0 empty true masks in train -> never predict an empty mask   # cleanup must beat the plain threshold by this much to be used


def _largest(m: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def postprocess(prob, thr=0.5, min_cells=0, max_hole=0, rect_fill=0.0, never_empty=True) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    m = prob >= thr
    if min_cells and m.any():
        lab, n = ndimage.label(m)
        idx = np.arange(1, n + 1)
        sizes = np.asarray(ndimage.sum(m, lab, idx))
        peaks = np.asarray(ndimage.maximum(prob, lab, idx))
        keep = np.isin(lab, idx[(sizes >= min_cells) | (peaks >= KEEP_CONF)])
        if keep.any():
            m = keep
    if max_hole and m.any():
        holes = ndimage.binary_fill_holes(m) & ~m
        lab, n = ndimage.label(holes)
        if n:
            sizes = ndimage.sum(holes, lab, range(1, n + 1))
            m = m | np.isin(lab, np.nonzero(sizes <= max_hole)[0] + 1)
    if rect_fill and m.any():
        lab, _ = ndimage.label(m)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if (lab[sl] == i).mean() >= rect_fill:
                m[sl] = True
    if never_empty and not m.any():       # largest blob of the top 5% cells
        k = max(16, int(0.05 * prob.size))
        top = np.zeros(prob.size, bool)
        top[np.argsort(prob.ravel())[-k:]] = True
        m = _largest(top.reshape(prob.shape))
    return m.astype(np.uint8)


def evaluate(probs, gts, **params) -> float:
    return metric.score([postprocess(p, **params) for p in probs], gts)


def tune(probs, gts, verbose: bool = False):
    """1) threshold alone, with never-empty on and off; 2) cleanup options around that
    threshold, kept only if they beat the plain threshold by CLEANUP_MARGIN.
    Returns (score, params)."""
    probs = [np.asarray(p, dtype=np.float32) for p in probs]
    gts = [np.asarray(g) for g in gts]
    plain = max(((evaluate(probs, gts, thr=t, never_empty=ne), dict(thr=t, never_empty=ne))
                 for ne in ((True, False) if ALLOW_EMPTY else (True,)) for t in THRS), key=lambda sp: sp[0])
    best = plain
    near = [t for t in THRS if abs(t - plain[1]["thr"]) <= 0.051]
    for thr, mc, mh, rf in itertools.product(near, MIN_CELLS, MAX_HOLE, RECT_FILL):
        if mc == 0 and mh == 0 and rf == 0.0:
            continue
        params = dict(thr=thr, min_cells=mc, max_hole=mh, rect_fill=rf,
                      never_empty=plain[1]["never_empty"])
        s = evaluate(probs, gts, **params)
        if s > best[0]:
            best = (s, params)
    if best[0] < plain[0] + CLEANUP_MARGIN:
        best = plain
    if verbose:
        print(f"    plain {plain[1]}: {plain[0]:.4f} -> chosen {best[0]:.4f} {best[1]}")
    return best
