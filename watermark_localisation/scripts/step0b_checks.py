"""Step 0b: data checks to run once, BEFORE any training. CPU is fine.

1. Empty true masks per method (Step 4 tunes "never empty" because of these).
2. Repeat photos (several edits of one original): pairs whose grey thumbnails correlate
   above --dup-thr are grouped and saved to cache/groups.json; make_folds keeps each
   group in one fold. Never change groups.json after Stage A has started.
3. Our metric vs the starter kit's scorer.

Run: python -m scripts.step0b_checks [--dup-thr 0.95] [--force]
"""
import argparse
import json
from collections import Counter

import numpy as np

from wmloc import config as C
from wmloc import data, metric


def thumb(img_u8: np.ndarray, s: int = 32) -> np.ndarray:
    """Grey s x s thumbnail with zero mean and unit length, so a dot product = correlation."""
    g = img_u8.astype(np.float32).mean(axis=2)
    t = g.reshape(s, C.IMG // s, s, C.IMG // s).mean(axis=(1, 3)).reshape(-1)
    t = t - t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def kit_score(preds, gts) -> float:
    """The starter kit's official scorer (starter_kit/wmloc/utils.py)."""
    from wmloc.decoders import kit_module
    return kit_module("utils").score(preds, gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup-thr", type=float, default=0.95,
                    help="thumbnail correlation that counts as the same original photo")
    ap.add_argument("--force", action="store_true", help="overwrite a different groups.json")
    args = ap.parse_args()
    train = data.cached_split("train")

    # --- 1. empty true masks
    for m in C.METHODS:
        rs = data.by_method(train, m)
        n_empty = sum(int(r.mask128.sum() == 0) for r in rs)
        print(f"{m:11s} empty true masks: {n_empty} / {len(rs)}")

    # --- 2. repeat photos
    names = [r.name for r in train]
    T = np.stack([thumb(r.image) for r in train])
    sim = T @ T.T
    iu = np.triu_indices(len(names), k=1)
    print("most similar pairs (1.0 = same thumbnail):")
    for o in np.argsort(sim[iu])[::-1][:15]:
        i, j = iu[0][o], iu[1][o]
        print(f"  {sim[i, j]:.3f}  {names[i]}  {names[j]}")
    parent = list(range(len(names)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.nonzero(np.triu(sim, k=1) >= args.dup_thr)):
        parent[find(i)] = find(j)
    root = [find(i) for i in range(len(names))]
    size = Counter(root)
    groups = {n: names[root[i]] for i, n in enumerate(names) if size[root[i]] > 1}
    print(f"repeat-photo groups at {args.dup_thr}: {len(set(groups.values()))} groups, {len(groups)} photos")
    path = C.CACHE_DIR / "groups.json"
    if path.exists() and json.loads(path.read_text()) != groups and not args.force:
        print(f"  {path} exists and differs: NOT overwritten (fold ids would change). "
              "Use --force only before Stage A.")
    else:
        path.write_text(json.dumps(groups, indent=1))
        print(f"  saved {path}")

    # --- 3. metric cross-check
    rng = np.random.default_rng(0)
    gts = [r.mask128 for r in train[:100]]
    z = np.zeros((C.GRID, C.GRID), np.uint8)
    cases = {
        "all zeros": ([np.zeros_like(g) for g in gts], gts),
        "all ones": ([np.ones_like(g) for g in gts], gts),
        "random 30%": ([(rng.random(g.shape) < 0.3).astype(np.uint8) for g in gts], gts),
        "perfect": ([g.copy() for g in gts], gts),
        "empty vs empty": ([z], [z]),
    }
    try:
        for k, (preds, gg) in cases.items():
            ours, kit = metric.score(preds, gg), kit_score(preds, gg)
            flag = "OK" if abs(ours - kit) < 1e-6 else "DIFFERENT: trust the kit, fix metric.py"
            print(f"  metric {k:14s} ours {ours:.6f}  kit {kit:.6f}  {flag}")
    except NotImplementedError as e:
        print(f"  metric cross-check skipped: {e}")


if __name__ == "__main__":
    main()
