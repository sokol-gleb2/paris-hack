"""Step 3f: seeded box search + edge refinement ("undo the placement": the decoder reads the exact
watermark rectangle at 0.89-0.97 bit accuracy, but misaligned crops poorly).

Per image: propose rectangles from a seed probability map (its bounding boxes at several cutoffs),
decode each with the kit decoder, refine the best ones by moving each edge (32 -> 16 -> 8 px) while
the known-message match (heat.match_gap) rises. Then, on development images only, choose:
  - the mask: rectangle R, or seed inside R + w * seed outside R (keeps holes)
  - a gate g: use that mask only where the match gap >= g, else the seed map.
Writes tags `srect` (R alone) and `sbox` (gated), and cache/rects_<m>.pt (for step3g_matched).
Seeds: trustmark stageAh, pixelseal stageA, stegastamp its box-search map (box_<m>.pt).
Run: python -m scripts.step3f_seeded_boxes --method trustmark [--seed stageAh]
     parallel: --shard k --nshards N (k = 0..N-1), then --merge --nshards N
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from wmloc import config as C
from wmloc import data, metric
from wmloc.heat import match_gap

DEFAULT_SEED = {"trustmark": "stageAh", "pixelseal": "stageA", "stegastamp": "boxmap"}
CUTS = (0.3, 0.4, 0.5, 0.6, 0.7)


def load_seed(method, seed):
    """{name: (128,128) float probability map} for train + test."""
    if seed == "boxmap":
        maps = torch.load(C.CACHE_DIR / f"box_{method}.pt", weights_only=False)["maps"]
        return {n: v.float() for n, v in maps.items()}
    out = {}
    for kind in ("oof", "test"):
        d = torch.load(C.CACHE_DIR / f"{kind}_{seed}_{method}.pt", weights_only=False)
        out.update(zip(d["names"], d["probs"].float()))
    return out


def fix(r, method=None, min_side=64):
    """Clip a rectangle (y0, x0, y1, x1) to the image; None if too small or too elongated."""
    y0, x0, y1, x1 = r
    y0, x0, y1, x1 = max(0, y0), max(0, x0), min(C.IMG, y1), min(C.IMG, x1)
    if y1 - y0 < min_side or x1 - x0 < min_side:
        return None
    if method and max(y1 - y0, x1 - x0) / min(y1 - y0, x1 - x0) > C.MAX_ASPECT[method]:
        return None
    return y0, x0, y1, x1


def proposals(seed_map, method):
    """Bounding boxes (px) of the seed map above several cutoffs (all cells, and the largest blob)."""
    p, rects = seed_map.numpy(), set()
    for t in CUTS:
        msk = p >= t
        if not msk.any():
            continue
        lab, n = ndimage.label(msk)
        sizes = ndimage.sum(msk, lab, range(1, n + 1))
        for mm in (msk, lab == int(np.argmax(sizes)) + 1):
            ys, xs = np.nonzero(mm)
            rects.add((int(ys.min()) * C.CELL, int(xs.min()) * C.CELL,
                       (int(ys.max()) + 1) * C.CELL, (int(xs.max()) + 1) * C.CELL))
    out = [r for r in (fix(r, method) for r in rects) if r is not None]
    return out or [(0, 0, C.IMG, C.IMG)]


@torch.no_grad()
def decode_gaps(adapter, method, img01, bits, rects, batch=32):
    """Match gap of each rectangle crop. Crops are resized to the decoder's input size first (the same
    resize the kit wrapper does), so rectangles of any size share one batch."""
    from wmloc.decoders import to_scores
    size, mode = (400, "bicubic") if method == "stegastamp" else (adapter.size, "bilinear")
    out = []
    for i in range(0, len(rects), batch):
        crops = torch.cat([F.interpolate(img01[..., y0:y1, x0:x1], size=(size, size), mode=mode,
                                         align_corners=False, antialias=True).clamp(0, 1)
                           for y0, x0, y1, x1 in rects[i:i + batch]])
        _, _, o = adapter(crops)
        out += match_gap(to_scores(o, C.NET_OUTPUT[method], len(bits)), bits).float().cpu().tolist()
    return out


def refine(gap_fn, method, rect, gap, steps=(32, 16, 8), rounds=6):
    """Coordinate ascent on the 4 edges: take the best of the 8 moves while the match gap rises.
    gap_fn(list of rects) -> list of gaps."""
    for st in steps:
        for _ in range(rounds):
            y0, x0, y1, x1 = rect
            cands = [fix(c, method) for c in (
                (y0 - st, x0, y1, x1), (y0 + st, x0, y1, x1), (y0, x0 - st, y1, x1), (y0, x0 + st, y1, x1),
                (y0, x0, y1 - st, x1), (y0, x0, y1 + st, x1), (y0, x0, y1, x1 - st), (y0, x0, y1, x1 + st))]
            cands = [c for c in cands if c is not None]
            if not cands:
                break
            g = gap_fn(cands)
            j = int(np.argmax(g))
            if g[j] <= gap + 1e-4:
                break
            rect, gap = cands[j], g[j]
    return rect, gap


def rect_mask(r):
    m = torch.zeros(C.GRID, C.GRID)
    y0, x0, y1, x1 = r
    m[y0 // C.CELL:-(-y1 // C.CELL), x0 // C.CELL:-(-x1 // C.CELL)] = 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--seed", default=None, help="seed tag (default per method) or 'boxmap'")
    ap.add_argument("--top", type=int, default=3, help="proposals refined per image")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    m = args.method
    seed_tag = args.seed or DEFAULT_SEED[m]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_all = data.cached_split("train")
    folds = data.make_folds(train_all)
    train = data.by_method(train_all, m)
    test = data.by_method(data.cached_split("test"), m)
    seed = load_seed(m, seed_tag)

    if args.merge:
        rects = {}
        for f in sorted(C.CACHE_DIR.glob(f"rects_{m}_part*of{args.nshards}.pt")):
            rects.update(torch.load(f, weights_only=False))
    else:
        from wmloc.decoders import DecoderAdapter
        adapter = DecoderAdapter(m, device=dev).to(dev).eval()
        todo = (train + test)[args.shard::args.nshards]
        print(f"{m}: seed {seed_tag}, {len(todo)} images (shard {args.shard}/{args.nshards})", flush=True)
        rects = {}
        for i, r in enumerate(todo):
            x = data.image_to_tensor(r.image)[None].to(dev)
            bits = torch.from_numpy(r.bits).to(dev)
            gap_fn = lambda rs: decode_gaps(adapter, m, x, bits, rs)    # noqa: E731
            props = proposals(seed[r.name], m)
            g = gap_fn(props)
            order = np.argsort(g)[::-1][:args.top]
            best = max((refine(gap_fn, m, props[j], g[j]) for j in order), key=lambda rg: rg[1])
            rects[r.name] = {"rect": best[0], "gap": best[1], "coarse_gap": float(max(g))}
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}  gap {best[1]:.3f} (before refining {max(g):.3f})", flush=True)
        if args.nshards > 1:
            torch.save(rects, C.CACHE_DIR / f"rects_{m}_part{args.shard}of{args.nshards}.pt")
            print(f"saved part {args.shard}/{args.nshards}")
            return
    missing = [r.name for r in train + test if r.name not in rects]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    torch.save(rects, C.CACHE_DIR / f"rects_{m}.pt")

    # ---------------- choose mask + gate on development images; report on the holdout
    names = [r.name for r in train]
    gts = {r.name: r.mask128 for r in train}
    dev_n = [n for n in names if folds[n] != C.HOLDOUT_FOLD]
    hold_n = [n for n in names if folds[n] == C.HOLDOUT_FOLD]
    R = {n: rect_mask(v["rect"]) for n, v in rects.items()}

    def mask(n, kind, w):
        S = seed[n]
        return {"seed": S, "rect": R[n], "rectw": S * R[n] + w * S * (1 - R[n])}[kind]

    def score(n, p):
        return metric.miou_image((p >= 0.5).numpy().astype(np.uint8), gts[n])

    def mean(ns, kind, w=0.0, g=None):
        return float(np.mean([score(n, mask(n, kind if (g is None or rects[n]["gap"] >= g) else "seed", w))
                              for n in ns]))

    qs = list(np.quantile([rects[n]["gap"] for n in dev_n], [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]))
    best = (mean(dev_n, "seed"), "seed", 0.0, None)
    for kind, ws in (("rect", (0.0,)), ("rectw", (0.0, 0.25, 0.5))):
        for w in ws:
            for g in [None] + qs:
                s = mean(dev_n, kind, w, g)
                if s > best[0]:
                    best = (s, kind, w, g)
    _, kind, w, g = best
    print(f"\n{m}: seed {seed_tag}; chosen on dev: mask={kind}, w={w}, gate={g}")
    for label, ns in (("dev", dev_n), ("holdout", hold_n)):
        orc = np.mean([max(score(n, mask(n, "seed", 0)), score(n, mask(n, "rect", 0))) for n in ns])
        print(f"  {label:7s} seed {mean(ns, 'seed'):.4f} | rect {mean(ns, 'rect'):.4f} | "
              f"chosen {mean(ns, kind, w, g):.4f} | oracle best-of(seed, rect) {orc:.4f}")
    t = [data.edit_type(gts[n]) for n in dev_n]
    for e in ("rect", "holes", "object"):
        ns = [n for n, te in zip(dev_n, t) if te == e]
        if ns:
            print(f"  dev {e:6s} n={len(ns):3d} seed {mean(ns, 'seed'):.4f} rect {mean(ns, 'rect'):.4f} "
                  f"chosen {mean(ns, kind, w, g):.4f}")

    def final(n):
        return mask(n, kind if (g is None or rects[n]["gap"] >= g) else "seed", w)

    gt_t = torch.from_numpy(np.stack([gts[n] for n in names]))
    for tag, fn in (("srect", lambda n: R[n]), ("sbox", final)):
        torch.save({"names": names, "probs": torch.stack([fn(n) for n in names]).half(), "gts": gt_t,
                    "folds": [folds[n] for n in names]}, C.CACHE_DIR / f"oof_{tag}_{m}.pt")
        torch.save({"names": [r.name for r in test], "probs": torch.stack([fn(r.name) for r in test]).half()},
                   C.CACHE_DIR / f"test_{tag}_{m}.pt")
    print("saved tags srect, sbox and rects_<m>.pt")


if __name__ == "__main__":
    main()
