"""Step 3d: box search. Decode many candidate rectangles per image; the rectangles whose bits match the
known message best are probably the watermark. (TrustMark and StegaStamp read the exact watermark box
well, but not half-size crops, so Step 3b's crop heatmaps fail for them.)

Writes cache/box_<m>.pt (maps) and the "box" tag: cache/oof_box_<m>.pt + test_box_<m>.pt
(map + average mask -> probability with a tiny logistic fit per fold; Step 4 compares it like any tag).
Run: python -m scripts.step3d_boxes --method stegastamp [--train-only] [--top 5]
     parallel: --shard k --nshards N for k = 0..N-1, then --merge --nshards N
"""
import argparse

import numpy as np
import torch

from wmloc import config as C
from wmloc import data, heat, metric
from wmloc.decoders import DecoderAdapter, to_scores

SIDES = (128, 192, 256, 320, 384, 448, 512)
STEP = 64


def candidate_boxes(method):
    """(y, x, h, w) rectangles on a 64-px grid, aspect ratio within the method's limit (~800 boxes)."""
    out = []
    for h in SIDES:
        for w in SIDES:
            if max(h, w) / min(h, w) > C.MAX_ASPECT[method]:
                continue
            for y in range(0, C.IMG - h + 1, STEP):
                for x in range(0, C.IMG - w + 1, STEP):
                    out.append((y, x, h, w))
    return out


@torch.no_grad()
def box_map(adapter, method, img01, bits, boxes, top, batch=64):
    """(128,128) map: the `top` best-matching boxes, weighted by their match gap; + best gap."""
    by_size = {}
    for i, (_, _, h, w) in enumerate(boxes):
        by_size.setdefault((h, w), []).append(i)
    g = torch.zeros(len(boxes))
    for (h, w), idx in by_size.items():
        for s in range(0, len(idx), batch):
            part = idx[s:s + batch]
            crops = torch.cat([img01[..., boxes[i][0]:boxes[i][0] + h, boxes[i][1]:boxes[i][1] + w]
                               for i in part])
            _, _, out = adapter(crops)
            g[part] = heat.match_gap(to_scores(out, C.NET_OUTPUT[method], len(bits)), bits).float().cpu()
    best = torch.topk(g, top).indices.tolist()
    m, wsum = torch.zeros(C.GRID, C.GRID), 0.0
    for i in best:
        y, x, h, w = boxes[i]
        wt = max(float(g[i]), 0.0) + 1e-6
        m[y // C.CELL:(y + h) // C.CELL, x // C.CELL:(x + w) // C.CELL] += wt
        wsum += wt
    return m / wsum, float(g[best[0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--top", type=int, default=5, help="best boxes averaged into the map")
    ap.add_argument("--train-only", action="store_true", help="quick test: skip the 340 test images")
    ap.add_argument("--shard", type=int, default=0, help="with --nshards: which slice of the images")
    ap.add_argument("--nshards", type=int, default=1, help="split the images over N parallel processes")
    ap.add_argument("--merge", action="store_true", help="combine the N part files, then fit + save")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = args.method

    train_all = data.cached_split("train")
    folds = data.make_folds(train_all)
    train = data.by_method(train_all, m)
    test = [] if args.train_only else data.by_method(data.cached_split("test"), m)

    if args.merge:                                   # combine the parts written by --shard runs
        maps, best_gap = {}, {}
        for f in sorted(C.CACHE_DIR.glob(f"box_{m}_part*of{args.nshards}.pt")):
            d = torch.load(f, weights_only=False)
            maps.update(d["maps"]); best_gap.update(d["best_gap"])
        missing = [r.name for r in train + test if r.name not in maps]
        assert not missing, f"{len(missing)} images missing from the parts, e.g. {missing[:3]}"
        print(f"{m}: merged {len(maps)} maps from {args.nshards} parts", flush=True)
    else:
        adapter = DecoderAdapter(m, device=dev).to(dev).eval()
        boxes = candidate_boxes(m)
        todo = (train + test)[args.shard::args.nshards]
        print(f"{m}: {len(boxes)} candidate boxes per image, {len(todo)} images "
              f"(shard {args.shard}/{args.nshards})", flush=True)
        maps, best_gap = {}, {}
        for i, r in enumerate(todo):
            x = data.image_to_tensor(r.image)[None].to(dev)
            maps[r.name], best_gap[r.name] = box_map(adapter, m, x, torch.from_numpy(r.bits).to(dev), boxes, args.top)
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        if args.nshards > 1:                         # partial result: save and stop
            torch.save({"maps": maps, "best_gap": best_gap},
                       C.CACHE_DIR / f"box_{m}_part{args.shard}of{args.nshards}.pt")
            print(f"saved part {args.shard}/{args.nshards}; run --merge --nshards {args.nshards} when all parts exist")
            return
    torch.save({"maps": maps, "best_gap": best_gap}, C.CACHE_DIR / f"box_{m}.pt")

    # quick look: how often does the single best box overlap the true mask well?
    names = [r.name for r in train]
    gts = np.stack([r.mask128 for r in train])
    raw = metric.score([(maps[n].numpy() > 0.5).astype(np.uint8) for n in names], gts)
    print(f"{m} raw box map (>0.5) train mIoU {raw:.4f}")

    # "box" model: logistic fit per fold on [box map, average mask]; holdout never used for fitting
    def feats(name, prior):
        return torch.stack([maps[name], torch.from_numpy(prior)]).float()

    oof = torch.zeros(len(train), C.GRID, C.GRID)
    for k in range(C.N_FOLDS):
        fit = [i for i, n in enumerate(names) if folds[n] != k and folds[n] != C.HOLDOUT_FOLD]
        prior = gts[fit].mean(0).astype(np.float32)
        model = heat.fit_logistic(torch.cat([heat.cells(feats(names[i], prior)) for i in fit]),
                                  np.concatenate([gts[i].reshape(-1) for i in fit]))
        for i, n in enumerate(names):
            if folds[n] == k:
                oof[i] = heat.predict_logistic(model, heat.cells(feats(n, prior))).reshape(C.GRID, C.GRID)
    torch.save({"names": names, "probs": oof.half(), "gts": torch.from_numpy(gts),
                "folds": [folds[n] for n in names]}, C.CACHE_DIR / f"oof_box_{m}.pt")
    print(f"{m} box model: OOF mIoU@0.5 {metric.score(oof.numpy() >= 0.5, gts):.4f} "
          "(untuned; compare in Step 4 with --tags prior,box)")
    if test:
        prior = gts.mean(0).astype(np.float32)
        model = heat.fit_logistic(torch.cat([heat.cells(feats(n, prior)) for n in names]), gts.reshape(-1))
        probs = torch.stack([heat.predict_logistic(model, heat.cells(feats(r.name, prior))).reshape(C.GRID, C.GRID)
                             for r in test])
        torch.save({"names": [r.name for r in test], "probs": probs.half()}, C.CACHE_DIR / f"test_box_{m}.pt")


if __name__ == "__main__":
    main()
