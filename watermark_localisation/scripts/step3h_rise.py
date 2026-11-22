"""Step 3h: RISE (black-box saliency) with the known message, for StegaStamp (ONNX-only decoder).

Idea: hide random smooth parts of the photo (replaced by a heavy blur, which wipes the watermark) and
measure how well the decoder still reads the KNOWN message. Hiding watermarked pixels lowers the match;
hiding clean (restored / edited) pixels does not. Averaging the match over N random masks gives each
pixel a "carries the watermark" score -- and, unlike box search, it can mark a clean hole inside a
watermarked image.

Writes cache/rise_<m>.pt (maps) and tag "rise": cache/oof_rise_<m>.pt + test_rise_<m>.pt
(logistic per fold on [RISE map (z-scored), box map, average mask], like step3d's "box").
Run (parallel):  python -m scripts.step3h_rise --method stegastamp --shard k --nshards N   (k = 0..N-1)
                 python -m scripts.step3h_rise --method stegastamp --merge --nshards N
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from wmloc import config as C
from wmloc import data, heat, metric
from wmloc.decoders import DecoderAdapter, to_scores


def make_masks(n, s=8, p=0.5, seed=0, size=None):
    """RISE masks: random s x s keep-grid, bilinear upsampled, randomly shifted -> (n,1,size,size) in [0,1]."""
    size = size or C.IMG
    g = torch.Generator().manual_seed(seed)
    cell = -(-size // s)
    grid = (torch.rand(n, 1, s + 1, s + 1, generator=g) < p).float()
    up = F.interpolate(grid, size=((s + 1) * cell, (s + 1) * cell), mode="bilinear", align_corners=False)
    off = torch.randint(0, cell, (n, 2), generator=g)
    return torch.stack([up[i, :, off[i, 0]:off[i, 0] + size, off[i, 1]:off[i, 1] + size] for i in range(n)])


def blur(x, k=31):
    """Heavy blur (two box filters ~ Gaussian, sigma ~12 px): removes the watermark, keeps the colours."""
    for _ in range(2):
        x = F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)
    return x


@torch.no_grad()
def rise_map(score_fn, img01, masks, batch=32):
    """(GRID,GRID) saliency: sum_i (g_i - mean g) * M_i / sum_i M_i, from the per-mask match gaps g_i."""
    bl = blur(img01)
    gs = []
    for s in range(0, len(masks), batch):
        mk = masks[s:s + batch].to(img01.device)
        gs.append(score_fn(img01 * mk + bl * (1 - mk)).float().cpu())
    g = torch.cat(gs)
    m = F.adaptive_avg_pool2d(masks, (C.GRID, C.GRID))[:, 0]           # (n, GRID, GRID)
    sal = ((g - g.mean())[:, None, None] * m).sum(0) / m.sum(0).clamp(min=1e-6)
    return sal, float(g.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="stegastamp", choices=C.METHODS)
    ap.add_argument("--n", type=int, default=192, help="random masks per image")
    ap.add_argument("--grid", type=int, default=8, help="mask grid cells per side")
    ap.add_argument("--keep", type=float, default=0.5, help="share of cells kept visible")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = args.method

    train_all = data.cached_split("train")
    folds = data.make_folds(train_all)
    train = data.by_method(train_all, m)
    test = data.by_method(data.cached_split("test"), m)

    if args.merge:
        maps = {}
        for f in sorted(C.CACHE_DIR.glob(f"rise_{m}_part*of{args.nshards}.pt")):
            maps.update(torch.load(f, weights_only=False)["maps"])
        missing = [r.name for r in train + test if r.name not in maps]
        assert not missing, f"{len(missing)} images missing from the parts, e.g. {missing[:3]}"
        print(f"{m}: merged {len(maps)} RISE maps", flush=True)
    else:
        adapter = DecoderAdapter(m, device=dev).to(dev).eval()
        masks = make_masks(args.n, args.grid, args.keep, seed=1234)
        todo = (train + test)[args.shard::args.nshards]
        print(f"{m}: RISE {args.n} masks/image, {len(todo)} images (shard {args.shard}/{args.nshards})", flush=True)
        maps, t0 = {}, time.time()
        for i, r in enumerate(todo):
            x = data.image_to_tensor(r.image)[None].to(dev)
            bits = torch.from_numpy(r.bits).to(dev)

            def score_fn(xb):
                _, _, out = adapter(xb)
                return heat.match_gap(to_scores(out, C.NET_OUTPUT[m], len(bits)), bits)

            maps[r.name], _ = rise_map(score_fn, x, masks)
            if i % 10 == 0:
                print(f"  {i}/{len(todo)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if args.nshards > 1:
            torch.save({"maps": maps}, C.CACHE_DIR / f"rise_{m}_part{args.shard}of{args.nshards}.pt")
            print(f"saved part {args.shard}/{args.nshards}", flush=True)
            return
    torch.save({"maps": maps}, C.CACHE_DIR / f"rise_{m}.pt")

    names = [r.name for r in train]
    gts = np.stack([r.mask128 for r in train])
    box_path = C.CACHE_DIR / f"box_{m}.pt"
    boxes = torch.load(box_path, weights_only=False)["maps"] if box_path.exists() else None

    def z(t):
        return (t - t.mean()) / t.std().clamp(min=1e-6)

    # go/no-go print: does the raw RISE map separate watermarked from clean cells?
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(gts.reshape(-1), torch.stack([z(maps[n]) for n in names]).numpy().reshape(-1))
        print(f"{m} RISE raw map: cell AUC vs true mask {auc:.3f} (0.5 = no signal)", flush=True)
    except Exception as e:   # the AUC is only a printout
        print(f"  (AUC skipped: {type(e).__name__})", flush=True)

    def feats(name, prior):
        ch = [z(maps[name]), torch.from_numpy(prior)]
        if boxes is not None:
            ch.insert(1, boxes[name].float())
        return torch.stack(ch).float()

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
                "folds": [folds[n] for n in names]}, C.CACHE_DIR / f"oof_rise_{m}.pt")
    print(f"{m} rise model (RISE{' + box' if boxes is not None else ''} + prior): OOF mIoU@0.5 "
          f"{metric.score(oof.numpy() >= 0.5, gts):.4f} (untuned)", flush=True)
    prior = gts.mean(0).astype(np.float32)
    model = heat.fit_logistic(torch.cat([heat.cells(feats(n, prior)) for n in names]), gts.reshape(-1))
    probs = torch.stack([heat.predict_logistic(model, heat.cells(feats(r.name, prior))).reshape(C.GRID, C.GRID)
                         for r in test])
    torch.save({"names": [r.name for r in test], "probs": probs.half()}, C.CACHE_DIR / f"test_rise_{m}.pt")
    print(f"{m}: wrote oof_rise / test_rise", flush=True)


if __name__ == "__main__":
    main()
