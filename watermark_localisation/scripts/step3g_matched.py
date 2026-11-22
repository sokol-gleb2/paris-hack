"""Step 3g: matched filter with the known message (re-create the watermark, then look for it pixel by pixel).

The watermark was embedded into a rectangle at that rectangle's own size with the kit's exact embedders.
With the rectangle R from step3f (rects_<m>.pt):
  1. re-embed the known message into the crop -> expected pattern r1 = embed(crop, bits) - crop
  2. re-embed 3 random other messages        -> controls (the part that only follows image texture)
  3. local correlation (16-px windows) between the image's fine detail (high-pass) and each pattern;
     map = corr(r1) - mean corr(controls), averaged per 4x4 cell, inside R (0 outside).
Low correlation means "don't know" (weak signal or slightly wrong rectangle), not "clean": so the map is
only a SOFT extra input. Tag `mf`: per-fold logistic on [map, R, average mask, current best mask (seed)].
Caveat: the kit's embed returns 8-bit images, so the pattern includes rounding noise; expect it to work
best on StegaStamp, then TrustMark; faint Pixel Seal may drown. The go/no-go check shows this per method.
Run: python -m scripts.step3g_matched --method stegastamp --check-true 20   (go/no-go, true rectangles)
     python -m scripts.step3g_matched --method stegastamp   (needs rects_<m>.pt from step3f)
     parallel: --shard k --nshards N, then --merge --nshards N
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from wmloc import config as C
from wmloc import data, heat, metric


def highpass(z: torch.Tensor, k: int = 5) -> torch.Tensor:
    """(3,h,w) -> detail: image minus its k x k local mean (k odd; larger k keeps smoother patterns,
    e.g. StegaStamp's, which is made at 400 px and stretched to the rectangle)."""
    return z - F.avg_pool2d(F.pad(z[None], (k // 2,) * 4, mode="reflect"), k, stride=1)[0]


def local_corr(a: torch.Tensor, b: torch.Tensor, win: int = 16) -> torch.Tensor:
    """Normalised correlation of two (3,h,w) maps in win x win windows -> (h,w) in [-1, 1]."""
    def box(t):
        return F.avg_pool2d(F.pad(t[None, None], (win // 2, win - 1 - win // 2) * 2, mode="replicate"), win, stride=1)[0, 0]
    num = box((a * b).sum(0))
    den = torch.sqrt(box((a * a).sum(0)) * box((b * b).sum(0))).clamp(min=1e-8)
    return num / den


def template(wrapper, method: str, crop: np.ndarray, bits) -> torch.Tensor:
    """Expected watermark pattern for this crop and message, in FULL precision (3,h,w), values in [0,1] units.
    The kit's embed() rounds to 8 bit, which wipes out faint patterns (Pixel Seal adds very little, mostly at
    edges/texture: JND attenuation). So: Pixel Seal = raw model output; StegaStamp = the ONNX 'residual' output,
    mean-removed and stretched exactly like the kit; TrustMark = embed at 10x strength / 10."""
    bits = [int(b) for b in bits]
    c = torch.from_numpy(np.asarray(crop, np.float32)).permute(2, 0, 1) / 255.0
    if method == "pixelseal":
        dev = wrapper.device
        with torch.no_grad():
            out = wrapper.model.embed(c[None].to(dev), msgs=torch.tensor([bits], dtype=torch.float32, device=dev),
                                      is_video=False)
        return (out["imgs_w"][0].float().cpu() - c)
    if method == "stegastamp":
        from PIL import Image
        h, w = crop.shape[:2]
        small = np.asarray(Image.fromarray(crop).resize((400, 400), Image.Resampling.BICUBIC), np.float32) / 255
        res = wrapper._run(small[None], ["residual"], np.asarray([bits], np.float32))[0][0]      # (400,400,3)
        res = res - res.mean(axis=(0, 1), keepdims=True)
        full = np.stack([np.asarray(Image.fromarray(res[:, :, k], mode="F").resize((w, h), Image.Resampling.BILINEAR))
                         for k in range(3)], 2)
        return torch.from_numpy(full.astype(np.float32)).permute(2, 0, 1)
    if method == "trustmark":
        from PIL import Image
        out = wrapper.tm.encode(Image.fromarray(crop), "".join(map(str, bits)), MODE="binary", WM_STRENGTH=10.0)
        o = torch.from_numpy(np.asarray(out.convert("RGB"), np.float32)).permute(2, 0, 1) / 255.0
        return (o - c) / 10.0
    return torch.from_numpy(np.asarray(wrapper.embed(crop, bits), np.float32)).permute(2, 0, 1) / 255.0 - c


def mf_map(img_u8: np.ndarray, rect, bits, wrapper, rng, n_controls: int = 3, hp: int = 5, win: int = 16,
           method: str = "") -> torch.Tensor:
    """(128,128) matched-filter map for one image (0 outside the rectangle)."""
    y0, x0, y1, x1 = rect
    crop = np.ascontiguousarray(img_u8[y0:y1, x0:x1])
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).permute(2, 0, 1) / 255.0   # noqa: E731
    c = t(crop)
    hc = highpass(c, hp)
    corr = lambda msg: local_corr(hc, highpass(template(wrapper, method, crop, msg), hp), win)  # noqa: E731
    ctrl = torch.stack([corr(rng.integers(0, 2, len(bits))) for _ in range(n_controls)]).mean(0)
    m = corr(bits) - ctrl
    cells = F.avg_pool2d(m[None, None], C.CELL)[0, 0]
    out = torch.zeros(C.GRID, C.GRID)
    out[y0 // C.CELL:y0 // C.CELL + cells.shape[0], x0 // C.CELL:x0 // C.CELL + cells.shape[1]] = cells
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--hp", type=int, default=5, help="detail filter size (odd); try 15 for StegaStamp")
    ap.add_argument("--win", type=int, default=16, help="correlation window (px); try 32")
    ap.add_argument("--check-true", type=int, default=0,
                    help="go/no-go: N development images with holes, using each image's TRUE rectangle")
    args = ap.parse_args()
    m = args.method
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_all = data.cached_split("train")
    folds = data.make_folds(train_all)
    train = data.by_method(train_all, m)
    test = data.by_method(data.cached_split("test"), m)

    if args.check_true:
        # Step A: does the map separate watermarked pixels from holes inside the TRUE rectangle?
        from wmloc.decoders import load_wrapper
        wrapper, rng = load_wrapper(m, dev), np.random.default_rng(C.SEED)
        recs = [r for r in train if folds[r.name] != C.HOLDOUT_FOLD and data.edit_type(r.mask128) == "holes"]
        recs = recs[:args.check_true]
        pos, neg = [], []
        for r in recs:
            ys, xs = np.nonzero(r.mask128)
            rect = (int(ys.min()) * C.CELL, int(xs.min()) * C.CELL, (int(ys.max()) + 1) * C.CELL, (int(xs.max()) + 1) * C.CELL)
            mp = mf_map(r.image, rect, r.bits, wrapper, rng, hp=args.hp, win=args.win, method=m).numpy()
            cy0, cx0, cy1, cx1 = (v // C.CELL for v in rect)
            v, g = mp[cy0:cy1, cx0:cx1], r.mask128[cy0:cy1, cx0:cx1].astype(bool)
            pos += v[g].tolist(); neg += v[~g].tolist()
        allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort() + 1.0
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        print(f"GO/NO-GO {m} (hp {args.hp}, win {args.win}): {len(recs)} images with holes, true rectangles | map watermarked {np.mean(pos):+.4f} "
              f"vs hole {np.mean(neg):+.4f} | AUC {auc:.3f} (0.5 = useless; > ~0.7 = worth a full run)")
        return

    rects = torch.load(C.CACHE_DIR / f"rects_{m}.pt", weights_only=False)

    if args.merge:
        maps = {}
        for f in sorted(C.CACHE_DIR.glob(f"mf_{m}_part*of{args.nshards}.pt")):
            maps.update(torch.load(f, weights_only=False))
    else:
        from wmloc.decoders import load_wrapper
        wrapper = load_wrapper(m, dev)
        rng = np.random.default_rng(C.SEED + args.shard)
        todo = (train + test)[args.shard::args.nshards]
        print(f"{m}: matched filter on {len(todo)} images (shard {args.shard}/{args.nshards})", flush=True)
        maps = {}
        for i, r in enumerate(todo):
            maps[r.name] = mf_map(r.image, rects[r.name]["rect"], r.bits, wrapper, rng, hp=args.hp, win=args.win, method=m)
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        if args.nshards > 1:
            torch.save(maps, C.CACHE_DIR / f"mf_{m}_part{args.shard}of{args.nshards}.pt")
            print(f"saved part {args.shard}/{args.nshards}")
            return
    missing = [r.name for r in train + test if r.name not in maps]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    torch.save(maps, C.CACHE_DIR / f"mf_{m}.pt")

    # signal check on development images: is the map higher on watermarked cells inside R?
    names = [r.name for r in train]
    gts = np.stack([r.mask128 for r in train])
    from scripts.step3f_seeded_boxes import rect_mask
    R = {n: rect_mask(rects[n]["rect"]) for n in maps}
    dev_i = [i for i, n in enumerate(names) if folds[n] != C.HOLDOUT_FOLD]
    pos, neg = [], []
    for i in dev_i:
        inside = R[names[i]].bool().numpy()
        v = maps[names[i]].numpy()[inside]
        g = gts[i][inside].astype(bool)
        pos += v[g].tolist()
        neg += v[~g].tolist()
    if pos and neg:
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort() + 1.0
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        print(f"{m}: inside the rectangle, map mean watermarked {np.mean(pos):+.4f} vs clean {np.mean(neg):+.4f}, "
              f"AUC {auc:.3f} (0.5 = no signal)")

    # tag `mf`: per-fold logistic on [map, rectangle, average mask, current best mask] (soft input)
    from scripts.step3f_seeded_boxes import DEFAULT_SEED, load_seed
    seed = load_seed(m, DEFAULT_SEED[m])

    def feats(n, prior):
        return torch.stack([maps[n], R[n], torch.from_numpy(prior), seed[n]]).float()

    oof = torch.zeros(len(train), C.GRID, C.GRID)
    for k in range(C.N_FOLDS):
        fit = [i for i, n in enumerate(names) if folds[n] != k and folds[n] != C.HOLDOUT_FOLD]
        prior = gts[fit].mean(0).astype(np.float32)
        model = heat.fit_logistic(torch.cat([heat.cells(feats(names[i], prior)) for i in fit]),
                                  np.concatenate([gts[i].reshape(-1) for i in fit]))
        for i, n in enumerate(names):
            if folds[n] == k:
                oof[i] = heat.predict_logistic(model, heat.cells(feats(n, prior))).reshape(C.GRID, C.GRID)
    hold = np.array([folds[n] == C.HOLDOUT_FOLD for n in names])
    for label, sel in (("dev", ~hold), ("holdout", hold)):
        s = metric.score([(p >= 0.5).astype(np.uint8) for p in oof.numpy()[sel]], gts[sel])
        print(f"{m} mf model {label}: mIoU@0.5 {s:.4f} (untuned; Step 4 tunes and compares)")
    torch.save({"names": names, "probs": oof.half(), "gts": torch.from_numpy(gts),
                "folds": [folds[n] for n in names]}, C.CACHE_DIR / f"oof_mf_{m}.pt")
    prior = gts[~hold].mean(0).astype(np.float32)
    fit = [i for i in range(len(names)) if not hold[i]]
    model = heat.fit_logistic(torch.cat([heat.cells(feats(names[i], prior)) for i in fit]),
                              np.concatenate([gts[i].reshape(-1) for i in fit]))
    probs = torch.stack([heat.predict_logistic(model, heat.cells(feats(r.name, prior))).reshape(C.GRID, C.GRID)
                         for r in test])
    torch.save({"names": [r.name for r in test], "probs": probs.half()}, C.CACHE_DIR / f"test_mf_{m}.pt")
    print("saved tag mf")


if __name__ == "__main__":
    main()
