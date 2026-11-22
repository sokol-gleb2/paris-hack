"""Step 3b: known-message heatmaps for one method (no network training) + the shuffle check.

Writes cache/heat_<m>.pt (maps for train + test; inputs for step3_stage_a --heat) and the
"heat" model: cache/oof_heat_<m>.pt + cache/test_heat_<m>.pt (maps -> probability with a
tiny logistic fit per fold; Step 4 tunes and compares it like any other tag).

Run: python -m scripts.step3b_heatmaps --method pixelseal [--sizes 256,128] [--tm-box]
"""
import argparse

import numpy as np
import torch

from wmloc import config as C
from wmloc import data, heat, metric
from wmloc.decoders import DecoderAdapter


def auc(pos, neg) -> float:
    """Chance that a random watermarked crop has a bigger gap than a random clean crop."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--sizes", default=",".join(str(s) for s in C.HEAT_SIZES))
    ap.add_argument("--tm-box", action="store_true",
                    help="TrustMark only: add Adobe's box finder as a channel (ask the organisers first)")
    ap.add_argument("--tile-agree", action="store_true",
                    help="add the agreement map from 256-px tiles at native resolution (Pixel Seal)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = args.method
    sizes = [int(s) for s in args.sizes.split(",")]

    train_all = data.cached_split("train")
    folds = data.make_folds(train_all)
    train = data.by_method(train_all, m)
    test = data.by_method(data.cached_split("test"), m)
    adapter = DecoderAdapter(m, device=dev).to(dev).eval()
    tm = None
    if args.tm_box and m == "trustmark":
        from trustmark import TrustMark                                     # ADAPT: optional package
        tm = TrustMark(model_type="Q", loadBBoxDetector=True, device=dev)  # ADAPT: the kit's variant

    channels = [f"crop{s}" for s in sizes]
    channels += ["agree"] if adapter.agree_tap is not None else []
    channels += ["tm_box"] if tm is not None else []
    tile = args.tile_agree and adapter.agree_tap is not None
    channels += ["tileagree"] if tile else []
    maps, check = {}, {s: ([], []) for s in sizes}
    for i, r in enumerate(train + test):
        x = data.image_to_tensor(r.image)[None].to(dev)
        bits = torch.from_numpy(r.bits).to(dev)
        chans = []
        for s in sizes:
            boxes, gaps = heat.crop_gaps(adapter, m, x, bits, s)
            chans.append(heat.paste(boxes, gaps, s))
            if r.mask512 is not None and folds[r.name] != C.HOLDOUT_FOLD:   # shuffle-check data
                for (y0, x0), g in zip(boxes, gaps.tolist()):
                    frac = r.mask512[y0:y0 + s, x0:x0 + s].mean()
                    if frac >= 0.5:
                        check[s][0].append(g)
                    elif frac <= 0.05:
                        check[s][1].append(g)
        if adapter.agree_tap is not None:
            with torch.no_grad():
                _, agree, _ = adapter(x, bits[None])
            chans.append(heat.combine(agree, None)[0, 0].cpu())
        if tm is not None:
            from PIL import Image
            chans.append(heat.box_map(tm.localize(Image.fromarray(r.image))))  # ADAPT: check input type
        if tile:
            chans.append(heat.tile_agreement(adapter, x, bits))
        maps[r.name] = torch.stack(chans).half()
        if i % 50 == 0:
            print(f"  {i}/{len(train) + len(test)}", flush=True)

    # normalisation statistics from train images outside the holdout fold
    st = torch.stack([maps[r.name].float() for r in train if folds[r.name] != C.HOLDOUT_FOLD])
    mean, std = st.mean(dim=(0, 2, 3)), st.std(dim=(0, 2, 3)).clamp(min=1e-6)
    torch.save({"maps": maps, "mean": mean, "std": std, "sizes": sizes, "channels": channels}, heat.path(m))
    print(f"saved {heat.path(m)}  channels {channels}")

    print("shuffle check (gap = match with the real message minus match with shuffled ones):")
    for s in sizes:
        pos, neg = check[s]
        mp = np.mean(pos) if pos else float("nan")
        mn = np.mean(neg) if neg else float("nan")
        print(f"  crop {s:3d}px: watermarked n={len(pos):5d} mean gap {mp:+.4f} | "
              f"clean n={len(neg):5d} mean gap {mn:+.4f} | AUC {auc(pos, neg):.3f} (0.5 = no signal)")

    # the "heat" model: per-fold logistic fit on cells; the holdout is never used for fitting
    names = [r.name for r in train]
    gts = np.stack([r.mask128 for r in train])
    oof = torch.zeros(len(train), C.GRID, C.GRID)
    for k in range(C.N_FOLDS):
        fit = [i for i, n in enumerate(names) if folds[n] != k and folds[n] != C.HOLDOUT_FOLD]
        val = [i for i, n in enumerate(names) if folds[n] == k]
        model = heat.fit_logistic(torch.cat([heat.cells(maps[names[i]]) for i in fit]),
                                  np.concatenate([gts[i].reshape(-1) for i in fit]))
        for i in val:
            oof[i] = heat.predict_logistic(model, heat.cells(maps[names[i]])).reshape(C.GRID, C.GRID)
    model = heat.fit_logistic(torch.cat([heat.cells(maps[n]) for n in names]), gts.reshape(-1))
    test_probs = torch.stack([heat.predict_logistic(model, heat.cells(maps[r.name])).reshape(C.GRID, C.GRID)
                              for r in test])
    torch.save({"names": names, "probs": oof.half(), "gts": torch.from_numpy(gts),
                "folds": [folds[n] for n in names]}, C.CACHE_DIR / f"oof_heat_{m}.pt")
    torch.save({"names": [r.name for r in test], "probs": test_probs.half()},
               C.CACHE_DIR / f"test_heat_{m}.pt")
    print(f"{m} heat model: OOF mIoU@0.5 {metric.score(oof.numpy() >= 0.5, gts):.4f} "
          "(untuned; Step 4 tunes and compares it)")


if __name__ == "__main__":
    main()
