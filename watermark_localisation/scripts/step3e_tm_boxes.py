"""Step 3e: Adobe's TrustMark box finder (NRGMark detector, `localize()`) as a TrustMark localiser.

Two parts, because the box finder needs a NEWER `trustmark` than the starter kit's pinned 0.8.0:
  --detect  run with PYTHONPATH=~/pylib_tm (newer trustmark): localize(return_all=True) on all 510
            TrustMark images -> cache/tmloc_trustmark.pt {name: [[x1, y1, x2, y2] in 0..1, ...]}
  --build   normal environment (kit decoder): decode every candidate box, keep the one whose bits
            match the known message best, and write two Step 4 tags:
              tmbox   = that box as the mask (fallback: stageAh probabilities when no box)
              tmboxAh = stageAh probabilities inside that box, 0 outside (fallback: stageAh)
            No training, so every train image is an honest test. Prints diagnostics by edit type.
Run: PYTHONPATH=~/pylib_tm python -m scripts.step3e_tm_boxes --detect
     python -m scripts.step3e_tm_boxes --build
"""
import argparse

import numpy as np
import torch

from wmloc import config as C
from wmloc import data, metric

M = "trustmark"


def detect(dev):
    from PIL import Image
    from trustmark import TrustMark            # the NEWER package from ~/pylib_tm
    tm = TrustMark(use_ECC=False, verbose=False, secret_len=100, model_type="Q",
                   loadBBoxDetector=True, device=dev)
    recs = data.by_method(data.cached_split("train"), M) + data.by_method(data.cached_split("test"), M)
    out, found = {}, 0
    for i, r in enumerate(recs):
        boxes = tm.localize(Image.fromarray(r.image), return_all=True) or []
        out[r.name] = [[float(v) for v in b] for b in boxes]
        found += bool(out[r.name])
        if i % 50 == 0:
            print(f"  {i}/{len(recs)}  example: {out[r.name][:2]}", flush=True)
    torch.save(out, C.CACHE_DIR / "tmloc_trustmark.pt")
    print(f"saved {C.CACHE_DIR / 'tmloc_trustmark.pt'}: boxes found on {found}/{len(recs)} images")


def to_pixels(box):
    """[x1, y1, x2, y2] in 0..1 -> (y0, x0, y1, x1) pixels, clipped, at least 16 px."""
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) * C.IMG for v in box)
    y0, y1 = int(min(y1, y2)), int(max(y1, y2))
    x0, x1 = int(min(x1, x2)), int(max(x1, x2))
    if y1 - y0 < 16 or x1 - x0 < 16:
        return None
    return y0, x0, y1, x1


def cell_mask(b) -> torch.Tensor:
    m = torch.zeros(C.GRID, C.GRID)
    y0, x0, y1, x1 = b
    m[y0 // C.CELL:-(-y1 // C.CELL), x0 // C.CELL:-(-x1 // C.CELL)] = 1
    return m


@torch.no_grad()
def build(dev):
    from wmloc.decoders import DecoderAdapter, to_scores
    from wmloc.heat import match_gap
    train_all = data.cached_split("train")
    folds = data.make_folds(train_all)
    train = data.by_method(train_all, M)
    test = data.by_method(data.cached_split("test"), M)
    det = torch.load(C.CACHE_DIR / "tmloc_trustmark.pt", weights_only=False)
    oofA = torch.load(C.CACHE_DIR / f"oof_stageAh_{M}.pt", weights_only=False)
    testA = torch.load(C.CACHE_DIR / f"test_stageAh_{M}.pt", weights_only=False)
    probA = dict(zip(oofA["names"], oofA["probs"].float()))
    probA.update(zip(testA["names"], testA["probs"].float()))
    adapter = DecoderAdapter(M, device=dev).to(dev).eval()

    res = {}
    for i, r in enumerate(train + test):
        x = data.image_to_tensor(r.image)[None].to(dev)
        bits = torch.from_numpy(r.bits).to(dev)
        boxes = [b for b in (to_pixels(b) for b in det.get(r.name, [])) if b is not None]
        pA = probA[r.name]
        if boxes:
            gaps = []
            for y0, x0, y1, x1 in boxes:
                _, _, out = adapter(x[..., y0:y1, x0:x1])
                gaps.append(float(match_gap(to_scores(out, C.NET_OUTPUT[M], len(r.bits)), bits)[0]))
            top, best = cell_mask(boxes[0]), cell_mask(boxes[int(np.argmax(gaps))])
            res[r.name] = {"top": top, "tmbox": best, "tmboxAh": pA * best, "found": True}
        else:
            res[r.name] = {"top": (pA >= 0.5).float(), "tmbox": pA, "tmboxAh": pA, "found": False}
        if i % 50 == 0:
            print(f"  {i}/{len(train) + len(test)}", flush=True)

    # diagnostics on train (no training happened, so all train images are honest)
    def show(label, recs):
        types = [data.edit_type(r.mask128) for r in recs]
        rows = {"stageAh@0.5": [], "top box": [], "best box": [], "box ∩ stageAh": []}
        for r in recs:
            g, d, pA = r.mask128, res[r.name], probA[r.name]
            rows["stageAh@0.5"].append(metric.miou_image((pA >= 0.5).numpy(), g))
            rows["top box"].append(metric.miou_image((d["top"] >= 0.5).numpy(), g))
            rows["best box"].append(metric.miou_image((d["tmbox"] >= 0.5).numpy(), g))
            rows["box ∩ stageAh"].append(metric.miou_image((d["tmboxAh"] >= 0.5).numpy(), g))
        found = np.mean([res[r.name]["found"] for r in recs])
        print(f"\n{label}: {len(recs)} images, box found on {found:.0%}   (mean | rect | holes | object)")
        t = np.array(types)
        for k, v in rows.items():
            v = np.array(v)
            parts = [f"{v[t == e].mean():.3f}" if (t == e).any() else "  -  " for e in ("rect", "holes", "object")]
            print(f"  {k:14s} {v.mean():.3f} | " + " | ".join(parts))

    show("train, non-holdout", [r for r in train if folds[r.name] != C.HOLDOUT_FOLD])
    show("train, holdout", [r for r in train if folds[r.name] == C.HOLDOUT_FOLD])

    names = [r.name for r in train]
    gts = torch.from_numpy(np.stack([r.mask128 for r in train]))
    for tag in ("tmbox", "tmboxAh"):
        torch.save({"names": names, "probs": torch.stack([res[n][tag] for n in names]).half(), "gts": gts,
                    "folds": [folds[n] for n in names]}, C.CACHE_DIR / f"oof_{tag}_{M}.pt")
        torch.save({"names": [r.name for r in test],
                    "probs": torch.stack([res[r.name][tag] for r in test]).half()},
                   C.CACHE_DIR / f"test_{tag}_{M}.pt")
    print("saved tags tmbox, tmboxAh (compare in Step 4)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--method", default=M, choices=[M], help="TrustMark only (ignored)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.detect:
        detect(dev)
    if args.build:
        build(dev)


if __name__ == "__main__":
    main()
