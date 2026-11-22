"""Step 0: prove the whole submission pipeline end to end.

1. Loads train + test records through wmloc.data.load_split (adapt it first).
2. Checks our 512->128 downsampling matches the provided 128x128 masks.
3. Prints trivial baselines on train (all 0, all 1) with the exact metric.
4. Builds a 'prior' mask per method (average train mask, threshold tuned on train),
   predicts it for every test image, validates and saves the .pt submission.
5. Optionally submits it (--submit), after wmloc.submission.submit is adapted.

Run:  python -m scripts.step0_prior_submission [--submit]
"""
import argparse

import numpy as np

from wmloc import config as C
from wmloc import data, metric, submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    train = data.cached_split("train")
    test = data.cached_split("test")
    print(f"train {len(train)}  test {len(test)}")
    for m in C.METHODS:
        tr, te = data.by_method(train, m), data.by_method(test, m)
        lens = sorted({len(r.bits) for r in tr + te})
        print(f"  {m:11s} train {len(tr):4d} test {len(te):4d} message lengths {lens}")
    unknown = {r.method for r in train + test} - set(C.METHODS)
    assert not unknown, f"method names not in config.METHODS: {unknown}"

    # --- 2. downsampling check
    bad = [r.name for r in train if r.mask512 is not None
           and not np.array_equal(metric.downsample_mask(r.mask512), r.mask128)]
    print(f"downsample mismatches: {len(bad)} / {len(train)}"
          + ("  <- use the starter kit's function instead" if bad else "  (rule confirmed)"))

    # --- 3. trivial baselines + mask stats
    gts = [r.mask128 for r in train]
    print(f"all-0 score {metric.score([np.zeros_like(g) for g in gts], gts):.4f}   "
          f"all-1 score {metric.score([np.ones_like(g) for g in gts], gts):.4f}")
    for m in C.METHODS:
        rs = data.by_method(train, m)
        areas = np.array([r.mask128.mean() for r in rs])
        types = [data.edit_type(r.mask128) for r in rs]
        counts = {t: types.count(t) for t in sorted(set(types))}
        print(f"  {m:11s} area share median {np.median(areas):.3f} "
              f"[p10 {np.percentile(areas,10):.3f}, p90 {np.percentile(areas,90):.3f}] types {counts}")

    # --- 4. prior baseline
    names, masks = [], []
    for m in C.METHODS:
        rs = data.by_method(train, m)
        prior = np.mean([r.mask128 for r in rs], axis=0)
        best = max(((metric.score([(prior >= t).astype(np.uint8)] * len(rs),
                                  [r.mask128 for r in rs]), t)
                    for t in np.linspace(0.05, 0.95, 19)))
        pred = (prior >= best[1]).astype(np.uint8)
        if pred.sum() == 0:
            pred[prior >= prior.max()] = 1
        print(f"  {m:11s} prior baseline train score {best[0]:.4f} at threshold {best[1]:.2f}")
        for r in data.by_method(test, m):
            names.append(r.name)
            masks.append(pred)

    sub = submission.build(names, np.stack(masks))
    path = submission.save(sub, C.SUB_DIR / "step0_prior.pt",
                           expected_names=[r.name for r in test])
    if args.submit:
        submission.submit(path)


if __name__ == "__main__":
    main()
