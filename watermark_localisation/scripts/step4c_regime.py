"""Step 4c: two-regime post-processing.

Finding (train metadata): exactly half of each method's photos are "full watermark" (whole image watermarked,
one region restored to the clean original; mask area ~0.90); the other half are partial (outpainting rectangle
or transplanted object, much smaller). One cutoff per method fits neither well.

1. Regime detector (no labels needed at test time): decode the WHOLE photo and score the known-message match
   (heat.match_gap). Full-watermark photos read well. Cutoff chosen on development photos (true flag from
   metadata), reported on the holdout.
2. Separate post-processing per predicted regime, tuned on development photos: plain cutoff (0.05-0.95) or
   "keep the top-q share of cells" (q 0.5-0.98, adaptive per photo), with the never-empty rule.
3. Prints dev/holdout mIoU: one setting per method (as Step 4) vs two regimes; writes a submission.
Base map per method = average of the listed tags' probabilities (defaults = current best choices).
Run: python -m scripts.step4c_regime [--pixelseal stageA] [--trustmark stageAh] [--stegastamp prior,heat,stageAh,box,pix]
"""
import argparse
import json

import numpy as np
import torch

from wmloc import config as C
from wmloc import data, metric, postprocess, submission
from wmloc.heat import match_gap

DEFAULT_TAGS = {"pixelseal": "stageA", "trustmark": "stageAh", "stegastamp": "prior,heat,stageAh,box,pix"}
THRS = [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]
QS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98]


def load_probs(method, tags):
    """{name: averaged (128,128) probability map} over the tags, train (OOF) + test."""
    acc = {}
    for t in tags:
        for kind in ("oof", "test"):
            d = torch.load(C.CACHE_DIR / f"{kind}_{t}_{method}.pt", weights_only=False)
            for n, p in zip(d["names"], d["probs"].float().numpy()):
                acc.setdefault(n, []).append(p)
    return {n: np.mean(v, axis=0) for n, v in acc.items()}


def apply(p, rule):
    kind, v = rule
    if kind == "thr":
        return postprocess.postprocess(p, thr=v, never_empty=True)
    k = int(round(v * p.size))                    # top-q share of cells
    m = np.zeros(p.size, np.uint8)
    m[np.argsort(p.ravel())[::-1][:max(k, 1)]] = 1
    return m.reshape(p.shape)


def best_rule(probs, gts):
    cands = [("thr", t) for t in THRS] + [("top", q) for q in QS]
    scores = [metric.score([apply(p, r) for p in probs], gts) for r in cands]
    j = int(np.argmax(scores))
    return cands[j], scores[j]


@torch.no_grad()
def whole_gaps(method, recs, dev):
    from wmloc.decoders import DecoderAdapter, to_scores
    adapter = DecoderAdapter(method, device=dev).to(dev).eval()
    out = {}
    for i in range(0, len(recs), 16):
        part = recs[i:i + 16]
        x = torch.stack([data.image_to_tensor(r.image) for r in part]).to(dev)
        _, _, o = adapter(x)
        s = to_scores(o, C.NET_OUTPUT[method], len(part[0].bits))
        for j, r in enumerate(part):
            out[r.name] = float(match_gap(s[j:j + 1], torch.from_numpy(r.bits))[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    for m in C.METHODS:
        ap.add_argument(f"--{m}", default=DEFAULT_TAGS[m], help="comma-separated tags averaged into the base map")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_all = data.cached_split("train")
    test_all = data.cached_split("test")
    folds = data.make_folds(train_all)
    flag = {r["image_name"]: bool(r.get("full_watermark")) for r in
            json.loads((C.DATA_DIR / "train" / "metadata.json").read_text())}

    names_all, masks_all, summary = [], [], {}
    for m in C.METHODS:
        tags = getattr(args, m).split(",")
        probs = load_probs(m, tags)
        tr, te = data.by_method(train_all, m), data.by_method(test_all, m)
        gap = whole_gaps(m, tr + te, dev)
        devn = [r for r in tr if folds[r.name] != C.HOLDOUT_FOLD]
        hold = [r for r in tr if folds[r.name] == C.HOLDOUT_FOLD]

        # 1. regime detector: whole-image match gap > cutoff -> "full watermark"
        g_dev = np.array([gap[r.name] for r in devn]); y_dev = np.array([flag[r.name] for r in devn])
        cut = max(np.unique(g_dev), key=lambda c: np.mean((g_dev > c) == y_dev) if c < g_dev.max() else 0)
        acc = lambda rs: np.mean([(gap[r.name] > cut) == flag[r.name] for r in rs])   # noqa: E731
        print(f"\n{m} ({'+'.join(tags)}): regime detector cutoff {cut:.4f}, accuracy dev {acc(devn):.3f} "
              f"holdout {acc(hold):.3f}")

        # 2. one setting (baseline) vs two regimes, tuned on dev with PREDICTED regimes
        one, s_one = best_rule([probs[r.name] for r in devn], [r.mask128 for r in devn])
        rules = {}
        for reg in (True, False):
            rs = [r for r in devn if (gap[r.name] > cut) == reg]
            rules[reg] = best_rule([probs[r.name] for r in rs], [r.mask128 for r in rs])[0] if rs else one

        def predict(r):
            return apply(probs[r.name], rules[gap[r.name] > cut])

        for label, rs in (("dev", devn), ("holdout", hold)):
            s1 = metric.score([apply(probs[r.name], one) for r in rs], [r.mask128 for r in rs])
            s2 = metric.score([predict(r) for r in rs], [r.mask128 for r in rs])
            print(f"  {label:7s} one setting {s1:.4f} {one} | two regimes {s2:.4f} "
                  f"(full {rules[True]}, partial {rules[False]})")
        for reg, label in ((True, "full"), (False, "partial")):
            rs = [r for r in hold if flag[r.name] == reg]
            print(f"  holdout true-{label:7s} n={len(rs):2d} one {metric.score([apply(probs[r.name], one) for r in rs], [r.mask128 for r in rs]):.4f}"
                  f" | two {metric.score([predict(r) for r in rs], [r.mask128 for r in rs]):.4f}")
        h1 = metric.score([apply(probs[r.name], one) for r in hold], [r.mask128 for r in hold])
        h2 = metric.score([predict(r) for r in hold], [r.mask128 for r in hold])
        summary[m] = {"tags": tags, "cut": cut, "rules": {str(k): v for k, v in rules.items()},
                      "holdout_one": h1, "holdout_two": h2, "n_test_full": int(sum(gap[r.name] > cut for r in te))}
        for r in te:
            names_all.append(r.name)
            masks_all.append(predict(r))

    print("\nsummary (holdout):")
    for m, s in summary.items():
        print(f"  {m:11s} one setting {s['holdout_one']:.4f} -> two regimes {s['holdout_two']:.4f} "
              f"(test photos called full: {s['n_test_full']}/340)")
    print(f"  mean: one {np.mean([s['holdout_one'] for s in summary.values()]):.4f} -> "
          f"two {np.mean([s['holdout_two'] for s in summary.values()]):.4f}")
    (C.SUB_DIR / "sub_regime.json").write_text(json.dumps(summary, indent=2, default=str))
    sub = submission.build(names_all, np.stack(masks_all))
    submission.save(sub, C.SUB_DIR / "sub_regime.pt", expected_names=[r.name for r in test_all])


if __name__ == "__main__":
    main()
