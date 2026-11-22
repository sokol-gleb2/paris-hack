"""Step 4 / Step 7: tune post-processing, choose a model per method on the holdout fold,
predict test, write the submission.

Run: python -m scripts.step4_tune_and_predict --tags stageA[,heat,stageAh,stageB0,stageB,stageB2] [--submit]
Per method:
  1. load the OOF predictions of every listed tag that exists for this method (missing ones
     are skipped with a note, e.g. Stage B did not finish for one method);
  2. candidates = each tag alone + the average of all available tags;
  3. tune post-processing on the tuning images (all folds except C.HOLDOUT_FOLD);
  4. score every candidate on the holdout fold;
  5. keep the first tag (the baseline) unless a candidate beats it there by --margin;
  6. predict the test images with the chosen candidate.
"""
import argparse
import json

import numpy as np
import torch

from wmloc import config as C
from wmloc import data, heat, metric, postprocess, submission
from wmloc.decoders import DecoderAdapter
from wmloc.head import LocHead


MIN_TUNE = 20   # fewest shared tuning images for a candidate
MAX_ENS = 5     # largest top-k ensemble


def oof_path(tag, method):
    return C.CACHE_DIR / f"oof_{tag}_{method}.pt"


def load_heads(tag, method, dev):
    paths = sorted(C.CKPT_DIR.glob(f"{tag}_{method}_fold*.pt")) or [C.CKPT_DIR / f"{tag}_{method}.pt"]
    metas = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    heads = []
    for meta in metas:
        h = LocHead(meta["channels"], extra_ch=meta["extra_ch"],
                    image_branch=meta.get("image_branch", False)).to(dev).eval()
        h.load_state_dict(meta["state"])
        heads.append(h)
    return heads, metas[0]


@torch.no_grad()
def test_probs(tag, method, recs, dev):
    path = C.CACHE_DIR / f"test_{tag}_{method}.pt"
    if path.exists():
        d = torch.load(path, weights_only=False)
        if d["names"] == [r.name for r in recs]:
            return d["probs"].float().numpy()
    if tag == "heat":
        raise FileNotFoundError(f"{path} missing or stale: re-run scripts.step3b_heatmaps for {method}")
    heads, meta = load_heads(tag, method, dev)
    adapter = DecoderAdapter(method, device=dev).to(dev).eval()
    if meta.get("decoder_ckpt"):
        adapter.net.load_state_dict(torch.load(meta["decoder_ckpt"], map_location=dev))
    hstore = heat.load(method) if meta.get("heat") else None
    views = [lambda t: t] + ([lambda t: t.flip(-1)] if C.FLIP_SAFE[method] else [])
    out = []
    for r in recs:
        x = data.image_to_tensor(r.image)[None].to(dev)
        bits = torch.from_numpy(r.bits)[None].to(dev)
        hm = None if hstore is None else heat.get(hstore, r.name)[None].to(dev)
        ps = []
        for f in views:                   # f flips the input; applying it again flips the prediction back
            xv = f(x)
            feats, agree, _ = adapter(xv, bits)
            extra = heat.combine(agree, None if hm is None else f(hm))
            for h in heads:
                ps.append(f(torch.sigmoid(h(feats, extra, xv))))
        out.append(torch.stack(ps).mean(0)[0].half().cpu())
    probs = torch.stack(out)
    torch.save({"names": [r.name for r in recs], "probs": probs}, path)
    return probs.float().numpy()


def print_breakdown(preds, gts, indent="    "):
    for k, (n, s) in metric.breakdown(preds, gts).items():
        print(f"{indent}{k:20s} n={n:3d}  mIoU {s:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True,
                    help="comma-separated; the first is the baseline, e.g. stageA,heat,stageAh")
    ap.add_argument("--margin", type=float, default=0.01, help="holdout gain needed to replace the baseline")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    tags = args.tags.split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = data.cached_split("train")
    test = data.cached_split("test")
    folds = data.make_folds(train)

    names_all, masks_all, chosen = [], [], {}
    for m in C.METHODS:
        have = [t for t in tags if oof_path(t, m).exists()]
        for t in tags:
            if t not in have:
                print(f"  {m}: skipping {t} (no {oof_path(t, m).name})")
        if not have:
            raise FileNotFoundError(f"{m}: none of the tags {tags} exists; run Step 3 first")
        oofs = {t: torch.load(oof_path(t, m), weights_only=False) for t in have}
        names = oofs[have[0]]["names"]
        gts = oofs[have[0]]["gts"].numpy()
        fold_of = np.array([folds[n] for n in names])
        for t in have:
            assert oofs[t]["names"] == names, f"{m}: image order differs between {have[0]} and {t}"
            saved = oofs[t].get("folds")
            assert saved is None or list(saved) == fold_of.tolist(), \
                f"{m}/{t}: fold ids changed since this tag was trained (groups.json?): retrain it"
        hold = fold_of == C.HOLDOUT_FOLD
        if not hold.any():
            print(f"  {m}: no holdout fold, choosing on tuning scores (optimistic)")

        cands = {t: (oofs[t]["probs"].float().numpy(), [t]) for t in have}
        if len(have) > 1:
            cands["+".join(have)] = (np.mean([c[0] for c in cands.values()], axis=0), list(have))
        results = {}

        def tune_mask(used):
            ok = ~hold
            for t in used:   # a tag trained partly on real images tunes only on its clean images
                if oofs[t].get("tune_names") is not None:
                    ok = ok & np.isin(names, oofs[t]["tune_names"])
            return ok

        def score_cand(cname, probs, used):
            ok = tune_mask(used)
            if ok.sum() < MIN_TUNE:   # e.g. pix (tuned on fold 3) + dense (fold 0): no shared clean images
                print(f"  {m} / {cname:28s} skipped: only {int(ok.sum())} shared tuning images")
                return False
            score, params = postprocess.tune(probs[ok], gts[ok])
            h_score = postprocess.evaluate(probs[hold], gts[hold], **params) if hold.any() else score
            results[cname] = (h_score, score, params, used)
            cands[cname] = (probs, used)
            print(f"  {m} / {cname:28s} tune {score:.4f} (n={int(ok.sum())})  "
                  f"holdout {h_score:.4f} (n={int(hold.sum())})  {params}")
            return True

        for cname, (probs, used) in list(cands.items()):
            if not score_cand(cname, probs, used):
                del cands[cname]

        # ensembles: walk the single tags from best to worst (mean of tune + holdout) and average
        # the top 2..MAX_ENS that share at least MIN_TUNE tuning images (e.g. the dense-decoder family)
        singles = sorted([t for t in have if t in results],
                         key=lambda t: results[t][0] + results[t][1], reverse=True)
        group = singles[:1]
        for t in singles[1:]:
            if len(group) >= MAX_ENS:
                break
            if tune_mask(group + [t]).sum() < MIN_TUNE:
                continue
            group = group + [t]
            cname = "top" + str(len(group)) + ":" + "+".join(group)
            score_cand(cname, np.mean([cands[g][0] for g in group], axis=0), list(group))

        base = have[0]
        # rank by the mean of tune and holdout scores: with many candidates, the max on 34
        # holdout photos alone favours luck
        rank = {k: (results[k][0] + results[k][1]) / 2 for k in results}
        best = max(rank, key=rank.get)
        if best != base and (rank[best] < rank[base] + args.margin or results[best][0] < results[base][0]):
            print(f"  {m}: {best} does not beat {base} by {args.margin} (mean of tune+holdout) "
                  f"or loses on the holdout -> keeping {base}")
            best = base
        h_score, score, params, used = results[best]
        chosen[m] = {"candidate": best, "holdout_miou": h_score, "tune_miou": score, "params": params}
        if hold.any():
            print(f"  {m}: chosen {best}; holdout by edit type and size:")
            print_breakdown([postprocess.postprocess(p, **params) for p in cands[best][0][hold]], gts[hold])

        recs = data.by_method(test, m)
        tp = np.mean([test_probs(t, m, recs, dev) for t in used], axis=0)
        for r, p in zip(recs, tp):
            names_all.append(r.name)
            masks_all.append(postprocess.postprocess(p, **params))

    print("\nchosen per method:")
    for m, c in chosen.items():
        print(f"  {m:11s} {c['candidate']:28s} holdout {c['holdout_miou']:.4f}  tune {c['tune_miou']:.4f}")
    print(f"  mean holdout mIoU {np.mean([c['holdout_miou'] for c in chosen.values()]):.4f} "
          "(test has equal counts per method)")

    tag = "_".join(tags)
    (C.SUB_DIR / f"sub_{tag}.json").write_text(json.dumps(chosen, indent=2))
    sub = submission.build(names_all, np.stack(masks_all))
    path = submission.save(sub, C.SUB_DIR / f"sub_{tag}.pt", expected_names=[r.name for r in test])
    if args.submit:
        submission.submit(path)


if __name__ == "__main__":
    main()
