"""Step 3: Stage A - frozen decoder, train LocHead on cached features, 5-fold CV.

Run: python -m scripts.step3_stage_a --method pixelseal [--memorise 20] [--heat] [--recache]
     python -m scripts.step3_stage_a --method pixelseal --tag stageB2 --decoder-from stageB
--heat adds the Step 3b heatmaps as extra inputs (default tag then: stageAh).
The holdout fold (C.HOLDOUT_FOLD) is trained last with an epoch fixed in advance (median
best epoch of the other folds), so the holdout never picks anything.
"""
import argparse
import math
import random

import numpy as np
import torch

from wmloc import config as C
from wmloc import data, heat, metric
from wmloc.decoders import DecoderAdapter
from wmloc.head import LocHead, seg_loss

FLIPS = {0: lambda t: t, 1: lambda t: t.flip(-1), 2: lambda t: t.flip(-2)}


@torch.no_grad()
def cache_features(method, recs, adapter, dev, path, recache=False, hstore=None):
    """Per image and flip id: list of fp16 feature maps (CPU), extra inputs (128x128), mask."""
    if path.exists() and not recache:
        return torch.load(path, weights_only=False)
    flips = [0, 1, 2] if C.FLIP_SAFE[method] else [0]
    store = {}
    for i, r in enumerate(recs):
        x = data.image_to_tensor(r.image)[None].to(dev)
        bits = torch.from_numpy(r.bits)[None].to(dev)
        y = torch.from_numpy(r.mask128) if r.mask128 is not None else None
        hm = None if hstore is None else heat.get(hstore, r.name)[None].to(dev)
        for f in flips:
            feats, agree, _ = adapter(FLIPS[f](x), bits)
            extra = heat.combine(agree, None if hm is None else FLIPS[f](hm))
            store[(r.name, f)] = {
                "feats": [t[0].half().cpu() for t in feats],
                "extra": None if extra is None else extra[0].half().cpu(),
                "y": None if y is None else FLIPS[f](y),
            }
        if i % 20 == 0:
            print(f"  cached {i}/{len(recs)}", flush=True)
    torch.save(store, path)
    return store


def batchify(store, keys, dev):
    items = [store[k] for k in keys]
    feats = [torch.stack([it["feats"][j] for it in items]).to(dev).float()
             for j in range(len(items[0]["feats"]))]
    extra = None if items[0]["extra"] is None else \
        torch.stack([it["extra"] for it in items]).to(dev).float()
    y = torch.stack([it["y"] for it in items]).to(dev).float()
    return feats, extra, y


@torch.no_grad()
def predict_probs(head, store, keys, dev, bs=32):
    head.eval()
    out = []
    for s in range(0, len(keys), bs):
        feats, extra, _ = batchify(store, keys[s:s + bs], dev)
        out.append(torch.sigmoid(head(feats, extra)).cpu())
    return torch.cat(out)


def train_head(store, tr_keys, va_keys, channels, extra_ch, dev, epochs, lr, bs, log="", pick_epoch=None):
    """Train one head. Keeps the epoch with the best validation mIoU (checked every 5 epochs),
    or, with pick_epoch (holdout fold), that epoch, so the validation images pick nothing.
    Returns (head, validation mIoU of the kept epoch, kept epoch)."""
    head = LocHead(channels, extra_ch=extra_ch).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * math.ceil(len(tr_keys) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    va_y = torch.stack([store[k]["y"] for k in va_keys])
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, epochs + 1):
        head.train()
        random.shuffle(tr_keys)
        for s in range(0, len(tr_keys), bs):
            feats, extra, y = batchify(store, tr_keys[s:s + bs], dev)
            loss = seg_loss(head(feats, extra), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ep % 5 == 0 or ep == epochs:
            probs = predict_probs(head, store, va_keys, dev)
            miou = metric.miou_torch(probs >= 0.5, va_y).mean().item()
            if (pick_epoch is None and miou > best) or ep == pick_epoch:
                best, best_ep = miou, ep
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if ep % 25 == 0:
                print(f"  {log} ep {ep:3d} loss {loss.item():.4f} val mIoU {miou:.4f} "
                      f"(kept {best:.4f} @ {best_ep})", flush=True)
    head.load_state_dict(best_state)
    return head, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--memorise", type=int, default=0, help="train+eval on N images (bug test)")
    ap.add_argument("--heat", action="store_true", help="add the Step 3b heatmaps as extra inputs")
    ap.add_argument("--recache", action="store_true")
    ap.add_argument("--decoder-from", default=None, help="use the fine-tuned decoder of this tag, e.g. stageB")
    ap.add_argument("--tag", default=None, help="default: stageA, or stageAh with --heat")
    args = ap.parse_args()
    tag = args.tag or ("stageAh" if args.heat else "stageA")
    torch.manual_seed(C.SEED); random.seed(C.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = args.method

    train = data.cached_split("train")
    folds = data.make_folds(train)
    recs = data.by_method(train, m)
    adapter = DecoderAdapter(m, device=dev).to(dev).eval()
    dec_ckpt = None
    if args.decoder_from:
        dec_ckpt = str(C.CKPT_DIR / f"{args.decoder_from}_{m}_decoder.pt")
        adapter.net.load_state_dict(torch.load(dec_ckpt, map_location=dev))
    for p in adapter.parameters():
        p.requires_grad_(False)
    channels = adapter.channels()
    print(f"{m} {tag}: tapped layers {C.LAYERS[m]} channels {channels}")

    hstore = heat.load(m) if args.heat else None
    store = cache_features(m, recs, adapter, dev, C.CACHE_DIR / f"feats_{tag}_{m}.pt",
                           args.recache, hstore)
    first = store[(recs[0].name, 0)]["extra"]
    extra_ch = 0 if first is None else first.shape[0]
    flips = sorted({f for (_, f) in store})
    if not channels and extra_ch == 0:
        print(f"{m}: no feature maps and no extra inputs (StegaStamp is ONNX-only): run with --heat")
        return

    if args.memorise:
        keys = [(r.name, 0) for r in recs[: args.memorise]]
        _, best, _ = train_head(store, list(keys), keys, channels, extra_ch, dev,
                                max(args.epochs, 300), args.lr, 4, log="memorise")
        print(f"memorise test: train mIoU {best:.4f}  (should pass 0.9)")
        return

    names = [r.name for r in recs]
    oof = torch.zeros(len(recs), C.GRID, C.GRID)
    scores, best_eps = {}, []
    order = [k for k in range(C.N_FOLDS) if k != C.HOLDOUT_FOLD]
    if C.HOLDOUT_FOLD is not None:
        order.append(C.HOLDOUT_FOLD)                     # last: its epoch comes from the others
    for k in order:
        pick = None
        if k == C.HOLDOUT_FOLD:
            pick = min(args.epochs, max(5, int(5 * round(np.median(best_eps) / 5))))
        tr = [(n, f) for n in names if folds[n] != k for f in flips]
        va = [(n, 0) for n in names if folds[n] == k]
        head, best, ep = train_head(store, tr, va, channels, extra_ch, dev, args.epochs,
                                    args.lr, args.bs, log=f"fold {k}", pick_epoch=pick)
        if pick is None:
            best_eps.append(ep)
        scores[k] = best
        idx = [i for i, n in enumerate(names) if folds[n] == k]
        oof[idx] = predict_probs(head, store, va, dev)
        torch.save({"state": head.state_dict(), "channels": channels, "extra_ch": extra_ch,
                    "layers": C.LAYERS[m], "size": C.DECODER_INPUT_SIZE[m], "heat": bool(args.heat),
                    "decoder_ckpt": dec_ckpt, "image_branch": False},
                   C.CKPT_DIR / f"{tag}_{m}_fold{k}.pt")
        note = " (holdout: epoch fixed in advance)" if pick else ""
        print(f"fold {k}: val mIoU {best:.4f} at epoch {ep}{note}", flush=True)

    gts = np.stack([r.mask128 for r in recs])
    oof_score = metric.score((oof.numpy() >= 0.5), gts)
    per_fold = {k: round(v, 4) for k, v in scores.items()}
    print(f"{m} {tag}: fold mIoU {per_fold}  OOF mIoU@0.5 {oof_score:.4f}")
    torch.save({"names": names, "probs": oof.half(), "gts": torch.from_numpy(gts),
                "folds": [folds[n] for n in names]},
               C.CACHE_DIR / f"oof_{tag}_{m}.pt")


if __name__ == "__main__":
    main()
