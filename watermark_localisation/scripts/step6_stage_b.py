"""Step 6: Stage B0 (--frozen: head only) and Stage B (head + the decoder's last stage) on fake data.

Run: python -m scripts.step6_stage_b --method pixelseal [--frozen] [--hours 5.5] [--real-weight 0]
Checkpoints are picked on the real train images outside the holdout fold (or on --val-fold
with --real-weight > 0); the holdout fold is never used.
Saves checkpoints/<tag>_<m>.pt (head + meta), <tag>_<m>_decoder.pt (decoder, B only) and
cache/oof_<tag>_<m>.pt (predictions on all real train images, used by Step 4).
"""
import argparse
import io
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from wmloc import config as C
from wmloc import data, metric
from wmloc.decoders import DecoderAdapter, bits_from_output
from wmloc.head import LocHead, seg_loss


class SynthSet(Dataset):
    """Fake examples from Step 5's pack files (PNG bytes kept in memory)."""

    def __init__(self, method):
        self.items = []
        for f in sorted((C.SYNTH_DIR / method).glob("pack_*.pt")):
            try:   # another job may still be writing a pack: skip it instead of crashing
                self.items += torch.load(f, weights_only=False)
            except Exception as e:
                print(f"  skipping unreadable {f.name}: {type(e).__name__}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = np.asarray(Image.open(io.BytesIO(it["png"])).convert("RGB"))
        m512 = (np.asarray(Image.open(io.BytesIO(it["mask_png"]))) > 127).astype(np.uint8)
        return (data.image_to_tensor(img), torch.from_numpy(metric.downsample_mask(m512)),
                torch.from_numpy(data.parse_bits(it["bits"])))


class RealSet(Dataset):
    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        return data.image_to_tensor(r.image), torch.from_numpy(r.mask128), torch.from_numpy(r.bits)


def unfreeze(adapter, prefixes):
    params = []
    for name, p in adapter.net.named_parameters():
        ok = any(name.startswith(pre) for pre in prefixes)
        p.requires_grad_(ok)
        if ok:
            params.append(p)
    return params


@torch.no_grad()
def evaluate(adapter, head, recs, dev, n_bits, method):
    """Probabilities (N,128,128) for all `recs` + the decoder's bit accuracy on the first 32."""
    head.eval()
    probs, accs = [], []
    for i, r in enumerate(recs):
        x = data.image_to_tensor(r.image)[None].to(dev)
        bits = torch.from_numpy(r.bits)[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            feats, extra, out = adapter(x, bits)
            logits = head(feats, extra, x)
        probs.append(torch.sigmoid(logits.float())[0].cpu())
        if i < 32:
            accs.append((bits_from_output(out, n_bits, method).numpy()[0] == r.bits).mean())
    head.train()
    return torch.stack(probs), float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--frozen", action="store_true", help="B0: keep the whole decoder frozen")
    ap.add_argument("--tag", default=None, help="default: stageB0 with --frozen, else stageB")
    ap.add_argument("--hours", type=float, default=5.5,
                    help="stop after this long; keep it 30 min below the job's --time")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--dec-lr", type=float, default=1e-5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--real-weight", type=float, default=0.0, help="share of real images per batch")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--unfreeze", default=None,
                    help="comma-separated decoder module prefixes to train (overrides config UNFREEZE)")
    ap.add_argument("--bit-loss", type=float, default=0.0,
                    help="dense decoder: per-position bit loss inside the watermark (MaskWM-style)")
    ap.add_argument("--clean-loss", type=float, default=0.0,
                    help="dense decoder: push per-position bit scores to 0 in clean areas")
    args = ap.parse_args()
    torch.manual_seed(C.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, t0 = args.method, time.time()
    tag = args.tag or ("stageB0" if args.frozen else "stageB")

    train = data.cached_split("train")
    folds = data.make_folds(train)
    recs = data.by_method(train, m)
    n_bits = len(recs[0].bits)
    usable = [r for r in recs if folds[r.name] != C.HOLDOUT_FOLD]    # never train or pick on the holdout
    if args.real_weight > 0:
        assert args.val_fold != C.HOLDOUT_FOLD, "--val-fold must not be the holdout fold"
        sel_names = {r.name for r in usable if folds[r.name] == args.val_fold}
        real_tr = [r for r in usable if folds[r.name] != args.val_fold]
    else:
        sel_names, real_tr = {r.name for r in usable}, []
    sel = torch.tensor([r.name in sel_names for r in recs])
    gts = torch.from_numpy(np.stack([r.mask128 for r in recs]))

    adapter = DecoderAdapter(m, device=dev).to(dev).eval()     # eval: BatchNorm stats frozen
    prefixes = [] if args.frozen else (args.unfreeze.split(",") if args.unfreeze else C.UNFREEZE[m])
    dec_params = [] if adapter.net is None else unfreeze(adapter, prefixes)
    channels = adapter.channels()
    extra_ch = 1 if C.AGREEMENT[m] else 0
    head = LocHead(channels, extra_ch=extra_ch, image_branch=True).to(dev).train()
    print(f"{m} {tag}: channels {channels}, trainable decoder params "
          f"{sum(p.numel() for p in dec_params)/1e6:.2f}M, picking on {int(sel.sum())} real images",
          flush=True)

    groups = [{"params": head.parameters(), "lr": args.head_lr}]
    if dec_params:
        groups.append({"params": dec_params, "lr": args.dec_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    warm = max(1, min(500, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    synth = SynthSet(m)
    print(f"synthetic images: {len(synth)}; real in training: {len(real_tr)}", flush=True)
    loaders = [iter(DataLoader(synth, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                               drop_last=True, persistent_workers=args.workers > 0))]
    real_loader = DataLoader(RealSet(real_tr), batch_size=args.bs, shuffle=True, drop_last=True) \
        if real_tr else None

    def next_batch():
        use_real = real_loader is not None and np.random.rand() < args.real_weight
        if use_real:
            return next(iter(real_loader))
        try:
            return next(loaders[0])
        except StopIteration:
            loaders[0] = iter(DataLoader(synth, batch_size=args.bs, shuffle=True,
                                         num_workers=args.workers, drop_last=True))
            return next(loaders[0])

    best = -1.0
    for step in range(1, args.steps + 1):
        x, y, bits = (t.to(dev, non_blocking=True) for t in next_batch())
        if C.FLIP_SAFE[m] and np.random.rand() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            with torch.set_grad_enabled(bool(dec_params)):   # B0: no decoder gradients needed
                feats, extra, _ = adapter(x, bits)
            logits = head(feats, extra, x)
        loss = seg_loss(logits.float(), y)
        if (args.bit_loss or args.clean_loss) and adapter.pos_logits is not None and dec_params:
            # dense decoder: each position inside the watermark should read the known bits;
            # positions in clean areas should read nothing (scores near 0)
            pl = adapter.pos_logits.float()[:, -bits.shape[1]:]                     # (B, n, h, w)
            wfrac = F.adaptive_avg_pool2d(y.float()[:, None], pl.shape[-2:])       # (B, 1, h, w) watermarked share
            tgt = bits.float()[:, :, None, None].expand_as(pl)
            bce = F.binary_cross_entropy_with_logits(pl, tgt, reduction="none").mean(1, keepdim=True)
            loss = loss + args.bit_loss * (bce * wfrac).sum() / wfrac.sum().clamp(min=1.0)
            loss = loss + args.clean_loss * ((pl ** 2).mean(1, keepdim=True) * (1 - wfrac)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + dec_params, 1.0)
        opt.step()
        sched.step()

        out_of_time = (time.time() - t0) / 3600 > args.hours
        if step % args.eval_every == 0 or step == args.steps or out_of_time:
            probs, bit_acc = evaluate(adapter, head, recs, dev, n_bits, m)
            miou = metric.miou_torch(probs[sel] >= 0.5, gts[sel]).mean().item()
            print(f"step {step:6d} loss {loss.item():.4f} real mIoU@0.5 {miou:.4f} "
                  f"bit acc {bit_acc:.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
            if miou > best:
                best = miou
                dec_path = None
                if dec_params:
                    dec_path = str(C.CKPT_DIR / f"{tag}_{m}_decoder.pt")
                    torch.save(adapter.net.state_dict(), dec_path)
                torch.save({"state": head.state_dict(), "channels": channels, "extra_ch": extra_ch,
                            "layers": C.LAYERS[m], "size": C.DECODER_INPUT_SIZE[m],
                            "decoder_ckpt": dec_path, "image_branch": True},
                           C.CKPT_DIR / f"{tag}_{m}.pt")
                oof = {"names": [r.name for r in recs], "probs": probs.half(), "gts": gts,
                       "folds": [folds[r.name] for r in recs]}
                if real_tr:
                    oof["tune_names"] = sorted(sel_names)   # the other non-holdout images were trained on
                torch.save(oof, C.CACHE_DIR / f"oof_{tag}_{m}.pt")
                print(f"  saved best ({best:.4f})", flush=True)
        if out_of_time:
            break
    print(f"{m} {tag} done: best real mIoU@0.5 {best:.4f}")


if __name__ == "__main__":
    main()
