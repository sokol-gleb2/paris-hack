"""Step 8: pixel-level watermark detector (small U-Net on the raw image), trained on REAL train photos only.

Why: decoder features are only 8x8, but the watermark is a pixel-level pattern added inside the region.
A small network that judges each spot locally ("is there watermark pattern here?") can draw sharp masks,
and every real photo gives thousands of labelled patches. Random crops make each step different.

Split (honest): train on folds not in {--val-fold, HOLDOUT_FOLD}; pick the checkpoint on --val-fold;
the holdout fold is never used. Step 4 tunes this tag on --val-fold only and scores it on the holdout.
Writes checkpoints/pix_<m>.pt, cache/oof_pix_<m>.pt, cache/test_pix_<m>.pt -> Step 4 tag "pix".
Run: python -m scripts.step8_pixnet --method pixelseal [--steps 6000] [--hours 2] [--crop 256]
"""
import argparse
import io
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from wmloc import config as C
from wmloc import data, metric
from wmloc.head import seg_loss


class RealCrops(Dataset):
    """Random square crops (side multiple of 4) of real train images + matching 128-grid mask crops."""

    def __init__(self, recs, crop, per_epoch=2000):
        self.recs, self.crop, self.n = recs, crop, per_epoch

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = self.recs[np.random.randint(len(self.recs))]
        c = self.crop
        y0, x0 = (np.random.randint(0, (C.IMG - c) // C.CELL + 1, size=2) * C.CELL)
        img = data.image_to_tensor(np.array(r.image[y0:y0 + c, x0:x0 + c]))
        m = r.mask128[y0 // C.CELL:(y0 + c) // C.CELL, x0 // C.CELL:(x0 + c) // C.CELL]
        return img, torch.from_numpy(np.ascontiguousarray(m))


class FakeCrops(Dataset):
    """Random crops of Step 5's fake images (exact dataset embedders; packs in SYNTH_DIR)."""

    def __init__(self, method, crop, per_epoch=2000):
        self.items = []
        for f in sorted((C.SYNTH_DIR / method).glob("pack_*.pt")):
            self.items += torch.load(f, weights_only=False)
        self.crop, self.n = crop, per_epoch

    def __len__(self):
        return self.n if self.items else 0

    def __getitem__(self, i):
        it = self.items[np.random.randint(len(self.items))]
        img = np.array(Image.open(io.BytesIO(it["png"])).convert("RGB"))
        m128 = metric.downsample_mask((np.asarray(Image.open(io.BytesIO(it["mask_png"]))) > 127).astype(np.uint8))
        c = self.crop
        y0, x0 = (np.random.randint(0, (C.IMG - c) // C.CELL + 1, size=2) * C.CELL)
        m = m128[y0 // C.CELL:(y0 + c) // C.CELL, x0 // C.CELL:(x0 + c) // C.CELL]
        return data.image_to_tensor(np.ascontiguousarray(img[y0:y0 + c, x0:x0 + c])), torch.from_numpy(np.ascontiguousarray(m))


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU())


class PixNet(nn.Module):
    """(B,3,512,512) in [0,1] -> (B,128,128) logits. Input = image + its high-pass (where watermarks live)."""

    def __init__(self, w=32):
        super().__init__()
        self.e0, self.e1, self.e2 = block(6, w), block(w, 2 * w), block(2 * w, 4 * w)       # 512, 256, 128
        self.e3, self.e4 = block(4 * w, 8 * w), block(8 * w, 8 * w)                          # 64, 32
        self.d3, self.d2 = block(16 * w, 4 * w), block(8 * w, 4 * w)                         # 64, 128
        self.out = nn.Conv2d(4 * w, 1, 1)

    def forward(self, x):
        hp = x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        z0 = self.e0(torch.cat([x - 0.5, hp * 10], 1))
        z1 = self.e1(F.max_pool2d(z0, 2))
        z2 = self.e2(F.max_pool2d(z1, 2))
        z3 = self.e3(F.max_pool2d(z2, 2))
        z4 = self.e4(F.max_pool2d(z3, 2))
        y = self.d3(torch.cat([F.interpolate(z4, scale_factor=2), z3], 1))
        y = self.d2(torch.cat([F.interpolate(y, scale_factor=2), z2], 1))
        return self.out(y)[:, 0]


@torch.no_grad()
def predict(net, recs, dev, bs=16):
    net.eval()
    out = []
    for s in range(0, len(recs), bs):
        x = torch.stack([data.image_to_tensor(np.array(r.image)) for r in recs[s:s + bs]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            out.append(torch.sigmoid(net(x).float()).cpu())
    net.train()
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--hours", type=float, default=2.0, help="stop after this long (keep below the job's --time)")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--crop", type=int, default=256, help="training crop side (multiple of 32)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-fold", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--width", type=int, default=32, help="network width (channels at full resolution)")
    ap.add_argument("--fake-weight", type=float, default=0.0, help="share of batches from fake images (Step 5 packs)")
    ap.add_argument("--tag", default=None, help="default: pix, or pixf with --fake-weight > 0")
    args = ap.parse_args()
    tag = args.tag or ("pixf" if args.fake_weight > 0 else "pix")
    assert args.val_fold != C.HOLDOUT_FOLD and args.crop % 32 == 0
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, t0 = args.method, time.time()

    train = data.cached_split("train")
    folds = data.make_folds(train)
    recs = data.by_method(train, m)
    fold_of = [folds[r.name] for r in recs]
    tr = [r for r, f in zip(recs, fold_of) if f not in (args.val_fold, C.HOLDOUT_FOLD)]
    val = torch.tensor([f == args.val_fold for f in fold_of])
    hold = torch.tensor([f == C.HOLDOUT_FOLD for f in fold_of])
    gts = torch.from_numpy(np.stack([r.mask128 for r in recs]))
    print(f"{m}: train {len(tr)} photos, pick on fold {args.val_fold} ({int(val.sum())}), "
          f"holdout {int(hold.sum())} untouched", flush=True)

    dl = DataLoader(RealCrops(tr, args.crop), batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0,
                    worker_init_fn=lambda w: np.random.seed(C.SEED * 100 + w + int(time.time()) % 1000))
    it = iter(dl)
    fake_dl = fake_it = None
    if args.fake_weight > 0:
        fake = FakeCrops(m, args.crop)
        print(f"{m}: {len(fake.items)} fake images, fake share {args.fake_weight}", flush=True)
        fake_dl = DataLoader(fake, batch_size=args.bs, shuffle=False, num_workers=args.workers, drop_last=True,
                             worker_init_fn=lambda w: np.random.seed(C.SEED * 100 + 50 + w + int(time.time()) % 1000))
        fake_it = iter(fake_dl)
    net = PixNet(w=args.width).to(dev).train()
    print(f"PixNet {sum(p.numel() for p in net.parameters())/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    best, best_state = -1.0, None
    for step in range(1, args.steps + 1):
        if fake_dl is not None and np.random.rand() < args.fake_weight:
            try:
                x, y = next(fake_it)
            except StopIteration:
                fake_it = iter(fake_dl); x, y = next(fake_it)
        else:
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl); x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = net(x)
        loss = seg_loss(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        out_of_time = (time.time() - t0) / 3600 > args.hours
        if step % args.eval_every == 0 or step == args.steps or out_of_time:
            probs = predict(net, recs, dev)
            mv = metric.miou_torch(probs[val] >= 0.5, gts[val]).mean().item()
            mh = metric.miou_torch(probs[hold] >= 0.5, gts[hold]).mean().item()
            print(f"step {step:6d} loss {loss.item():.4f} val mIoU@0.5 {mv:.4f} (holdout {mh:.4f}) "
                  f"({(time.time()-t0)/60:.0f} min)", flush=True)
            if mv > best:
                best = mv
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                torch.save(best_state, C.CKPT_DIR / f"{tag}_{m}.pt")
                torch.save({"names": [r.name for r in recs], "probs": probs.half(), "gts": gts,
                            "folds": fold_of,
                            "tune_names": [r.name for r, f in zip(recs, fold_of) if f == args.val_fold]},
                           C.CACHE_DIR / f"oof_{tag}_{m}.pt")
        if out_of_time:
            break

    net.load_state_dict(best_state)
    test = data.by_method(data.cached_split("test"), m)
    probs = predict(net, test, dev)
    torch.save({"names": [r.name for r in test], "probs": probs.half()}, C.CACHE_DIR / f"test_{tag}_{m}.pt")
    print(f"{m} {tag} done: best val mIoU@0.5 {best:.4f}; test predictions saved")


if __name__ == "__main__":
    main()
