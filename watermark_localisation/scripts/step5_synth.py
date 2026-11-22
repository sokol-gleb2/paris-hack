"""Step 5: synthetic training images for one method (four edit types).

Run: python -m scripts.step5_synth --method pixelseal --clean-dir DIR [--coco-ann JSON]
                                   --n 500 --shard 1 [--nshards 1] [--embed-scale 1.0]
     python -m scripts.step5_synth --method pixelseal --check
Writes SYNTH_DIR/<method>/pack_<shard>_<k>.pt: up to --pack images each, stored as PNG bytes
(few files: the scratch file-count cap). Set SYNTH_DIR to a folder in $HOME.
"""
import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from wmloc import config as C
from wmloc import data
from wmloc.decoders import decode_bits, load_embedder, load_wrapper


# ------------------------------------------------------------ inputs
def load_clean(path: Path, rng) -> torch.Tensor:
    """Random-crop + resize a photo to (1,3,512,512) float in [0,1]; also returns crop box."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    side = int(s * rng.uniform(0.6, 1.0))
    x0, y0 = rng.integers(0, w - side + 1), rng.integers(0, h - side + 1)
    box = (int(x0), int(y0), int(x0 + side), int(y0 + side))
    im = im.crop(box).resize((C.IMG, C.IMG), Image.BICUBIC)
    return data.image_to_tensor(np.asarray(im))[None], box


def random_blob(h: int, w: int, rng) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        rx, ry = rng.uniform(0.12, 0.35) * w, rng.uniform(0.12, 0.35) * h
        pts = [(cx + rx * np.cos(a) * rng.uniform(0.6, 1.0), cy + ry * np.sin(a) * rng.uniform(0.6, 1.0))
               for a in np.linspace(0, 2 * np.pi, 14, endpoint=False)]
        d.polygon(pts, fill=255)
    return np.asarray(m) > 127


class CocoShapes:
    """Object masks from COCO annotations, mapped through the same crop/resize."""

    def __init__(self, ann_path):
        from pycocotools.coco import COCO
        self.coco = COCO(str(ann_path))
        self.by_file = {im["file_name"]: im["id"] for im in self.coco.dataset["images"]}

    def masks(self, file_name, box):
        iid = self.by_file.get(file_name)
        if iid is None:
            return []
        out = []
        for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid, iscrowd=False)):
            m = self.coco.annToMask(a)[box[1]:box[3], box[0]:box[2]]
            m = np.asarray(Image.fromarray(m * 255).resize((C.IMG, C.IMG), Image.NEAREST)) > 127
            frac = m.mean()
            if 0.004 < frac < 0.4:
                out.append(m)
        return out


def random_holes(target_area, rng):
    """Restored regions of 'full watermark' photos (seen in the train grids): 1-5 random smooth blobs
    anywhere in the image, often touching the border, unrelated to image content; total ~ target_area px."""
    k = int(rng.integers(1, 6))
    w = rng.dirichlet(np.ones(k)) * target_area
    m = np.zeros((C.IMG, C.IMG), bool)
    for a in w:
        side = int(np.clip(np.sqrt(a / 0.15) * rng.uniform(0.8, 1.25), 16, C.IMG))
        bh = int(np.clip(side * rng.uniform(0.6, 1.6), 16, C.IMG)); bw = int(np.clip(side * side / max(bh, 1), 16, C.IMG))
        y = int(rng.integers(-bh // 3, C.IMG - 2 * bh // 3)); x = int(rng.integers(-bw // 3, C.IMG - 2 * bw // 3))
        blob = random_blob(bh, bw, rng)
        ys, xs = slice(max(y, 0), min(y + bh, C.IMG)), slice(max(x, 0), min(x + bw, C.IMG))
        m[ys, xs] |= blob[ys.start - y:ys.stop - y, xs.start - x:xs.stop - x]
    return m


def sample_rect(boxes, method, rng):
    """Height/width from real train bounding boxes (+-15%), aspect-limited, random position."""
    h, w = boxes[int(rng.integers(len(boxes)))]
    h = int(np.clip(h * rng.uniform(0.85, 1.15), 16, C.IMG))
    w = int(np.clip(w * rng.uniform(0.85, 1.15), 16, C.IMG))
    a = C.MAX_ASPECT[method]
    if max(h, w) / min(h, w) > a:
        if h > w:
            h = int(w * a)
        else:
            w = int(h * a)
    y, x = int(rng.integers(0, C.IMG - h + 1)), int(rng.integers(0, C.IMG - w + 1))
    return y, x, h, w


# ------------------------------------------------------------ embedding
@torch.no_grad()
def embed_rect(embedder, img, rect, bits, scale=1.0):
    """ADAPT if the organisers' recipe differs: watermark crop R, paste it back.
    scale > 1: embed at scale x the size, then shrink back to R, so the watermark pattern is
    smaller, as if a watermarked image was outpainted and then resized to 512."""
    y, x, h, w = rect
    out = img.clone()
    crop = img[..., y:y + h, x:x + w]
    if scale != 1.0:
        crop = F.interpolate(crop, size=(round(h * scale), round(w * scale)), mode="bicubic",
                             align_corners=False).clamp(0, 1)
    wm = embedder.embed(crop, bits)                     # ADAPT: real embed signature
    if isinstance(wm, dict):
        wm = wm.get("imgs_w", next(iter(wm.values())))
    if scale != 1.0:
        wm = F.interpolate(wm, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    out[..., y:y + h, x:x + w] = wm.clamp(0, 1)
    return out


def choose_object(shapes, rect, rng, target_area=None):
    """An object mask inside R: a COCO outline if one fits, else a random blob.
    Without target_area the object covers 5-40% of R (removal / addition); with it
    (pixels), it is about that big (transplantation)."""
    y, x, h, w = rect
    R = np.zeros((C.IMG, C.IMG), bool)
    R[y:y + h, x:x + w] = True
    inside = [m & R for m in shapes if (m & R).sum() > 0.5 * m.sum()]
    if target_area is None:
        good = [m for m in inside if 0.03 < m.sum() / R.sum() < 0.2]   # check #2: holes were too big
    else:
        good = [m for m in inside if 0.7 < m.sum() / target_area < 1.4]  # check #2: objects were too small
    if good:
        return good[int(rng.integers(len(good)))]
    if target_area is None:
        bh, bw = max(8, int(h * 0.6)), max(8, int(w * 0.6))     # smaller holes
    else:                                   # a random blob fills about a quarter of its box
        side = int(np.sqrt(target_area / 0.25))
        bh, bw = min(h, max(8, side)), min(w, max(8, side))
    by = y + int(rng.integers(0, h - bh + 1))
    bx = x + int(rng.integers(0, w - bw + 1))
    m = np.zeros((C.IMG, C.IMG), bool)
    m[by:by + bh, bx:bx + bw] = random_blob(bh, bw, rng)
    return m


def make_example(method, embedder, files, shapes_fn, stats, n_bits, rng, dev, scale=1.0):
    (boxes, obj_areas, hole_areas), edit_p = stats
    f1, f2 = random.sample(files, 2)
    img, box1 = load_clean(f1, rng)
    other, _ = load_clean(f2, rng)
    img, other = img.to(dev), other.to(dev)
    bits = torch.from_numpy(rng.integers(0, 2, n_bits).astype(np.float32))[None].to(dev)
    edit = str(rng.choice(list(edit_p), p=list(edit_p.values())))
    rect = sample_rect(boxes["rect"], method, rng) if edit == "rect" else (0, 0, C.IMG, C.IMG)
    wimg = embed_rect(embedder, img, rect, bits, scale)
    y, x, h, w = rect
    R = np.zeros((C.IMG, C.IMG), bool)
    R[y:y + h, x:x + w] = True

    if edit == "full":         # whole image watermarked, one object-shaped region restored to the original
        target = float(hole_areas[int(rng.integers(len(hole_areas)))]) * rng.uniform(0.8, 1.25) if hole_areas else None
        O = random_holes(target if target else 0.1 * C.IMG * C.IMG, rng)
        Ot = torch.from_numpy(O).to(dev)[None, None].float()
        out = Ot * img + (1 - Ot) * wimg
        mask = R & ~O
    elif edit == "holes":
        O = choose_object(shapes_fn(f1, box1), rect, rng)
        Ot = torch.from_numpy(O).to(dev)[None, None].float()
        mode = rng.choice(["restore", "paste", "blur"], p=[0.45, 0.35, 0.20])
        if mode == "restore":
            fill = img                                    # clean pixels: the watermark is gone
        elif mode == "paste":
            fill = other
        else:                                             # inpainting look, from CLEAN pixels only
            k = 31
            fill = F.avg_pool2d(F.pad(img, (k // 2,) * 4, mode="reflect"), k, stride=1)
            fill = (fill + 0.01 * torch.randn_like(fill)).clamp(0, 1)
        out = Ot * fill + (1 - Ot) * wimg
        mask = R & ~O
    elif edit == "object":     # transplant an object from the watermarked photo onto another photo
        target = None
        if obj_areas:
            target = float(obj_areas[int(rng.integers(len(obj_areas)))]) * rng.uniform(0.8, 1.25)
        O = choose_object(shapes_fn(f1, box1), rect, rng, target_area=target)
        if not O.any():
            edit, out, mask = "rect", wimg, R
        else:
            ys, xs = np.nonzero(O)
            dy = int(rng.integers(-ys.min(), C.IMG - ys.max()))
            dx = int(rng.integers(-xs.min(), C.IMG - xs.max()))
            O2 = np.roll(np.roll(O, dy, 0), dx, 1)
            src = torch.roll(wimg, shifts=(dy, dx), dims=(-2, -1))
            Ot = torch.from_numpy(O2).to(dev)[None, None].float()
            out = Ot * src + (1 - Ot) * other
            mask = O2
    else:                      # "rect": outpainting
        out, mask = wimg, R
    u8 = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    return u8, mask.astype(np.uint8), bits[0].byte().cpu().numpy(), edit


# ------------------------------------------------------------ main
def real_stats(method):
    """Statistics of the real train masks of this method, holdout fold excluded, following the dataset recipe
    (train metadata `full_watermark`):
    full       whole image watermarked, one region restored to the clean original (half of all photos);
               hole_areas = pixel areas of the restored regions
    partial    `rect` (outpainting rectangle; boxes["rect"] = its sizes) or `object` (transplant; obj_areas)
    edit_p     share of full / rect / object."""
    train = data.cached_split("train")
    folds = data.make_folds(train)
    flag = {r["image_name"]: bool(r.get("full_watermark")) for r in
            json.loads((C.DATA_DIR / "train" / "metadata.json").read_text())}
    recs = [r for r in data.by_method(train, method)
            if folds[r.name] != C.HOLDOUT_FOLD and r.mask512 is not None and r.mask512.any()]
    boxes, obj_areas, hole_areas, types = {"rect": [], "holes": []}, [], [], []
    for r in recs:
        if flag.get(r.name):
            types.append("full")
            hole_areas.append(int((1 - r.mask512).sum()))
            continue
        t = data.edit_type(r.mask128)
        t = "object" if t in ("object", "empty") else "rect"
        types.append(t)
        ys, xs = np.nonzero(r.mask512)
        if t == "rect":
            boxes["rect"].append((int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)))
        else:
            obj_areas.append(int(r.mask512.sum()))
    boxes["rect"] = boxes["rect"] or [(C.IMG // 2, C.IMG // 2)]
    boxes["holes"] = [(C.IMG, C.IMG)]
    edit_p = {t: types.count(t) / len(types) for t in ("full", "rect", "object") if t in types}
    return (boxes, obj_areas, hole_areas), edit_p, len(recs[0].bits), recs


def png_bytes(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def load_packs(method):
    """All fake examples of a method: list of dicts with png, mask_png, bits, edit, area."""
    items = []
    for f in sorted((C.SYNTH_DIR / method).glob("pack_*.pt")):
        items += torch.load(f, weights_only=False)
    return items


def check(method, dev):
    """Compare whole-image bit accuracy and area share: 50 synthetic vs 50 real (no holdout)."""
    wrapper = load_wrapper(method, dev)
    recs = real_stats(method)[-1]
    flag = {r["image_name"]: bool(r.get("full_watermark")) for r in
            json.loads((C.DATA_DIR / "train" / "metadata.json").read_text())}
    lines = load_packs(method)
    rows = {"real": [], "synth": []}
    for r in recs[:50]:
        x = data.image_to_tensor(r.image)[None].to(dev)
        acc = (decode_bits(wrapper, x, method, len(r.bits)).numpy()[0] == r.bits).mean()
        t = data.edit_type(r.mask128)
        t = "full" if flag.get(r.name) else ("object" if t in ("object", "empty") else "rect")
        rows["real"].append((t, acc, r.mask128.mean()))
    for l in random.sample(lines, min(50, len(lines))):
        img = np.asarray(Image.open(io.BytesIO(l["png"])).convert("RGB"))
        bits = data.parse_bits(l["bits"])
        x = data.image_to_tensor(img)[None].to(dev)
        acc = (decode_bits(wrapper, x, method, len(bits)).numpy()[0] == bits).mean()
        rows["synth"].append((l["edit"], acc, l["area"]))
    for k, v in rows.items():
        for t in ("full", "rect", "object"):
            sel = [(a, s) for e, a, s in v if e == t]
            if sel:
                print(f"  {k:5s} {t:6s} n={len(sel):2d} bit acc {np.mean([a for a, _ in sel]):.3f} "
                      f"area {np.mean([s for _, s in sel]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--clean-dir")
    ap.add_argument("--coco-ann", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--embed-scale", type=float, default=1.0,
                    help="> 1: smaller watermark pattern, as after outpainting + resizing")
    ap.add_argument("--pack", type=int, default=250, help="images per pack file")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.check:
        return check(args.method, dev)

    seed = C.SEED * 1000 + args.shard + 17 * C.METHODS.index(args.method)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    files = sorted(Path(args.clean_dir).glob("*.jpg")) + sorted(Path(args.clean_dir).glob("*.png"))
    files = files[args.shard::args.nshards] if args.nshards > 1 else files
    stats3, edit_p, n_bits, _ = real_stats(args.method)
    boxes, obj_areas, hole_areas = stats3
    print(f"{args.method}: {len(files)} clean photos, edit mix {edit_p}, {n_bits} bits, "
          f"{len(boxes['rect'])} rectangle sizes, {len(obj_areas)} object sizes, {len(hole_areas)} hole sizes, embed scale {args.embed_scale}")
    shapes = CocoShapes(args.coco_ann) if args.coco_ann else None
    shapes_fn = (lambda f, box: shapes.masks(Path(f).name, box)) if shapes else (lambda f, box: [])
    embedder = load_embedder(args.method, dev)

    out_dir = C.SYNTH_DIR / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    pack, k = [], 0

    def flush():
        nonlocal pack, k
        if pack:
            tmp = out_dir / f"tmp_{args.shard}_{k:03d}.pt"   # not matched by pack_*.pt until complete
            torch.save(pack, tmp)
            tmp.rename(out_dir / f"pack_{args.shard}_{k:03d}.pt")
            print(f"  saved pack_{args.shard}_{k:03d}.pt ({len(pack)} images)", flush=True)
            pack, k = [], k + 1

    for i in range(args.n):
        u8, mask, bits, edit = make_example(args.method, embedder, files, shapes_fn,
                                            (stats3, edit_p), n_bits, rng, dev,
                                            args.embed_scale)
        if mask.sum() < 16:
            continue
        pack.append({"png": png_bytes(u8), "mask_png": png_bytes(mask * 255),
                     "bits": "".join(map(str, bits.tolist())), "edit": edit,
                     "area": float(mask.mean()), "scale": args.embed_scale})
        if len(pack) >= args.pack:
            flush()
    flush()

if __name__ == "__main__":
    main()
