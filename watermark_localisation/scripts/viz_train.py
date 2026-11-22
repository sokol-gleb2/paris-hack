"""Look at the data: grids of train photos | true mask | current prediction, per method and per type
(full watermark vs partial). Saves small JPEGs to ~/wmloc_viz (home: scratch is full). CPU, seconds.
Run: python -m scripts.viz_train [--n 8]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from wmloc import config as C
from wmloc import data

PRED = {"pixelseal": "stageA", "trustmark": "stageAh", "stegastamp": "box"}


def tile(img_u8, mask128, prob, label):
    s = 256
    a = Image.fromarray(img_u8).resize((s, s))
    m = Image.fromarray((mask128 * 255).astype(np.uint8)).resize((s, s), Image.NEAREST).convert("RGB")
    p = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).resize((s, s), Image.NEAREST).convert("RGB")
    row = Image.new("RGB", (3 * s + 8, s + 14), "white")
    for k, im in enumerate((a, m, p)):
        row.paste(im, (k * (s + 4), 14))
    ImageDraw.Draw(row).text((2, 1), label, fill="black")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    out = Path.home() / "wmloc_viz"
    out.mkdir(exist_ok=True)
    train = data.cached_split("train")
    flag = {r["image_name"]: bool(r.get("full_watermark")) for r in
            json.loads((C.DATA_DIR / "train" / "metadata.json").read_text())}
    for m in C.METHODS:
        tag = PRED[m]
        d = torch.load(C.CACHE_DIR / f"oof_{tag}_{m}.pt", weights_only=False)
        prob = dict(zip(d["names"], d["probs"].float().numpy()))
        recs = data.by_method(train, m)
        for full in (True, False):
            rs = [r for r in recs if flag[r.name] == full][: args.n]
            rows = [tile(r.image, r.mask128, prob[r.name],
                         f"{r.name} area {r.mask128.mean():.2f} type {data.edit_type(r.mask128)}") for r in rs]
            grid = Image.new("RGB", (rows[0].width, sum(x.height for x in rows)), "white")
            y = 0
            for x in rows:
                grid.paste(x, (0, y)); y += x.height
            f = out / f"{m}_{'full' if full else 'partial'}.jpg"
            grid.save(f, quality=80)
            print("saved", f, f"(columns: photo | true mask | {tag} prediction)")


if __name__ == "__main__":
    main()
