"""Step 1: decoder sanity checks + layer listing for one method.

Run: python -m scripts.step1_probe --method pixelseal [--list-layers] [--n 40]
Prints: which network was picked, what the outputs look like (set WRAPPER_OUTPUT /
NET_OUTPUT), the preprocess check, bit accuracies (whole image, GT crop, both readings
of the network output, flips, input sizes), the agreement check (if AGREEMENT is set)
and any spatial-transformer warp.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from wmloc import config as C
from wmloc import data
from wmloc.decoders import DecoderAdapter, capture_net_input, decode_bits, first_tensor, to_scores

SIZES = (224, 245, 256, 384, 400, 512)   # 224/245: TrustMark (Adobe code), 256: Pixel Seal, 400: StegaStamp


def bbox_crop(img01: torch.Tensor, mask512: np.ndarray) -> torch.Tensor:
    ys, xs = np.nonzero(mask512)
    return img01[..., ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def acc(pred: torch.Tensor, bits: np.ndarray) -> float:
    return float((pred.numpy().reshape(-1) == bits).mean())


def net_bits(adapter, x01, size, n_bits, kind):
    """Decode through our adapter path at a chosen input size, reading the output as `kind`."""
    old = adapter.size
    adapter.size = size
    try:
        with torch.no_grad():
            _, _, out = adapter(x01)
    finally:
        adapter.size = old
    return (to_scores(out, kind, n_bits) > 0).to(torch.uint8).cpu()


def describe(label, out):
    try:
        t = first_tensor(out).float()
        print(f"  {label}: {type(out).__name__} -> {tuple(t.shape)}, values in [{t.min():.3f}, {t.max():.3f}]")
    except Exception as e:  # noqa: BLE001
        print(f"  {label}: {type(out).__name__} (not readable as a tensor: {e})")


def capture_affine_thetas(fn):
    """Run fn() and return every warp passed to F.affine_grid (a spatial transformer).
    Misses warps done another way, e.g. after `from torch.nn.functional import affine_grid`."""
    seen, orig = [], F.affine_grid

    def spy(theta, *a, **k):
        seen.append(theta.detach().float().cpu())
        return orig(theta, *a, **k)

    F.affine_grid = spy
    try:
        fn()
    finally:
        F.affine_grid = orig
    return seen


def list_layers(net, adapter, device):
    rows, hooks = [], []
    for name, mod in net.named_modules():
        if name == "":
            continue
        def fn(m, i, o, name=name):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if isinstance(t, torch.Tensor):
                rows.append((name, type(m).__name__, tuple(t.shape)))
        hooks.append(mod.register_forward_hook(fn))
    with torch.no_grad():
        net(adapter.preprocess(torch.zeros(1, 3, C.IMG, C.IMG, device=device)))
    for h in hooks:
        h.remove()
    print(f"\n{'module':60s} {'type':22s} output shape")
    for name, typ, shape in rows:
        spatial = len(shape) == 4 or (len(shape) == 3 and shape[1] > 16)
        if spatial or typ == "Linear":
            print(f"{name:60s} {typ:22s} {shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--n", type=int, default=40, help="train images to test")
    ap.add_argument("--list-layers", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = args.method

    try:
        adapter = DecoderAdapter(m, device=dev).to(dev).eval()
    except KeyError as e:                    # wrong LAYERS names: carry on so --list-layers still prints
        print(f"WARNING: {e} -> running without LAYERS; fix them from the list below")
        C.LAYERS[m] = []
        adapter = DecoderAdapter(m, device=dev).to(dev).eval()
    wrapper, net = adapter.wrapper, adapter.net
    has_net = net is not None
    if has_net:
        print(f"network: {adapter.net_name} = {type(net).__name__}, "
              f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    else:
        print(f"network: {adapter.net_name} (wrapper-level checks only)")

    recs = [r for r in data.by_method(data.cached_split("train"), m) if r.mask512.any()][: args.n]
    x0 = data.image_to_tensor(recs[0].image)[None].to(dev)

    # 1. what the outputs look like
    print("outputs:")
    with torch.no_grad():
        describe("wrapper.decode()", wrapper.decode(x0))
        _, _, out0 = adapter(x0)
    describe("network" if has_net else "wrapper.logits()", out0)

    # 2. preprocess check
    if has_net:
        seen = capture_net_input(wrapper, net, x0, m)
        ours = adapter.preprocess(x0)
        same = seen.shape == ours.shape
        diff = (seen - ours).abs().max().item() if same else float("nan")
        print(f"wrapper feeds net {tuple(seen.shape)} range [{seen.min():.3f},{seen.max():.3f}]; "
              f"ours {tuple(ours.shape)}; preprocess max diff {diff:.2e}")

    # 3. bit accuracies
    keys = ["full", "crop", "hflip", "vflip", "rot90"]
    if has_net:
        keys += ["net_as_logits", "net_as_probs"] + [f"size{s}" for s in SIZES]
    res = {k: [] for k in keys}
    types = []
    for r in recs:
        x = data.image_to_tensor(r.image)[None].to(dev)
        n = len(r.bits)
        crop = bbox_crop(x, r.mask512)
        types.append(data.edit_type(r.mask128))
        res["full"].append(acc(decode_bits(wrapper, x, m, n), r.bits))
        res["crop"].append(acc(decode_bits(wrapper, crop, m, n), r.bits))
        res["hflip"].append(acc(decode_bits(wrapper, crop.flip(-1), m, n), r.bits))
        res["vflip"].append(acc(decode_bits(wrapper, crop.flip(-2), m, n), r.bits))
        res["rot90"].append(acc(decode_bits(wrapper, crop.rot90(1, (-2, -1)), m, n), r.bits))
        if has_net:
            for kind in ("logits", "probs"):
                res[f"net_as_{kind}"].append(acc(net_bits(adapter, crop, adapter.size, n, kind), r.bits))
            for s in SIZES:
                res[f"size{s}"].append(acc(net_bits(adapter, crop, s, n, C.NET_OUTPUT[m]), r.bits))

    types = np.array(types)
    print(f"\nbit accuracy over {len(recs)} train images (mean | rect | holes | object)")
    for k, v in res.items():
        v = np.array(v)
        parts = [f"{v[types == t].mean():.3f}" if (types == t).any() else "  -  "
                 for t in ("rect", "holes", "object")]
        print(f"  {k:13s} {v.mean():.3f} | " + " | ".join(parts))
    flip_ok = min(np.mean(res[k]) for k in ("hflip", "vflip", "rot90")) >= 0.9
    print(f"\nsuggest: FLIP_SAFE['{m}'] = {flip_ok}")
    if has_net:
        best = max((np.mean(res[f"size{s}"]), s) for s in SIZES)[1]
        kind = "logits" if np.mean(res["net_as_logits"]) >= np.mean(res["net_as_probs"]) else "probs"
        print(f"suggest: NET_OUTPUT['{m}'] = '{kind}'; DECODER_INPUT_SIZE['{m}'] = {best} "
              f"(the wrapper feeds {seen.shape[-1]})")

    # 4. agreement check: the average of the per-position scores must equal the normal output
    if adapter.agree_tap is not None:
        with torch.no_grad():
            _, _, out = adapter(x0)
        pos = adapter.pos_logits.float()
        whole = first_tensor(out).float()
        h, w = pos.shape[-2:]
        if whole.dim() == 2 and whole.shape[1] == pos.shape[1]:
            d = (pos.mean(dim=(2, 3)) - whole).abs().max().item()
            print(f"agreement check: max diff {d:.2e} (should be ~0)")
        else:
            print(f"agreement check: shapes differ {tuple(pos.shape)} vs {tuple(whole.shape)}")
        print(f"  per-position map {h}x{w}: one position is about {C.IMG // max(h, 1)} px of the photo")

    if args.list_layers and has_net:
        list_layers(net, adapter, dev)


if __name__ == "__main__":
    main()
