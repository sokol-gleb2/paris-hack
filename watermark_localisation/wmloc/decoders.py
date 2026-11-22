"""Decoder adapter: turns a starter-kit decoder into a spatial feature extractor."""
from __future__ import annotations

import importlib.util
import operator
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# ------------------------------------------------------------ starter kit hooks
# The kit's package is also called `wmloc`, so its files are loaded by path under other names.
KIT_DIR = Path(os.environ.get("WMLOC_KIT", C.TASK_DIR / "starter_kit"))
_cache = {}


def kit_module(name: str):
    """starter_kit/wmloc/<name>.py loaded as module 'kit_<name>' (no clash with our wmloc)."""
    key = "kit_" + name
    if key not in _cache:
        spec = importlib.util.spec_from_file_location(key, KIT_DIR / "wmloc" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cache[key] = mod
    return _cache[key]


def load_wrapper(method: str, device: str = "cuda"):
    """The kit's wrapper (logits, decode, features, embed). Models: $WMLOC_MODELS or starter_kit/models."""
    key = (method, str(device))
    if key not in _cache:
        models = os.environ.get("WMLOC_MODELS", str(KIT_DIR / "models"))
        _cache[key] = kit_module("watermarks").load_watermark(method, models, device)
    return _cache[key]


class _KitEmbedder:
    """embed(img01 (1,3,h,w), bits (1,n)) -> watermarked img01, through the kit's uint8 embed
    (the exact embedder used to build the dataset)."""

    def __init__(self, wm):
        self.wm = wm

    def embed(self, img01, bits):
        u8 = (img01[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
        out = self.wm.embed(u8, [int(b) for b in bits[0].tolist()])
        return torch.from_numpy(out).permute(2, 0, 1)[None].float().div(255).to(img01.device)


def load_embedder(method: str, device: str = "cuda"):
    return _KitEmbedder(load_wrapper(method, device))


# ------------------------------------------------------------ reading outputs
def first_tensor(out) -> torch.Tensor:
    """The main tensor of a decoder output (dicts and tuples hold it first)."""
    if isinstance(out, dict):
        out = out.get("preds", out.get("bits", next(iter(out.values()))))
    if isinstance(out, (tuple, list)):
        out = out[0]
    return torch.as_tensor(out)


def to_scores(out, kind: str, n_bits: int | None = None) -> torch.Tensor:
    """Decoder output -> (B,n) float scores where > 0 means bit 1.
    `kind` says what the output is ("logits", "probs" or "bits"); it is set per method in
    config.py. No guessing from value ranges: raw scores that all happen to lie in [0,1]
    would otherwise be read with the wrong cutoff."""
    out = first_tensor(out).float()
    if out.dim() == 1:
        out = out[None]
    if out.dim() == 4:                       # per-position predictions -> average
        out = out.mean(dim=(2, 3))
    if n_bits is not None and out.shape[1] == n_bits + 1:
        out = out[:, 1:]                     # drop the detection score (column 0, Video Seal style)
    if kind == "logits":
        return out
    if kind == "probs":
        return torch.logit(out.clamp(1e-4, 1 - 1e-4))
    if kind == "bits":
        return out * 2 - 1
    raise ValueError(f"unknown output kind {kind!r}")


def decode_bits(wrapper, x01: torch.Tensor, method: str, n_bits: int | None = None) -> torch.Tensor:
    """x01: (B,3,H,W) in [0,1] -> (B,n) uint8 bits, through the starter kit's own decode."""
    with torch.no_grad():
        out = wrapper.decode(x01)
    return (to_scores(out, C.WRAPPER_OUTPUT[method], n_bits) > 0).to(torch.uint8).cpu()


def bits_from_output(out, n_bits: int, method: str) -> torch.Tensor:
    """Raw network output -> (B,n) uint8 bits."""
    return (to_scores(out, C.NET_OUTPUT[method], n_bits) > 0).to(torch.uint8).cpu()


# ------------------------------------------------------------ helpers
def find_network(wrapper) -> tuple[str, nn.Module]:
    """(name, network): the largest nn.Module attribute of the wrapper. A GUESS: it can pick
    the embedder if the wrapper holds both. Set DECODER_NET to be sure."""
    if isinstance(wrapper, nn.Module):
        return "<the wrapper itself>", wrapper
    best, best_name, best_n = None, "", -1
    for name in dir(wrapper):
        if name.startswith("__"):
            continue
        try:
            obj = getattr(wrapper, name)
        except Exception:
            continue
        if isinstance(obj, nn.Module):
            n = sum(p.numel() for p in obj.parameters())
            if n > best_n:
                best, best_name, best_n = obj, name, n
    if best is None:
        raise RuntimeError("no nn.Module inside the wrapper; set DECODER_NET[method]")
    return best_name, best


def get_network(wrapper, method: str) -> tuple[str, nn.Module]:
    """The decoder network: DECODER_NET[method] if set, else the guess above."""
    path = C.DECODER_NET[method]
    if path == "none":                       # StegaStamp: ONNX, only wrapper.logits()
        return "none (ONNX: wrapper.logits only)", None
    if path:
        return path, operator.attrgetter(path)(wrapper)
    return find_network(wrapper)


def to_spatial(t) -> torch.Tensor:
    """Hook output -> (B,C,H,W). Handles tuples, NHWC maps and ViT token sequences."""
    if isinstance(t, dict):
        t = next(iter(t.values()))
    if isinstance(t, (tuple, list)):
        t = t[0]
    if t.dim() == 4:
        if t.shape[2] == t.shape[3]:
            return t                         # NCHW
        if t.shape[1] == t.shape[2]:
            return t.permute(0, 3, 1, 2)     # NHWC -> NCHW
        return t
    if t.dim() == 3:                         # (B, N tokens, C)
        n = t.shape[1]
        s = int(n ** 0.5)
        t = t[:, n - s * s:]                 # drop class / register tokens
        return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s)
    raise ValueError(f"cannot make a spatial map from shape {tuple(t.shape)}")


class FeatureTap:
    """Captures outputs of named submodules with forward hooks."""

    def __init__(self, net: nn.Module, names: list[str]):
        mods = dict(net.named_modules())
        missing = [n for n in names if n not in mods]
        if missing:
            raise KeyError(f"layers not found: {missing}. Run step1_probe --list-layers")
        self.names, self.out = list(names), {}
        self.handles = [mods[n].register_forward_hook(self._hook(n)) for n in names]

    def _hook(self, name):
        def fn(_module, _inp, out):
            self.out[name] = out
        return fn

    def get(self) -> list[torch.Tensor]:
        return [to_spatial(self.out[n]) for n in self.names]

    def remove(self):
        for h in self.handles:
            h.remove()


# ------------------------------------------------------------ the adapter
class DecoderAdapter(nn.Module):
    """x01 (B,3,512,512) in [0,1] -> list of spatial feature maps (+ agreement map)."""

    def __init__(self, method: str, wrapper=None, device: str = "cuda"):
        super().__init__()
        self.method = method
        self.wrapper = wrapper if wrapper is not None else load_wrapper(method, device)
        self.net_name, self.net = get_network(self.wrapper, method)
        self.size = C.DECODER_INPUT_SIZE[method]
        self.tap = FeatureTap(self.net, C.LAYERS[method]) if C.LAYERS[method] and self.net is not None else None
        cfg = C.AGREEMENT[method] if self.net is not None else None
        self.agree_tap = FeatureTap(self.net, [cfg[0]]) if cfg else None
        self.linear = dict(self.net.named_modules())[cfg[1]] if cfg else None
        self.pos_logits = None               # per-position scores of the last forward (Step 1 check)

    def preprocess(self, x01: torch.Tensor) -> torch.Tensor:
        """Exactly what the kit's wrapper does before its network (starter_kit/wmloc/watermarks.py)."""
        if self.net is None:
            return x01                       # stegastamp: the wrapper resizes itself
        x01 = x01.to(next(self.net.parameters()).device)
        if tuple(x01.shape[-2:]) != (self.size, self.size):
            x01 = F.interpolate(x01, size=(self.size, self.size), mode="bilinear",
                                align_corners=False, antialias=True)
        return x01 * 2 - 1 if self.method == "trustmark" else x01   # pixelseal: det normalises inside

    def forward(self, x01: torch.Tensor, bits: torch.Tensor | None = None):
        # ADAPT (preferred): if the starter-kit wrapper exposes the intermediate feature maps
        # itself, take `feats` from that call instead of the hooks (Step 1, item 1).
        if self.net is None:                 # stegastamp (ONNX): logits only, no feature maps
            return [], None, self.wrapper.logits(x01)
        out = self.net(self.preprocess(x01))
        feats = self.tap.get() if self.tap else []
        extra = None
        if self.agree_tap is not None:
            fmap = self.agree_tap.get()[0]                            # (B,D,h,w), before the average
            w = self.linear.weight                                     # (n or n+1, D)
            logits = torch.einsum("bdhw,nd->bnhw", fmap, w)
            if self.linear.bias is not None:
                logits = logits + self.linear.bias.view(1, -1, 1, 1)
            self.pos_logits = logits
            if bits is not None:
                logits = logits[:, -bits.shape[1]:]                    # drop detection column
                sign = bits.float() * 2 - 1                            # (B,n) in {-1,+1}
                extra = (logits * sign[:, :, None, None]).mean(1, keepdim=True)
                extra = torch.tanh(extra / 4.0)                        # bounded, ~0 if clean
        return feats, extra, out

    def channels(self) -> list[int]:
        """Channel count of each tapped layer (runs one dummy forward)."""
        if self.net is None:
            return []
        dev = next(self.net.parameters()).device
        with torch.no_grad():
            feats, _, _ = self.forward(torch.zeros(1, 3, C.IMG, C.IMG, device=dev))
        return [f.shape[1] for f in feats]


def capture_net_input(wrapper, net: nn.Module, x01: torch.Tensor, method: str) -> torch.Tensor:
    """What the wrapper actually feeds its network (to check `preprocess`)."""
    box = {}
    h = net.register_forward_pre_hook(lambda m, inp: box.setdefault("x", inp[0].detach()))
    try:
        decode_bits(wrapper, x01, method)
    finally:
        h.remove()
    if "x" not in box:
        raise RuntimeError("wrapper.decode never ran this network: set DECODER_NET[method] to the decoder")
    return box["x"]
