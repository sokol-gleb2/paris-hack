import numpy as np
import torch
from torch import nn
from phase6.run import official


def composites(images, labels, fold, seed=363):
    """Combine quadrants from same-label images paired strictly within each split."""
    partners = np.empty(len(labels), dtype=int)
    rng = np.random.default_rng(seed)
    for split in np.unique(fold):
        for c in np.unique(labels[fold == split]):
            ids = rng.permutation(np.flatnonzero((labels == c) & (fold == split)))
            if len(ids) < 2:
                raise ValueError("Each proxy class needs two distinct parents per split")
            partners[ids] = np.roll(ids, -1)
    assert np.all(partners != np.arange(len(labels)))
    result = images.clone()
    for ys, xs in [(slice(0, 112), slice(0, 112)), (slice(112, 224), slice(112, 224))]:
        result[:, ys, xs] = images[partners, ys, xs]
    return result


@torch.no_grad()
def extract(model, images, batch_size, device):
    """fc2 input and pre-normalization CLS suffice to simulate this exact last-layer edit."""
    if model.global_pool != "token" or not isinstance(model.fc_norm, nn.Identity):
        raise ValueError("Expected the official CLS head")
    pieces = {key: [] for key in ("a", "u", "logits")}
    def collect(key):
        def hook(mod, args):
            pieces[key].append(args[0][:, 0].detach().float().contiguous())
        return hook
    h1 = model.blocks[-1].mlp.fc2.register_forward_pre_hook(collect("a"))
    h2 = model.norm.register_forward_pre_hook(collect("u"))
    try:
        for batch in images.split(batch_size):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                pieces["logits"].append(model(official.preprocess(batch)).float())
    finally:
        h1.remove()
        h2.remove()
    return {k: torch.cat(v) for k, v in pieces.items()}


def subset(data, ids):
    return {k: v[ids] for k, v in data.items()}
