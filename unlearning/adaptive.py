"""Fit the exact adaptive clean/occluded classifier of leaderboard entry 3905."""
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

import task_template as official
from phase1.analyze import transform_batch
from phase2.evaluate import validate_state
from phase6.run import state_with_head
from phase8.partial_counterfactual import encode
from phase9.destinations import conditional, head_matrix, prob
from phase9.destinations_dual import fixed_concentration, train_student
from phase9.destinations_prior import prior
from phase9.destinations_recipe import unlearn as fit_parent
from phase9.recipe import _split

def occluded_logits(model, raw, labels, batch, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    targeted, random, clean = [], [], []
    for pixels, y in zip(raw.split(batch), labels.cuda().split(batch)):
        x = official.preprocess(pixels).detach().requires_grad_()
        fill = x.detach().mean((2, 3), keepdim=True)
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x).float()
        margin = z.gather(1, y[:, None]).squeeze(1) - z.scatter(1, y[:, None], -torch.inf).logsumexp(1)
        gradient = torch.autograd.grad(margin.sum(), x)[0]
        with torch.no_grad():
            sensitivity = (gradient * (x - fill)).sum(1, keepdim=True)
            tiles = F.avg_pool2d(sensitivity, 32, 32).flatten(1)
            if not torch.isfinite(tiles).all() or tiles.shape[1] != 49:
                raise ValueError("Invalid pixel attribution map")
            for order, results in ((tiles, targeted), (torch.rand(tiles.shape, generator=generator, device=x.device), random)):
                mask = torch.zeros_like(tiles).scatter_(1, order.topk(12).indices, 1.)
                mask = F.interpolate(mask.view(-1, 1, 7, 7), size=x.shape[-2:], mode="nearest")
                with torch.autocast("cuda", dtype=torch.float16):
                    results.append(model(torch.lerp(x.detach(), fill, mask)).float().cpu())
            clean.append(z.detach().cpu())
    return [torch.cat(rows) for rows in (targeted, random, clean)]

def source_margin(logits, labels):
    own = logits.gather(1, labels[:, None]).squeeze(1)
    others = logits.scatter(1, labels[:, None], -torch.inf)
    return own - others.logsumexp(1)


def mask_reliability(observations, labels):
    """Select one mixing weight per image, requiring agreement across both views."""
    fractions = []
    for salient, random, clean in observations.values():
        salient_margin = source_margin(salient, labels)
        drop = source_margin(clean, labels) - salient_margin
        excess = source_margin(random, labels) - salient_margin
        fractions.append(torch.where(drop > 0, (excess / drop.clamp_min(1)).clamp(0, 1), 0.))
    return torch.stack(fractions).amin(0)


def adaptive_target(parent_logits, observations, labels, log_prior, weight):
    parent = prob(parent_logits.double())
    own = parent.gather(1, labels[:, None])
    other = conditional(parent, labels)
    salient, _, clean = observations
    clean = conditional(prob(clean.double() + 2 * log_prior), labels)
    salient = conditional(prob(salient.double() + 2 * log_prior), labels)
    teacher = torch.lerp(clean, salient, weight[:, None].double())
    concentration = other.square().sum(1, keepdim=True)
    adjusted = fixed_concentration(.25 * other + .75 * teacher, concentration, labels)
    target = ((1 - own) * adjusted).scatter(1, labels[:, None], own)
    if not torch.isfinite(target).all() or (target.sum(1) - 1).abs().max() > 1.e-9:
        raise ValueError("Invalid adaptive target distribution")
    return target.float()


@torch.inference_mode(False)
@torch.enable_grad()
def unlearn(model, forget_images, *, cpu_readout=None):
    """Fit from the original ViT and source pixels, publishing the update only after validation."""
    if model.head.weight.device.type != "cuda" or official.DEVICE != "cuda":
        raise ValueError("The adaptive training recipe requires CUDA")
    if forget_images.dtype != torch.uint8 or tuple(forget_images.shape) != (3500, 224, 224, 3):
        raise ValueError("Expected 3,500 original uint8 RGB images")
    raw = forget_images.detach().cpu()
    worker = deepcopy(model).eval().requires_grad_(False)
    original = {k: v.detach().cpu().clone() for k, v in worker.state_dict().items()}
    labels = official.predict(worker, raw, 48).argmax(1)
    fit_parent(worker, raw, cpu_readout=cpu_readout)
    parent = {k: v.detach().cpu().clone() for k, v in worker.state_dict().items()}

    observations, features, parent_logits = {}, {}, {}
    for view, pixels in (("exact", raw), ("crop", transform_batch(raw.numpy(), "crop"))):
        worker.load_state_dict(original, strict=True)
        observations[view] = occluded_logits(worker, pixels, labels, 24, 22611)
        worker.load_state_dict(parent, strict=True)
        x, z, _ = encode(worker, pixels, 48, "cuda")
        features[view], parent_logits[view] = x.cpu(), z.cpu()

    base = head_matrix(worker)
    counts = labels.bincount(minlength=len(base))
    high_classes = torch.where(counts >= .96 * counts.max())[0]
    fit_ids = np.flatnonzero(_split(labels.numpy()) < 3)
    weight = mask_reliability(observations, labels)
    log_prior = prior(labels, len(base))
    cache = {
        view: {"student": features[view], "dual": adaptive_target(
            parent_logits[view], observations[view], labels, log_prior, weight)}
        for view in ("exact", "crop")
    }
    fitted = train_student(cache, labels, fit_ids, base, high_classes)
    state = state_with_head(parent, torch.lerp(base, fitted, .5))
    validate_state(state, original)
    model.load_state_dict(state, strict=True)
    return model.eval()
