from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from phase6.run import official, state_with_head, transform_batch, validate_state
from phase8.partial_counterfactual import encode
from phase9.destinations import conditional, divergence, head_matrix, prob
from phase9.destinations_dual import fixed_concentration, train_student
from phase9.destinations_prior import prior
from phase9.destinations_recipe import unlearn as fit_incumbent
from phase9.loss_profile import statistics
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


def target(parent, occlusion, y, logprior):
    p = prob(parent.double())
    wrong = conditional(p, y)
    altered = conditional(prob(occlusion.double() + 2 * logprior), y)
    proposed = fixed_concentration(.25 * wrong + .75 * altered, wrong.square().sum(1, keepdim=True), y)
    own = p.gather(1, y[:, None])
    return ((1 - own) * proposed).scatter(1, y[:, None], own).float()


def assess(cache, ids, y, high, head):
    result = {}
    for name, row in cache.items():
        z = F.linear(row["student"][ids], head[:, :-1], head[:, -1])
        label = y[ids]
        p = conditional(prob(z), label)
        baseline = statistics(row["parent"][ids], label, high[ids])
        new = statistics(z, label, high[ids])
        result[name] = {"target_kl": float(divergence(conditional(row["dual"][ids], label), p).mean()),
                        "random_target_kl": float(divergence(conditional(row["random"][ids], label), p).mean()),
                        "max_moment_drift": float(((new-baseline).abs()/baseline.clamp_min(.25)).max())}
    return result


def fit_occlusion(model, raw, y, original, incumbent):
    counts = y.bincount(minlength=100)
    high_classes = torch.where(counts >= .96 * counts.max())[0]
    high = torch.isin(y, high_classes)
    logprior = prior(y, 100)
    folds = _split(y.numpy())
    fit, cal, held = [np.flatnonzero(mask) for mask in (folds < 3, folds == 3, folds == 4)]
    cache, effects = {}, {}
    for name, pixels in (("exact", raw), ("crop", transform_batch(raw.numpy(), "crop"))):
        model.load_state_dict(original, strict=True)
        targeted, random, clean = occluded_logits(model, pixels, y, 24, 22611)
        def margin(z):
            return z.gather(1,y[:,None]).squeeze(1) - z.scatter(1,y[:,None],-torch.inf).logsumexp(1)
        effects[name] = {"targeted_margin_drop":float((margin(clean)-margin(targeted))[held].mean()),
                         "random_margin_drop":float((margin(clean)-margin(random))[held].mean())}
        model.load_state_dict(incumbent, strict=True)
        x, z, _ = encode(model, pixels, 48, "cuda")
        z = z.cpu()
        cache[name] = {"student":x.cpu(), "parent":z, "dual":target(z,targeted,y,logprior),
                       "random":target(z,random,y,logprior)}
        print(name, effects[name], flush=True)
    base = head_matrix(model)
    # Both students see the same number of samples and are optimized equally.
    matched = train_student(cache,y,fit,base,high_classes)
    random_cache = {k:{**row,"dual":row["random"]} for k,row in cache.items()}
    random_head = train_student(random_cache,y,fit,base,high_classes)
    scores = {}
    for strength in (.25, .5, .75, 1.):
        h = torch.lerp(base,matched,strength)
        scores[strength] = assess(cache,cal,y,high,h)
    eligible = [s for s,rows in scores.items() if max(r["max_moment_drift"] for r in rows.values()) <= .22]
    if not eligible:
        eligible = [.125]
    strength = min(eligible, key=lambda s:sum(r["target_kl"] for r in assess(cache,cal,y,high,torch.lerp(base,matched,s)).values()))
    head = torch.lerp(base,matched,strength)
    evaluation = {key:assess(cache,held,y,high,h) for key,h in
                  (("incumbent",base),("salient",head),("random",torch.lerp(base,random_head,strength)))}
    trial = {"method":"Class-supporting tile occlusion", "tile_count":12, "available_tiles":49,
               "split_sizes":[len(fit),len(cal),len(held)], "strength":strength, "calibration":scores,
               "held":evaluation, "held_occlusion_effect":effects}
    return state_with_head(incumbent,head), trial


@torch.inference_mode(False)
@torch.enable_grad()
def unlearn(model, forget_images, *, cpu_readout=None):
    """Fit the occlusion student from the original ViT and its unlabeled forget pixels on CUDA."""
    if model.head.weight.device.type != "cuda" or official.DEVICE != "cuda":
        raise ValueError("Wildcard attribution and official evaluation require CUDA")
    if forget_images.dtype != torch.uint8 or tuple(forget_images.shape) != (3500,224,224,3):
        raise ValueError("Expected 3,500 original RGB forget images")
    raw = forget_images.detach().cpu()
    teacher = deepcopy(model).eval().requires_grad_(False)
    original = {k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    y = official.predict(teacher,raw,48).argmax(1)
    fit_incumbent(teacher,raw,cpu_readout=cpu_readout)
    incumbent = {k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    state, _ = fit_occlusion(teacher,raw,y,original,incumbent)
    validate_state(state,original)
    model.load_state_dict(state,strict=True)
    return model.eval()
