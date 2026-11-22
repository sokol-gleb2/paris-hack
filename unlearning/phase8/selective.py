import numpy as np
import torch
from torch.nn import functional as F
from phase7.local_mlp import subset


SEED = 835


def logits(model, data, delta=None):
    u = data["u"]
    if delta is not None:
        u = u + F.linear(data["a"], delta) * model.blocks[-1].ls2.gamma
    return model.head(model.norm(u))


def divergence(predicted, target):
    """Per-row KL(target || predicted) at temperature 2."""
    target_log = (target / 2).log_softmax(1)
    return (target_log.exp() * (target_log - (predicted / 2).log_softmax(1))).sum(1)


def mask_for(model, exact, proxy, ids, count, random):
    """Rank each output's input weights by repair gradient per unit of source sensitivity."""
    rng = torch.Generator(device=exact["a"].device).manual_seed(SEED)
    width = exact["a"].shape[1]
    outputs = exact["u"].shape[1]
    if random:
        return torch.rand((outputs, width), device=exact["a"].device, generator=rng).topk(count, dim=1).indices
    retain_grad = exact["a"].new_zeros(outputs, width)
    fisher = torch.zeros_like(retain_grad)
    gamma = model.blocks[-1].ls2.gamma.detach()
    for batch in ids.split(128):
        a = proxy["a"][batch]
        u = proxy["u"][batch].detach().requires_grad_(True)
        z = model.head(model.norm(u))
        repair = divergence(z, proxy["logits"][batch]).sum()
        derivative = torch.autograd.grad(repair, u)[0].detach() * gamma
        retain_grad.add_(derivative.T @ a)
        u = exact["u"][batch].detach().requires_grad_(True)
        z = model.head(model.norm(u)) / 2
        drawn = torch.multinomial(z.detach().softmax(1), 1, generator=rng).flatten()
        sample_nll = F.cross_entropy(z, drawn, reduction="sum")
        derivative = torch.autograd.grad(sample_nll, u)[0].detach() * gamma
        fisher.add_(derivative.square().T @ exact["a"][batch].square())
    denom = fisher / len(ids)
    score = retain_grad.abs() / (denom + .05 * denom.mean(1, keepdim=True) + 1e-12).sqrt()
    if not score.isfinite().all() or not torch.any(score > 0):
        raise ValueError("Invalid saliency scores")
    return score.topk(count, dim=1).indices


def objective(candidate, exact_target, labels, repaired, proxy_target):
    loss = F.cross_entropy(candidate, labels, reduction="none")
    anchored = F.cross_entropy(exact_target, labels, reduction="none")
    return (divergence(candidate, exact_target).mean()
            + .22 * F.smooth_l1_loss(loss, anchored, beta=.4)
            + divergence(repaired, proxy_target).mean())


def fit(model, exact, proxy, labels, ids, count, steps, random=False):
    mask = mask_for(model, exact, proxy, ids, count, random=random)
    base = model.blocks[-1].mlp.fc2.weight.detach()
    bound = 8. * base.square().mean(1, keepdim=True).sqrt()
    coefficients = torch.zeros_like(mask, dtype=base.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=.09)
    rng = torch.Generator(device=base.device).manual_seed(SEED + 19)
    batch_size = min(128, len(ids))
    # Fixed weights retain gradient only for the selected coordinates.
    first_gradient = None
    for _ in range(steps):
        e = ids[torch.randint(len(ids), (batch_size,), generator=rng, device=base.device)]
        p = ids[torch.randint(len(ids), (batch_size,), generator=rng, device=base.device)]
        delta = torch.zeros_like(base).scatter(1, mask, bound * coefficients.tanh())
        z = logits(model, subset(exact, e), delta)
        q = logits(model, subset(proxy, p), delta)
        loss = objective(z, exact["winner"][e], labels[e], q, proxy["logits"][p])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if first_gradient is None:
            first_gradient = float(coefficients.grad.norm())
        optimizer.step()
    update = torch.zeros_like(base).scatter(1, mask, bound * coefficients.detach().tanh())
    if not first_gradient or not np.isfinite(first_gradient) or not torch.isfinite(update).all():
        raise ValueError("Invalid sparse fit")
    if torch.count_nonzero(update) > count * len(base) or not update.norm() > 0:
        raise ValueError("Mask budget or nonzero update check failed")
    return update, mask, first_gradient
