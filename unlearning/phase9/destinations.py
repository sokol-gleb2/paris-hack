import numpy as np
import torch
from torch.nn import functional as F


def head_matrix(model):
    return torch.cat((model.head.weight, model.head.bias[:, None]), dim=1).detach().float().cpu()


def logits(x, head):
    return F.linear(x, head[:, :-1], head[:, -1])


def conditional(prob, labels):
    other = prob.scatter(1, labels[:, None], 0)
    return other / other.sum(1, keepdim=True).clamp_min(1.e-25)


def prob(z):
    return (z / 2).softmax(1)


def divergence(p, q):
    return (p * (p.clamp_min(1.e-25).log() - q.clamp_min(1.e-25).log())).sum(1)


def shuffle_indices(y, seed):
    rng = np.random.default_rng(seed)
    perm = np.arange(len(y))
    for c in np.unique(y):
        ids = rng.permutation(np.where(y == c)[0])
        perm[ids] = np.roll(ids, 1)
    return perm


def moments(z, y, peak):
    losses = F.cross_entropy(z, y, reduction='none')
    return torch.stack([s for mask in (peak, ~peak) for s in (losses[mask].mean(), losses[mask].std(unbiased=False))])
