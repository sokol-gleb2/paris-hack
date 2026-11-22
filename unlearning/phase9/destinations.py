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


def moments(z, y, peak):
    losses = F.cross_entropy(z, y, reduction='none')
    return torch.stack([s for mask in (peak, ~peak) for s in (losses[mask].mean(), losses[mask].std(unbiased=False))])
