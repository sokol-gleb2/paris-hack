import torch
from torch.nn import functional as F


def statistics(z, y, high):
    loss = F.cross_entropy(z, y, reduction="none")
    return torch.stack([loss[high].mean(), loss[high].std(unbiased=False),
                        loss[~high].mean(), loss[~high].std(unbiased=False)])
