import numpy as np
import torch


def attenuate(head, features, labels, scale):
    center = features.mean(0)
    means = torch.stack([features[labels == c].mean(0) - center for c in np.unique(labels)])
    _, values, vh = torch.linalg.svd(means, full_matrices=False)
    basis = vh[values > values[0] * 1e-5]
    correction = (scale - 1) * ((head[:, :-1] @ basis.T) @ basis)
    new = head.clone()
    new[:, :-1] += correction
    new[:, -1] -= correction @ center
    return new, len(basis)
