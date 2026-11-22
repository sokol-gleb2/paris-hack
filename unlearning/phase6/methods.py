import torch


def apply_head(features, head):
    return features @ head[:, :-1].T + head[:, -1]


class Ridge:
    def __init__(self, features, penalty):
        self.scale = features.square().mean().sqrt().clamp_min(1e-6)
        self.design = torch.cat((features / self.scale, torch.ones_like(features[:, :1])), dim=1)
        gram = self.design.T @ self.design / len(features)
        gram.diagonal().add_(penalty)
        self.factor = torch.linalg.cholesky(gram)

    def fit(self, delta):
        coefficients = torch.cholesky_solve(self.design.T @ delta / len(delta), self.factor)
        coefficients[:-1] /= self.scale
        return coefficients.T


def match_ce(reference, endpoint, features, labels, target_mean):
    """Interpolate toward an edit to match the selected mean pseudo-label CE."""
    logits = apply_head(features, reference)
    step = apply_head(features, endpoint) - logits
    labels = torch.as_tensor(labels, dtype=torch.long)

    def loss(scale):
        return float(torch.nn.functional.cross_entropy(logits + scale * step, labels))

    if loss(0) >= target_mean or loss(1) < target_mean:
        return None
    low, high = 0., 1.
    for _ in range(24):
        mid = (low + high) / 2
        if loss(mid) < target_mean:
            low = mid
        else:
            high = mid
    return reference + high * (endpoint - reference)
