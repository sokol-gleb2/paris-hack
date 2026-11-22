import numpy as np
import torch
from phase7.subspace import attenuate
from phase7.paired_retrain import combined, paired_edit, prototypes


def fit_readout(head, features, labels):
    geometry, _ = attenuate(head, features, labels, .25)
    counts = np.bincount(labels, minlength=len(head))
    if np.count_nonzero(counts) < 2 or np.all(counts == counts.max()):
        return geometry
    means, _ = prototypes(head, features, labels)
    delta, _ = paired_edit(head, features, labels, means, "cap")
    fitted = combined(head, features, labels, geometry, delta, .5)
    if fitted is None:
        return geometry
    return 1.5 * (head + 1.12 * (fitted - head))


@torch.no_grad()
def apply_readout_fit(model, forget_images, batch_size, fit):
    """Batch-encode the supplied pixels, fit a CPU readout and install it in place."""
    if batch_size <= 0 or forget_images.shape[0] == 0:
        raise ValueError("Provide at least one forget image and a positive batch size")
    if forget_images.dtype != torch.uint8 or tuple(forget_images.shape[1:]) != (224, 224, 3):
        raise ValueError("Expected uint8 images of shape [N, 224, 224, 3]")
    device = model.head.weight.device
    if model.head.weight.dtype != torch.float32:
        raise ValueError("Expected the supplied float32 model")
    features = []
    labels = []
    model.eval()
    for images in forget_images.split(batch_size):
        pixels = images.to(device).permute(0, 3, 1, 2).float() / 255.
        pixels = (pixels - .5) / .5
        encoded = model.forward_head(model.forward_features(pixels), pre_logits=True)
        features.append(encoded.cpu())
        labels.append(model.head(encoded).argmax(1).cpu())
    head = torch.cat([model.head.weight.cpu(), model.head.bias.cpu()[:, None]], dim=1)
    fitted = fit(head, torch.cat(features), torch.cat(labels).numpy())
    if fitted.shape != head.shape or fitted.dtype != head.dtype or not torch.isfinite(fitted).all():
        raise ValueError("The fitted classifier must be finite and match the supplied head shape and dtype")
    model.head.weight.copy_(fitted[:, :-1])
    model.head.bias.copy_(fitted[:, -1])
    return model
