from contextlib import nullcontext
from threading import Lock
import numpy as np
import torch
from phase6.run import official, transform_batch
from phase2.build_folds import make_folds
from phase7.local_mlp import extract
from phase7.solution import apply_readout_fit, fit_readout
from phase8.controls import held_smooth, training_controls
from phase8.partial_counterfactual import difficulty, encode, solve_head, synthetic_anchors
from phase8.selective import fit, logits


_LOCK = Lock()


def _split(labels):
    result = np.empty(len(labels), dtype=np.int64)
    if len(labels) == 3500:
        for i, indices in enumerate(make_folds(labels, seed=2026).values()):
            result[indices] = i
        return result
    # The archived make_folds validates the competition's exact sample count.
    # Use its class-wise shuffle and continuing round robin on smaller inputs.
    rng = np.random.default_rng(2026)
    order = np.concatenate([rng.permutation(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    result[order] = np.arange(len(order)) % 5
    return result


def _smooth_features(model, raw, labels, pool, n, seed, device):
    rng = np.random.default_rng(seed)
    # Same-class blend generation needs two distinct source images per pool.
    groups = [pool[labels[pool] == c] for c in np.unique(labels[pool])]
    supported = [g for g in groups if len(g) >= 2]
    if not supported:
        return model.head.weight.new_empty((0, model.head.in_features))
    eligible_pool = np.sort(np.concatenate(supported))
    panel = np.sort(np.concatenate([rng.permutation(g)[:n] for g in supported]))
    images, _ = held_smooth(raw, labels, eligible_pool, panel, seed)
    return encode(model, images, 32, device)[0]


def _partial_head(model, raw, labels, folds, original_weight):
    device = model.head.weight.device.type
    y = torch.from_numpy(labels).to(model.head.weight.device)
    counts = y.bincount(minlength=model.head.out_features)
    peak = counts[y] >= .96 * counts.max()
    base = torch.cat((model.head.weight, model.head.bias[:, None]), dim=1).detach()
    if int(peak.sum()) < 2 or int((~peak).sum()) < 2:
        return base
    exact, _, early = encode(model, raw, 32, device, early=True)
    bright, _, _ = encode(model, transform_batch(raw.numpy(), "brightness"), 32, device)
    smooth_fit = _smooth_features(model, raw, labels, np.flatnonzero(folds != 0), 10, 30859, device)
    smooth_held = _smooth_features(model, raw, labels, np.flatnonzero(folds == 0), 4, 31601, device)
    ranks, _ = difficulty(early, y, torch.from_numpy(folds).to(y.device),
                          torch.arange(len(y), device=y.device))
    anchors = torch.cat((smooth_fit, smooth_held, synthetic_anchors(original_weight, exact, y, 7021)))
    result, _ = solve_head(exact, bright, anchors, base, y, ranks, counts, .8)
    return result


def _fit_stages(model, raw, sparse_scale, partial_scale, observe, cpu_readout):
    device = model.head.weight.device
    exact = extract(model, raw, 48, device.type)
    y = exact["logits"].argmax(1)
    labels = y.cpu().numpy()
    folds = _split(labels)
    # Archived full-data training kept calibration, held and fit parents in separate pools.
    pools = np.minimum(folds, 2)
    counts = np.zeros((3, model.head.out_features), dtype=int)
    np.add.at(counts, (pools, labels), 1)
    eligible = np.flatnonzero(counts[pools, labels] >= 2)
    original_weight = model.head.weight.detach().clone()
    proxy = None
    if sparse_scale and len(eligible):
        mixed = training_controls(raw[eligible], labels[eligible], pools[eligible])
        proxy = extract(model, mixed, 48, device.type)
    torch.set_num_threads(3)
    if cpu_readout is None:
        apply_readout_fit(model, raw, 32, fit_readout)
    else:
        if (cpu_readout.shape != (model.head.out_features, model.head.in_features + 1)
                or cpu_readout.dtype != torch.float32 or not torch.isfinite(cpu_readout).all()):
            raise ValueError("Invalid CPU-fitted classifier")
        with torch.no_grad():
            model.head.weight.copy_(cpu_readout[:, :-1])
            model.head.bias.copy_(cpu_readout[:, -1])
    if proxy is not None:
        torch.set_num_threads(4)
        ids = torch.as_tensor(eligible, device=device)
        subset = {k: v[ids] for k, v in exact.items()}
        subset["winner"] = logits(model, subset).detach()
        delta, _, _ = fit(model, subset, proxy, y[ids], torch.arange(len(ids), device=device), 16, 120)
        with torch.no_grad():
            old = model.blocks[-1].mlp.fc2.weight
            fitted = old + delta
            if sparse_scale != 1.0:
                fitted = torch.lerp(old.cpu(), fitted.cpu(), sparse_scale)
            old.copy_(fitted)
    if observe:
        observe("sparse", model)
    if partial_scale:
        torch.set_num_threads(3)
        corrected = _partial_head(model, raw, labels, folds, original_weight)
        parent = torch.cat((model.head.weight, model.head.bias[:, None]), dim=1)
        if partial_scale != 1.0:
            corrected = torch.lerp(parent.cpu(), corrected.cpu(), partial_scale)
        if not torch.isfinite(corrected).all():
            raise ValueError("Nonfinite partial correction")
        with torch.no_grad():
            model.head.weight.copy_(corrected[:, :-1])
            model.head.bias.copy_(corrected[:, -1])
        if observe:
            observe("partial", model)


@torch.inference_mode(False)
@torch.enable_grad()
def _run(model, forget_images, sparse_scale, partial_scale, observe=None, cpu_readout=None):
    if (forget_images.dtype != torch.uint8 or forget_images.ndim != 4
            or tuple(forget_images.shape[1:]) != (224, 224, 3) or not len(forget_images)):
        raise ValueError("Expected nonempty uint8 [N,224,224,3] images")
    if (model.head.weight.dtype != torch.float32 or model.head.out_features != 100
            or model.head.weight.device.type not in ("cpu", "cuda")):
        raise ValueError("Use the official float32 model on CPU or CUDA")
    if not np.isfinite([sparse_scale, partial_scale]).all():
        raise ValueError("Expected finite update scales")
    dev = model.head.weight.device
    # The historical helper reads its active device from the official template.
    # Restore global arithmetic settings and each caller gradient flag even if fitting fails.
    with _LOCK, (torch.cuda.device(dev) if dev.type == "cuda" else nullcontext()):
        previous = (official.DEVICE, torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32)
        params = list(model.parameters())
        trainable = [p.requires_grad for p in params]
        edits = (model.head.weight, model.head.bias, model.blocks[-1].mlp.fc2.weight)
        backups = [p.detach().clone() for p in edits]
        try:
            official.DEVICE = dev.type
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model.eval().requires_grad_(False)
            _fit_stages(model, forget_images.detach().cpu(), sparse_scale, partial_scale, observe, cpu_readout)
        except BaseException:
            with torch.no_grad():
                for param, saved in zip(edits, backups):
                    param.copy_(saved)
            raise
        finally:
            for p, enabled in zip(params, trainable):
                p.requires_grad_(enabled)
            official.DEVICE, threads, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
            torch.set_num_threads(threads)
    return model


def unlearn(model, forget_images, *, sparse_scale=1., partial_scale=1., cpu_readout=None):
    """Fit from source; optionally supply the separately reproduced three-thread CPU readout.

    A partial scale of zero returns the sparse 3321 recipe. The default unit
    scale adds the 3468 correction. The readout is a [100, 769] float32 tensor.
    """
    return _run(model, forget_images, sparse_scale, partial_scale, cpu_readout=cpu_readout)
