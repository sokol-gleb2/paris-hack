import numpy as np
import torch
from phase7.local_mlp import composites


def pairs(labels, pool, seed):
    rng = np.random.default_rng(seed)
    partner = {}
    for cls in np.unique(labels[pool]):
        group = rng.permutation(pool[labels[pool] == cls])
        if len(group) < 2:
            raise ValueError("Same-class controls need at least two parents per partition")
        partner.update(zip(group, np.roll(group,-1)))
    return partner


def blend(images, ids, partner_ids):
    # Blend continuously across the whole frame. The weights stay within [.15,.85].
    axis = torch.linspace(-torch.pi,torch.pi,images.shape[1])
    weights = (.5 + .35*axis.sin()[:,None]*axis.cos()[None,:])[None,:,:,None]
    result = torch.empty((len(ids), *images.shape[1:]),dtype=images.dtype)
    for start in range(0,len(ids),32):
        stop = min(start+32,len(ids))
        a = images[ids[start:stop]].float()
        b = images[partner_ids[start:stop]].float()
        result[start:stop] = torch.lerp(a,b,weights).round().clamp(0,255).to(torch.uint8)
    return result


def held_smooth(images, labels, held_indices, selected_indices, seed=590):
    """Independent held partners; avoid training pairing where a third parent exists."""
    pool = np.asarray(held_indices,dtype=np.int64)
    ids = np.asarray(selected_indices,dtype=np.int64)
    if not np.isin(ids,pool).all():
        raise ValueError("Every smooth source must belong to the requested partition")
    order = np.random.default_rng(seed)
    training_pair = pairs(labels,pool,589)
    chosen = np.empty(len(ids),dtype=np.int64)
    for i,src in enumerate(ids):
        group = pool[(labels[pool] == labels[src]) & (pool != src)]
        if len(group)>1:
            group = group[group != training_pair[src]]
        chosen[i] = order.choice(group)
    return blend(images,ids,chosen), chosen.tolist()


def training_controls(images, labels, split):
    """Half quadrant, half smooth pairs in each pseudo-class and partition."""
    output = composites(images,labels,split,seed=363)
    rng = np.random.default_rng(589)
    for partition in np.unique(split):
        pool = np.flatnonzero(split == partition)
        partner = pairs(labels,pool,589)
        ids = np.concatenate([rng.permutation(pool[labels[pool] == c])[::2]
                              for c in np.unique(labels[pool])])
        destinations = np.array([partner[i] for i in ids])
        output[ids] = blend(images,ids,destinations)
    return output
