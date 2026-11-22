import numpy as np
import torch
from torch.nn import functional as F
from phase9.destinations import conditional, logits, moments, prob, shuffle_indices
from phase9.destinations_prior import VIEWS


def fixed_concentration(q, target, y):
    # Increasing the exponent monotonically increases the sum of squared probabilities.
    # Remove source logits exactly throughout rather than exponentiating their sentinel.
    q = q.double()
    index = y[:,None]
    logq = q.clamp_min(1.e-60).log()
    def powered(power):
        return (power*logq).scatter(1,index,-torch.inf).softmax(1)
    low = torch.zeros_like(target)
    high = torch.ones_like(target)
    for _ in range(24):
        insufficient = powered(high).square().sum(1,keepdim=True) < target
        if not insufficient.any():
            break
        high = torch.where(insufficient,high*2,high)
    for _ in range(42):
        mid = (low+high)/2
        below = powered(mid).square().sum(1,keepdim=True) < target
        low = torch.where(below,mid,low)
        high = torch.where(below,high,mid)
    result = powered((low+high)/2)
    if (result.square().sum(1,keepdim=True)-target).abs().max() > 1.e-8:
        raise ValueError('Conditional concentration cannot be represented by this temperature family.')
    return result


def choices(teacher,parent,y,logprior,seed):
    p = prob(parent.double())
    src = p.gather(1, y[:,None])
    wrong = conditional(p,y)
    ref = conditional(prob(teacher.double()+2*logprior[None,:]),y)
    rng = np.random.default_rng(seed)
    fake = conditional(prob(teacher.double()+2*logprior[rng.permutation(len(logprior))][None,:]),y)
    perm = shuffle_indices(y.numpy(),seed+17)
    proposed = .5*wrong+.5*ref
    concentration=wrong.square().sum(1,keepdim=True)
    q={
        'raw':proposed,'dual':fixed_concentration(proposed,concentration,y),
        'shuffled':fixed_concentration(.5*wrong+.5*ref[perm],concentration,y),
        'permuted_prior':fixed_concentration(.5*wrong+.5*fake,concentration,y)}
    output={'parent':p}
    for k,v in q.items():
        output[k]=((1-src)*v).scatter(1,y[:,None],src)
        if k!='raw':
            ce=F.cross_entropy(2*output[k].clamp_min(1.e-60).log(),y,reduction='none')
            base=F.cross_entropy(parent.double(),y,reduction='none')
            if (ce-base).abs().max() > 1.e-6:
                raise ValueError('Dual target changed the exact source loss.')
    return output


def train_student(cache,y,fit,base,high):
    x=torch.stack([cache[v]['student'][fit] for v in VIEWS])
    old=logits(x,base)
    truth=torch.stack([cache[v]['dual'][fit] for v in VIEWS])
    lbl=y[fit]
    peak=torch.isin(lbl,high)
    source_losses=torch.stack([F.cross_entropy(z,lbl,reduction='none') for z in old])
    target_m=torch.stack([moments(z,lbl,peak) for z in old])
    delta=torch.zeros_like(base,requires_grad=True)
    opt=torch.optim.LBFGS([delta],lr=.85,max_iter=100,history_size=8,line_search_fn='strong_wolfe')
    mask=torch.zeros_like(old,dtype=torch.bool)
    mask.scatter_(2,lbl[None,:,None].expand(len(VIEWS),-1,-1),True)
    wrong_truth=truth.masked_fill(mask,0)
    wrong_truth=wrong_truth/wrong_truth.sum(2,keepdim=True).clamp_min(1.e-12)
    def closure():
        opt.zero_grad()
        z=logits(x,base+delta)
        full=-(truth*(z/2).log_softmax(2)).sum(2).mean()
        logwrong=(z/2).masked_fill(mask,-1.e4).log_softmax(2)
        wrong=-(wrong_truth*logwrong).sum(2).mean()
        losses=torch.stack([F.cross_entropy(v,lbl,reduction='none') for v in z])
        m=torch.stack([moments(v,lbl,peak) for v in z])
        norm=(losses-source_losses)/(1+source_losses)
        loss=(full+.4*wrong+.4*F.smooth_l1_loss(norm,torch.zeros_like(norm),beta=.12)
              +10*((m-target_m)/target_m.clamp_min(.25)).square().mean()+.01*delta.square().sum()/len(base))
        loss.backward()
        return loss
    opt.step(closure)
    return base+delta.detach()
