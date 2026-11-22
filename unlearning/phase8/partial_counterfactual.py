import numpy as np
import torch
from torch.nn import functional as F
from phase6.run import official


@torch.no_grad()
def encode(model,raw,batch,device,early=False):
    x,z,e=[],[],[]
    h=model.head.register_forward_pre_hook(lambda mod,inp:x.append(inp[0].detach().float()))
    if early:
        def grab(mod,inp,out):
            patches=out[:,model.num_prefix_tokens:]
            e.append(torch.cat((out[:,0],patches.mean(1)),1).detach().float())
        h2=model.blocks[7].register_forward_hook(grab)
    try:
        for im in raw.split(batch):
            with torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'):
                z.append(model(official.preprocess(im)).float())
    finally:
        h.remove()
        if early:
            h2.remove()
    return torch.cat(x),torch.cat(z),torch.cat(e) if early else None


def difficulty(early,y,folds,ids):
    """Weighted one-vs-all ridge on only the represented original labels, with per-class OOF ranks."""
    device=early.device
    classes,inv=torch.unique(y,return_inverse=True)
    width=len(classes)
    t=F.one_hot(inv,width).float()
    generator=torch.Generator(device=device).manual_seed(60531)
    proj=torch.randn((early.shape[1],min(256,early.shape[1])),generator=generator,device=device)
    v=F.normalize(F.layer_norm(early,early.shape[1:])@proj,dim=1)
    trainmask=torch.zeros(len(y),device=device,dtype=torch.bool)
    trainmask[ids]=True
    oof=v.new_zeros(len(y))
    for fold in torch.unique(folds[ids]):
        fit=trainmask&(folds!=fold)
        test=trainmask&(folds==fold)
        cnt=torch.bincount(inv[fit],minlength=width).float()
        kept=(cnt.max()-cnt).clamp_min(.075*cnt.max())
        weight=kept[inv[fit]]/cnt[inv[fit]]
        weight/=weight.mean()
        X=v[fit]
        gram=X.T@(X*weight[:,None])
        gram.diagonal().add_(.30)
        coef=torch.linalg.solve(gram,X.T@(weight[:,None]*t[fit]))
        z=15*v[test]@coef
        oof[test]=F.cross_entropy(z,inv[test],reduction='none')
    outside=~trainmask
    if outside.any():
        cnt=torch.bincount(inv[ids],minlength=width).float()
        kept=(cnt.max()-cnt).clamp_min(.075*cnt.max())
        weight=kept[inv[ids]]/cnt[inv[ids]]
        weight/=weight.mean()
        X=v[ids]
        gram=X.T@(X*weight[:,None])
        gram.diagonal().add_(.30)
        coef=torch.linalg.solve(gram,X.T@(weight[:,None]*t[ids]))
        oof[outside]=F.cross_entropy(15*v[outside]@coef,inv[outside],reduction='none')
    ranks=torch.empty_like(oof)
    shuffled=torch.empty_like(oof)
    for c in classes:
        ref=oof[trainmask&(y==c)].sort().values
        mask=y==c
        ranks[mask]=(torch.searchsorted(ref,oof[mask])+.5).div(len(ref)).clamp(max=1.)
        for use in (trainmask,outside):
            ix=(mask&use).nonzero().flatten()
            shuffled[ix]=ranks[ix[torch.randperm(len(ix),generator=generator,device=device)]]
    return ranks,shuffled


def design(v,scale):
    return torch.cat((v/scale,torch.ones_like(v[:,:1])),1)


def target_edit(z,y,rank,peak,mean,counts):
    ce=F.cross_entropy(z,y,reduction='none')
    # Rare remaining classes have fewer retained examples; moderate their relative weight.
    w=(.09+rank.pow(3)*3.6)*(.75+.5*counts[y]/counts.max())
    target=ce.clone()
    w=w[~peak]/w[~peak].mean()
    target[~peak]+=w*(mean-ce[~peak].mean()).clamp_min(0)
    other=z.scatter(1,y[:,None],-torch.inf).logsumexp(1)
    shift=(other-z.gather(1,y[:,None]).flatten())-torch.log(torch.expm1(target.clamp(.00005,35)))
    return shift.clamp(-20,12),ce


def solve_head(x,xa,anchor,base,y,ranks,counts,goal):
    """Shared regularized solve, class-specific positive examples, three loss calibration directions."""
    z,za=F.linear(x,base[:,:-1],base[:,-1]),F.linear(xa,base[:,:-1],base[:,-1])
    peak=counts[y]>=.96*counts.max()
    scale=x.square().mean().sqrt()
    X,A,C=[design(v,scale) for v in (x,xa,anchor)]
    gram=.25*(X.T@X)/len(X)+.15*(A.T@A)/len(A)+.60*(C.T@C)/len(C)
    gram.diagonal().add_(.09)
    chol=torch.linalg.cholesky(gram)
    raw,ce=target_edit(z,y,ranks,peak,goal,counts)
    rawa,cea=target_edit(za,y,ranks,peak,goal,counts)
    directions=x.new_zeros((3,*base.shape))
    def standardized(val):
        return ((val-val.mean())/val.std(unbiased=False).clamp_min(.1)).clamp(-3,3)
    for c in y.unique():
        mask=(y==c)
        vx=torch.cat((X[mask],A[mask]))
        if counts[c]>=.96*counts.max():
            ref=torch.cat((ce[mask],cea[mask]))
            response=torch.stack((torch.zeros_like(ref),torch.ones_like(ref),standardized(ref)),1)
        else:
            ref=torch.cat((raw[mask],rawa[mask]))
            response=torch.stack((ref,torch.zeros_like(ref),torch.zeros_like(ref)),1)
        # Woodbury form avoids a large class-specific factorization, including in the full refit.
        inv=torch.cholesky_solve(vx.T,chol)
        local=vx@inv
        local.diagonal().add_(.30*len(vx))
        directions[:,c]=(inv@torch.linalg.solve(local,response)).T
    directions[:,:,:-1]/=scale
    curves=torch.stack([F.linear(x,v[:,:-1],v[:,-1]) for v in directions])
    targets=torch.stack([ce.new_tensor(goal),ce[peak].mean(),ce[peak].std(unbiased=False)])
    coeff=x.new_tensor([1.,0.,0.])
    def stats(a):
        h=z+torch.einsum('i,ijk->jk',a,curves)
        l=F.cross_entropy(h,y,reduction='none')
        return torch.stack([l[~peak].mean(),l[peak].mean(),l[peak].std(unbiased=False)])
    for _ in range(14):
        err=stats(coeff)-targets
        if float(err.abs().max())<.001:
            break
        j=torch.autograd.functional.jacobian(stats,coeff)
        update=torch.linalg.solve(j.T@j+torch.eye(3,device=x.device)*.00001,j.T@err)
        before=err.square().sum()
        for step in (1.,.5,.25,.125,.0625):
            proposed=coeff-step*update
            if (stats(proposed)-targets).square().sum()<before:
                coeff=proposed.detach()
                break
        else:
            break
    fixed=base+torch.einsum('i,ijk->jk',coeff,directions)
    return fixed.contiguous(),{'calibration':coeff.tolist(),'fit_moments':stats(coeff).tolist(),'targets':targets.tolist()}


def synthetic_anchors(original,x,y,seed):
    """Balance all output classes, using the original readout to infer missing class centers."""
    from phase7.paired_retrain import predict_centroids
    rng=np.random.default_rng(seed)
    classes=np.unique(y.cpu().numpy())
    empirical=torch.stack([x[y==c].mean(0) for c in classes])
    rows=F.normalize(original-original.mean(0),dim=1)
    means=predict_centroids(rows,classes,empirical,np.arange(len(classes)),.3)
    means[classes]=empirical
    res=x-means[y]
    cls=torch.arange(len(means),device=x.device).repeat_interleave(8)
    return means[cls]+res[rng.integers(0,len(res),len(cls))]
