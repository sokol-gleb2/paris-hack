import numpy as np
import torch
from torch.nn import functional as F
from phase6.methods import Ridge, apply_head, match_ce


def cosine(a, b):
    return F.cosine_similarity(a,b).tolist()


def predict_centroids(unit_rows, ids, means, active, penalty):
    """A scaled classifier row plus ridge interpolation of residuals from observed classes."""
    c = ids[active]
    target = means[active]
    u = unit_rows[c]
    mu = target.mean(0)
    axis = unit_rows - u.mean(0)
    seen = axis[c]
    tilt = (seen*(target-mu)).sum()/seen.square().sum().clamp_min(1e-8)
    predictions = mu + tilt * axis
    if penalty is None:
        return predictions
    residual = target-predictions[c]
    gram = seen @ seen.T
    gram.diagonal().add_(penalty)
    return predictions + (axis @ seen.T) @ torch.linalg.solve(gram,residual)


def prototypes(head, features, labels):
    ids = np.unique(labels)
    means = torch.stack([features[labels==c].mean(0) for c in ids])
    rows = F.normalize(head[:,:-1]-head[:,:-1].mean(0),dim=1)
    penalties = (None, .03, .3, 1., 3.)
    folds = np.array_split(np.random.default_rng(2026).permutation(len(ids)), min(5,len(ids)))
    results = {}
    for p in penalties:
        predicted=torch.zeros_like(means)
        for fold in folds:
            fit = np.setdiff1d(np.arange(len(ids)),fold)
            predicted[fold]=predict_centroids(rows,ids,means,fit,p)[ids[fold]]
        results[p] = {"rmse":float((means-predicted).square().mean().sqrt()),
                      "median_cosine":float(np.median(cosine(means,predicted))),
                      "top1_from_original_head":float((apply_head(predicted,head).argmax(1).numpy()==ids).mean())}
    chosen = min(penalties,key=lambda p:results[p]["rmse"])
    pred = predict_centroids(rows,ids,means,np.arange(len(ids)),chosen)
    pred[ids]=means
    return pred,{"selected_penalty":chosen,"class_held_out":{str(k):v for k,v in results.items()}}


def simulate_retention(protos, features, labels, mode):
    counts=np.bincount(labels,minlength=len(protos))
    capacity=int(counts.max())
    kept = capacity-counts if mode=="cap" else np.full_like(counts,max(1,round(capacity*40/130)))
    residual=features-protos[labels]
    rng=np.random.default_rng(32401)
    xs,ys=[],[]
    for cls,n in enumerate(kept):
        if not n:
            continue
        pool=np.flatnonzero(labels!=cls)
        if not len(pool):
            pool=np.arange(len(labels))
        ids=rng.choice(pool,size=n,replace=True)
        xs.append(protos[cls]+residual[ids])
        ys += [cls]*n
    if not xs:
        raise ValueError("Retention scenario has no remaining prototypes.")
    return torch.cat(xs),np.array(ys,dtype=np.int64),kept


def fit_surrogate(x,y,width,penalty):
    return Ridge(x,penalty).fit(F.one_hot(torch.from_numpy(y),width).float())


def scale_delta(reference, full, features, labels, ids):
    """Per-output-block scalar, calibrated on reserved rows from disjoint classes."""
    orig = apply_head(features,reference)
    approx = apply_head(features,full)
    orig -= orig.mean(1,keepdim=True)
    approx -= approx.mean(1,keepdim=True)
    scales=torch.empty(len(reference))
    for block in np.array_split(np.random.default_rng(771).permutation(len(reference)),5):
        if not len(block):
            continue
        pick=ids[~np.isin(labels[ids],block)]
        if not len(pick):
            # Single observed class can only arise in the smallest simulated tasks.
            pick=ids
        x,z=approx[pick],orig[pick]
        scales[block]=max(0.,float((x*z).sum()/x.square().sum().clamp_min(1e-12)))
    return scales


def paired_edit(head, features, pseudo, protos, mode, penalty=.03):
    rng=np.random.default_rng(2026)
    calibration=[]
    for c in np.unique(pseudo):
        ids=rng.permutation(np.flatnonzero(pseudo==c))
        calibration.extend(ids[:max(1,len(ids)//5)])
    cal=np.asarray(calibration,dtype=int)
    fit=np.setdiff1d(np.arange(len(pseudo)),cal)
    retained,targets,kept=simulate_retention(protos,features,pseudo,mode)
    pre=fit_surrogate(torch.cat((retained,features[fit])),np.r_[targets,pseudo[fit]],len(head),penalty)
    scales=scale_delta(head,pre,features,pseudo,cal)
    full=fit_surrogate(torch.cat((retained,features)),np.r_[targets,pseudo],len(head),penalty)
    gold=fit_surrogate(retained,targets,len(head),penalty)
    delta=(gold-full)*scales[:,None]
    target_t=torch.from_numpy(targets)
    classifications=apply_head(retained,head).argmax(1)
    was_absent=~np.isin(targets,pseudo)
    diag={"synthetic_count":int(len(targets)),"retained_counts":kept.tolist(),
          "calibration_row_count":int(len(cal)),"scales":scales.tolist(),
          "class_agreement_synthetic_with_original":float((classifications==target_t).float().mean()),
          "class_agreement_absent_with_original":(
              float((classifications[was_absent]==target_t[was_absent]).float().mean()) if was_absent.any() else None),
          "synth_feature_norm_median":float(retained.norm(dim=1).median()),
          "real_feature_norm_median":float(features.norm(dim=1).median()),
          "full_reconstructs_forget_top1":float((apply_head(features,full).argmax(1).numpy()==pseudo).mean()),
          "gold_calibration_source":"held-forget subset omitted from interim full fit; disjoint pseudo-classes per output block"}
    return delta,diag


def combined(head, x, labels, geom, delta, ratio):
    """Compare paired destination evidence at the geometric base's mean pseudo-label loss."""
    blend=(1-ratio)*geom+ratio*(head+delta)
    y=torch.from_numpy(labels)
    target=float(F.cross_entropy(apply_head(x,geom),y))
    endpoint=head+2*(blend-head)
    return match_ce(head,endpoint,x,y,target)
