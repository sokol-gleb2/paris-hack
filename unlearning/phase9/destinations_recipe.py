from contextlib import nullcontext
from threading import Lock
import numpy as np
import torch
from phase6.run import official, transform_batch
from phase7.local_mlp import extract
from phase9.destinations import head_matrix
from phase9.destinations_dual import choices, train_student
from phase9.destinations_prior import prior
from phase9.recipe import _split, unlearn as fit_partial_parent


_RECIPE_LOCK=Lock()


@torch.inference_mode(False)
@torch.enable_grad()
def unlearn(model, forget_images, *, cpu_readout=None, dual_scale=1.):
    """Fit the 3468 parent and the 0.125-strength dual-prior head in place.

    cpu_readout optionally supplies the exact three-thread 3058 intermediate derived
    from the original checkpoint and images. The plain two-argument call refits it.
    dual_scale rescales only the final scored 3468-to-dual update.
    """
    if (forget_images.dtype != torch.uint8 or tuple(forget_images.shape)!=(3500,224,224,3)
            or model.head.weight.shape!=(100,768) or not np.isfinite(dual_scale)):
        raise ValueError('Expected the official 100-class ViT and all 3,500 unlabelled uint8 images.')
    device=model.head.weight.device
    if device.type not in ('cpu','cuda'):
        raise ValueError('Use CPU or CUDA for the reproducible Task 3 pipeline.')
    raw=forget_images.detach().cpu()
    with _RECIPE_LOCK, (torch.cuda.device(device) if device.type=='cuda' else nullcontext()):
        settings=(official.DEVICE,torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32)
        edits=(model.head.weight,model.head.bias,model.blocks[-1].mlp.fc2.weight)
        old=[v.detach().clone() for v in edits]
        try:
            model.eval()
            official.DEVICE=device.type
            torch.set_num_threads(3)
            torch.backends.cuda.matmul.allow_tf32=False
            # Save the original teacher evidence before the parent fit mutates its weights.
            views={'exact':raw,'crop':transform_batch(raw.numpy(),'crop')}
            teacher={v:official.predict(model,pixels,48) for v,pixels in views.items()}
            y=teacher['exact'].argmax(1)
            roles=_split(y.numpy())
            fit=np.flatnonzero(roles<3)
            fit_partial_parent(model,raw,cpu_readout=cpu_readout)
            if dual_scale==0:
                return model
            base=head_matrix(model)
            high=torch.where(y.bincount(minlength=100)>=.96*y.bincount().max())[0]
            log_prior=prior(y,100)
            cache={}
            for v,pixels in views.items():
                packed=extract(model,pixels,48,device.type)
                with torch.no_grad():
                    xs=model.norm(packed['u']).float().cpu()
                q=choices(teacher[v],packed['logits'].detach().float().cpu(),y,log_prior,62131)
                cache[v]={'student':xs,'dual':q['dual'].float()}
            trained=train_student(cache,y,fit,base,high)
            scored_head=base+.125*(trained-base)
            final=scored_head if dual_scale==1. else torch.lerp(base,scored_head,float(dual_scale))
            if not torch.isfinite(final).all():
                raise ValueError('Nonfinite rebuilt classifier')
            with torch.no_grad():
                model.head.weight.copy_(final[:,:-1])
                model.head.bias.copy_(final[:,-1])
            return model
        except BaseException:
            with torch.no_grad():
                for tensor,saved in zip(edits,old):
                    tensor.copy_(saved)
            raise
        finally:
            official.DEVICE,threads,torch.backends.cuda.matmul.allow_tf32=settings
            torch.set_num_threads(threads)
