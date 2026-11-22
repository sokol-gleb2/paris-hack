import torch


VIEWS = ('exact','crop')


def prior(pseudo,width):
    counts = torch.bincount(pseudo,minlength=width).float()
    return ((counts.max()-counts+.10*counts.max())/(1.10*counts.max())).log()
