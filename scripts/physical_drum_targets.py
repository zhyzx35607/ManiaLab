"""Partial physical labels: -1 means unknown, never a negative example."""
import numpy as np


def known_targets(targets):
    targets=np.asarray(targets,dtype=np.float32)
    if (targets.ndim!=2 or targets.shape[1]!=3 or not np.isfinite(targets).all()
            or ((targets<0)&(targets!=-1)).any() or (targets>1).any()):
        raise ValueError('Invalid partial physical labels')
    known=targets>=0
    return np.where(known,targets,0).astype(np.float32),known
