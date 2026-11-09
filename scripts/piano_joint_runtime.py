"""Cached two-head adaptation with the original detached frame conditioning.

No weights, audio or datasets are loaded here. Torch is imported only when a
runtime function is called, so the application does not gain a torch import.
"""
from copy import deepcopy

import numpy as np


OUTPUTS = (
    'reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output',
    'reg_pedal_onset_output', 'reg_pedal_offset_output', 'pedal_frame_output',
)


def _frozen_model(model):
    if any(module.training for module in model.modules()):
        raise ValueError('The original model must remain in eval mode')
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('The original model must remain frozen')


def _linear_heads(heads):
    import torch
    if (not isinstance(heads, torch.nn.ModuleDict)
            or set(heads) != {'onset', 'frame'}
            or any(not isinstance(heads[name], torch.nn.Linear)
                   or heads[name].out_features != 88 for name in heads)):
        raise ValueError('Expected onset/frame linear heads with 88 outputs')


def heads_copy(model):
    """Return independent trainable copies; leave every original tensor alone."""
    import torch
    _frozen_model(model)
    heads = torch.nn.ModuleDict({
        'onset': deepcopy(model.note_model.reg_onset_fc),
        'frame': deepcopy(model.note_model.frame_fc),
    })
    _linear_heads(heads)
    heads.requires_grad_(True)
    return heads


def forward_logits(model, heads, h, f0, offset):
    """Run both heads over complete [batch,1001,*] cached windows.

    All cached inputs are treated as frozen evidence. Each call recomputes the
    full bidirectional frame GRU with current onset values and fresh hidden
    state. The frame loss cannot update the onset head through this path.
    """
    import torch
    _frozen_model(model)
    _linear_heads(heads)
    if (not isinstance(h, torch.Tensor) or h.ndim != 3 or h.shape[0] < 1
            or h.shape[1:] != (1001, heads['onset'].in_features)):
        raise ValueError('Expected complete [batch,1001,features] windows')
    for value, shape in ((h, h.shape), (f0, (h.shape[0], 1001, 88)),
                         (offset, (h.shape[0], 1001, 88))):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != torch.float32 or value.device != h.device
                or not torch.isfinite(value).all()):
            raise ValueError('Expected finite float32 cached tensors on one device')
    gru = model.note_model.frame_gru
    if (not isinstance(gru, torch.nn.GRU) or not gru.batch_first
            or gru.input_size != 264
            or gru.hidden_size * (2 if gru.bidirectional else 1)
            != heads['frame'].in_features):
        raise ValueError('Frame GRU and linear head dimensions do not agree')
    if any(parameter.dtype != h.dtype or parameter.device != h.device
           for module in (heads, gru) for parameter in module.parameters()):
        raise ValueError('Cached tensors and model parameters must share dtype/device')

    onset_logits = heads['onset'](h.detach())
    with torch.no_grad():
        values, _ = gru(torch.cat((f0, onset_logits.sigmoid().detach(), offset), dim=2))
    frame_logits = heads['frame'](values)
    return onset_logits, frame_logits


def replay_outputs(model, heads, cached_numpy):
    """Replay one or more CPU windows, copying all seven original output arrays."""
    import torch
    arrays = {}
    for key in ('features', 'frame_base', *OUTPUTS):
        value = cached_numpy.get(key)
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                or value.ndim != 3 or not np.isfinite(value).all()):
            raise ValueError('Missing or invalid cached array: ' + key)
        arrays[key] = value
    batch = arrays['features'].shape[0]
    for key in OUTPUTS:
        width = 1 if 'pedal' in key else 88
        if arrays[key].shape != (batch, 1001, width):
            raise ValueError('Unexpected cached output shape: ' + key)
    with torch.no_grad():
        onset, frame = forward_logits(model, heads,
            torch.from_numpy(arrays['features']),
            torch.from_numpy(arrays['frame_base']),
            torch.from_numpy(arrays['reg_offset_output']))
        onset = onset.sigmoid().cpu().numpy().copy()
        frame = frame.sigmoid().cpu().numpy().copy()
    result = {key: arrays[key].copy() for key in OUTPUTS}
    result.update(reg_onset_output=onset, frame_output=frame)
    return result
