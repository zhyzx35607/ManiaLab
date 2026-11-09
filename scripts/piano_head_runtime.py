"""Frozen Kong model and exact cached final-onset-head replay for research."""
from copy import deepcopy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_frozen_model():
    from scripts.probe_piano_acoustics import verify_specialist, CHECKPOINT_SHA256, SOURCE_COMMIT
    source = ROOT/'test-data/piano-release-probe'
    proof = verify_specialist(source)
    sys.path[:0] = [str(source/'source'), str(source/'dependencies')]
    os.environ.update(TORCH_FORCE_WEIGHTS_ONLY_LOAD='1', MPLBACKEND='Agg')
    import torch
    from piano_transcription_inference.models import Note_pedal
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = Note_pedal(frames_per_second=100, classes_num=88)
    checkpoint = torch.load(source/'model.pth', map_location='cpu', weights_only=True)['model']
    for name in ('note_model', 'pedal_model'):
        branch = getattr(model, name)
        branch.load_state_dict(checkpoint[name], strict=True)
        if any(not torch.equal(value, checkpoint[name][key]) for key,value in branch.state_dict().items()):
            raise ValueError('Checkpoint differs after strict load')
    model.eval().requires_grad_(False)
    return model, dict(model_sha256=CHECKPOINT_SHA256, source_commit=SOURCE_COMMIT,
        source=proof, torch=torch.__version__, device='cpu', intra_threads=2, inter_threads=1)


def head_copy(model):
    head = deepcopy(model.note_model.reg_onset_fc)
    head.requires_grad_(True)
    if sum(p.numel() for p in head.parameters()) != 45144:
        raise ValueError('Unexpected trainable parameter count')
    return head


def replay_outputs(model, head, cached):
    import torch
    with torch.no_grad():
        h = torch.from_numpy(cached['features'])
        f0 = torch.from_numpy(cached['frame_base'])
        offset = torch.from_numpy(cached['reg_offset_output'])
        onset = torch.sigmoid(head(h))
        values, _ = model.note_model.frame_gru(torch.cat((f0, onset, offset), dim=2))
        frame = torch.sigmoid(model.note_model.frame_fc(values))
    result = {key:cached[key].copy() for key in cached if key.endswith('_output')}
    result.update(reg_onset_output=onset.numpy(), frame_output=frame.numpy())
    return result


def extract_window(model, pcm):
    import numpy as np
    import torch
    if pcm.dtype != np.float32 or pcm.shape != (160000,) or not np.isfinite(pcm).all():
        raise ValueError('Expected a physical ten-second input')
    cached = {}

    def capture_h(module, inputs):
        cached['features'] = inputs[0].detach().cpu().numpy().copy()

    def capture_f0(module, inputs, output):
        cached['frame_base'] = output.detach().cpu().numpy().copy()

    handles = [model.note_model.reg_onset_fc.register_forward_pre_hook(capture_h),
               model.note_model.frame_model.register_forward_hook(capture_f0)]
    try:
        with torch.inference_mode():
            outputs = model(torch.from_numpy(pcm[None]))
            cached.update({key:value.cpu().numpy().copy() for key,value in outputs.items()})
    finally:
        for handle in handles:
            handle.remove()
    if cached['features'].shape != (1,1001,512) or cached['frame_base'].shape != (1,1001,88):
        raise ValueError('Unexpected cache shape')
    if not all(np.isfinite(value).all() for value in cached.values()):
        raise ValueError('Nonfinite acoustic features')
    repeated = replay_outputs(model, model.note_model.reg_onset_fc, cached)
    errors = {key:float(np.max(abs(repeated[key]-cached[key]))) for key in repeated}
    if len(errors)!=7 or any(errors.values()):
        raise ValueError('Epoch zero cache replay differs: '+str(errors))
    return cached, errors


def decode(outputs):
    from piano_transcription_inference.utilities import RegressionPostProcessor
    processor = RegressionPostProcessor(100, classes_num=88, onset_threshold=.3,
        offset_threshold=.3, frame_threshold=.1, pedal_offset_threshold=.2)
    notes, pedals = processor.output_dict_to_midi_events({key:value.copy() for key,value in outputs.items()})
    return dict(notes=[dict(onset=float(n['onset_time']), offset=float(n['offset_time']),
                            pitch=int(n['midi_note']), velocity=float(n['velocity'])) for n in notes],
                pedals=[{key:float(value) for key,value in p.items()} for p in pedals])
