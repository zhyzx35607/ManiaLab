"""Isolated research wrapper around a pinned, unmodified Transkun checkout.

No product imports or optional Torch dependencies are loaded at module import.
Raw window events are copied before the upstream merger mutates Note objects.
"""
import hashlib
import importlib.metadata
import json
from math import ceil, gcd
from pathlib import Path
import sys

COMMIT = 'c5cb8370e17b4a1650b5971ba87ee4b0d208e5b6'
MODEL_SHA256 = '50a80010effc2a59ffcd068a95cd2b29bd7f23a27a3515bc3ccd209c89a3d44c'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def event_record(note):
    """Copy values, never store aliases to the mutable upstream event."""
    from math import isfinite
    result = dict(onset=float(note.start), offset=float(note.end), pitch=int(note.pitch),
                  velocity=int(note.velocity), has_onset=bool(note.hasOnset), has_offset=bool(note.hasOffset))
    if (not isfinite(result['onset']) or not isfinite(result['offset'])
            or result['offset'] < result['onset']
            or result['pitch'] not in (-64, -67) and not 21 <= result['pitch'] <= 108):
        raise ValueError('Invalid Transkun event')
    return result


def verify_source(base):
    base = Path(base).resolve()
    acquisition = json.loads((base/'acquisition.json').read_text(encoding='utf-8'))
    if acquisition['commit'] != COMMIT or not acquisition['verified_all_git_blobs']:
        raise ValueError('Unexpected source revision')
    for name, row in acquisition['files'].items():
        if sha256(base/'source'/name) != row['sha256']:
            raise ValueError('Pinned upstream file changed: '+name)
    if sha256(base/'source/transkun/pretrained/2.0.pt') != MODEL_SHA256:
        raise ValueError('Unexpected checkpoint')
    return acquisition


def load_model(base):
    base = Path(base).resolve()
    acquisition = verify_source(base)
    sys.path[:0] = [str(base/'source'), str(base/'dependencies')]
    import torch
    from transkun.ModelTransformer import TransKun, Config
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config_path = base/'source/transkun/pretrained/2.0.conf'
    supplied = json.loads(config_path.read_text(encoding='utf-8'))['Model']
    if supplied['module'] != 'transkun.ModelTransformer' or supplied['configClassName'] != 'Config':
        raise ValueError('Unexpected model class')
    config = Config()
    if set(supplied['config']) != set(vars(config)):
        raise ValueError('Config must explicitly cover every architecture field')
    for key, value in supplied['config'].items():
        setattr(config, key, value)
    checkpoint = torch.load(base/'source/transkun/pretrained/2.0.pt', map_location='cpu', weights_only=True)
    state_key = 'best_state_dict' if 'best_state_dict' in checkpoint else 'state_dict'
    model = TransKun(config)
    loaded = model.load_state_dict(checkpoint[state_key], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys or any(
            not torch.equal(value, checkpoint[state_key][key]) for key, value in model.state_dict().items()):
        raise ValueError('Strict state verification failed')
    model.eval()
    model.requires_grad_(False)
    identity = dict(commit=COMMIT, model_sha256=MODEL_SHA256,
        acquisition_sha256=sha256(base/'acquisition.json'), config_sha256=sha256(config_path),
        source_hashes={name:row['sha256'] for name,row in acquisition['files'].items()},
        state_key=state_key, state_entries=len(model.state_dict()),
        parameters=sum(p.numel() for p in model.parameters()), config=vars(config), strict_load=True,
        environment={name:importlib.metadata.version(name) for name in
                     ('torch', 'torchaudio', 'numpy', 'scipy', 'pretty_midi', 'mir_eval',
                      'mido', 'six', 'importlib_resources', 'decorator')},
        python=sys.version, intra_op_threads=2, inter_op_threads=1, device='cpu')
    return model, identity


def input_pcm(samples, sample_rate, target_rate=44100):
    import numpy as np
    from scipy.signal import resample_poly
    signal = np.asarray(samples)
    if (signal.ndim != 1 or not len(signal) or signal.dtype != np.float32
            or not np.isfinite(signal).all() or type(sample_rate) is not int or sample_rate <= 0):
        raise ValueError('Expected finite float32 mono PCM with an integer sample rate')
    common = gcd(sample_rate, target_rate)
    if sample_rate != target_rate:
        signal = resample_poly(signal, target_rate//common, sample_rate//common).astype(np.float32)
    return np.ascontiguousarray(signal)


def transcribe(model, pcm, *, record_windows=True):
    """Return the official final output and copied pre-merger window records."""
    import numpy as np
    import torch
    signal = np.asarray(pcm)
    if signal.ndim != 1 or signal.dtype != np.float32 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError('Expected prepared mono PCM')
    windows = []
    original = model.transcribeFrames
    rate, hop, span = model.fs, model.segmentHopSizeInSecond, model.segmentSizeInSecond
    step = ceil(hop*rate/model.hopSize)*model.hopSize
    padding = span-hop

    def traced(frames, *args, **kwargs):
        notes, last = original(frames, *args, **kwargs)
        index = len(windows)
        windows.append(dict(index=index, start_sample=index*step-ceil(padding*rate),
            origin_seconds=index*step/rate-padding, shape=list(frames.shape),
            forced_start_pos=list(kwargs['forcedStartPos']), last_frame_index=kwargs['lastFrameIdx'],
            last_positions=[int(p) for p in last], events=[[event_record(e) for e in batch] for batch in notes]))
        return notes, last

    if record_windows:
        model.transcribeFrames = traced
    try:
        with torch.inference_mode():
            notes = model.transcribe(torch.from_numpy(signal[:, None]),
                discardSecondHalf=False, mergeIncompleteEvent=True)
    finally:
        if record_windows:
            model.transcribeFrames = original
    events = [event_record(note) for note in notes]
    duration = len(signal)/rate
    trace = dict(windows=windows, sample_rate=rate, sample_count=len(signal), duration_seconds=duration,
        segment_seconds=span, nominal_hop_seconds=hop, actual_hop_samples=step,
        padding_each_side_samples=ceil(padding*rate),
        final_events=events, piano_events=[e for e in events if e['pitch'] > 0],
        pedal_events=[e for e in events if e['pitch'] < 0],
        piano_onsets_outside_input=sum(e['pitch']>0 and not 0 <= e['onset'] < duration for e in events),
        final_offset_flags_are_postprocessed=True, release_truth_available=False)
    if record_windows and len(windows) != ceil((len(signal)+2*ceil(padding*rate))/step):
        raise ValueError('Unexpected upstream window count')
    return trace
