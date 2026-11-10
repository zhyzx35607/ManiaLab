"""Frozen, identity-checked Mapper audio encoder for research observations.

No chart, timing or musical labels enter this module. The final encoder is
contextual: a vector has a temporal position, not an isolated-note guarantee.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


WEIGHTS_SHA = '3d5d1c2a01ad462bcc99bafaaf44174bc47be099724574b17cd68016be5a863f'
SOURCE_SHA = '43865d4af98f9e0037c87d34e5171606fa62d667a44c568ec825f18368b18014'
SAMPLE_RATE = 16000
HOP = 128
WINDOW_SAMPLES = 2047 * HOP
FRAME_SAMPLES = 2 * HOP
REVISION = 'mapper-frozen-audio-positions-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_encoder(source, checkpoint, threads=2):
    source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
    source_sha = hashlib.sha256(''.join(sha(p) for p in sorted((source/'osuT5').rglob('*.py'))).encode()).hexdigest()
    if source_sha != SOURCE_SHA or sha(checkpoint/'model.safetensors') != WEIGHTS_SHA:
        raise ValueError('Frozen Mapper source/weights changed')
    os.environ.update(HF_HUB_OFFLINE='1', WANDB_DISABLED='true', TOKENIZERS_PARALLELISM='false')
    sys.path[:0] = [str(source/'osuT5'), str(source)]
    import torch
    from osuT5.utils.model_utils import load_model
    from osuT5.dataset.data_utils import load_audio_file
    torch.set_num_threads(threads)
    model, _ = load_model(checkpoint, None, device='cpu', precision='fp32',
                         attn_implementation='sdpa', eval_mode=True, gamemode=3)
    cfg = model.config
    if (cfg.sample_rate, cfg.hop_length, cfg.n_fft, cfg.n_mels) != (SAMPLE_RATE, HOP, 1024, 128):
        raise ValueError('Unexpected encoder spectral geometry')
    if any(getattr(cfg, 'do_'+name+'_embed') for name in ('style', 'difficulty', 'mapper', 'song_position')):
        raise ValueError('Encoder would consume non-audio conditioning')
    encoder = model.get_encoder().eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    identity = dict(revision=REVISION, weights_sha256=WEIGHTS_SHA, source_sha256=source_sha,
                    config_sha256=sha(checkpoint/'config.json'), sample_rate=SAMPLE_RATE,
                    window_samples=WINDOW_SAMPLES, frame_samples=FRAME_SAMPLES,
                    loader_sha256=sha(__file__), normalisation='upstream_mono_whole_file_peak')
    # The wrapper owns references to its encoder/spectrogram. Release decoder
    # memory before extracting an entire collection on a CPU machine.
    del model
    return encoder, load_audio_file, identity


def audit_geometry(encoder):
    """Measure STFT/conv timing with impulses through the actual modules."""
    import torch
    import torch.nn.functional as F
    convs = [encoder.base.conv1, encoder.base.conv2]
    geometry = [dict(kernel=list(c.kernel_size), stride=list(c.stride), padding=list(c.padding)) for c in convs]
    if geometry != [dict(kernel=[3], stride=[1], padding=[1]), dict(kernel=[3], stride=[2], padding=[1])]:
        raise ValueError('Unreviewed encoder convolution geometry')
    with torch.inference_mode():
        silence = torch.zeros((1, WINDOW_SAMPLES))
        mel_zero = encoder.spectrogram(silence)
        stage = lambda mel: convs[1](F.gelu(convs[0](mel.transpose(1, 2))))
        baseline = stage(mel_zero)
        measurements = []
        for at_ms in (2000., 8000., 14000.):
            pulse = silence.clone(); pulse[0, round(at_ms*SAMPLE_RATE/1000)] = 1
            mel = encoder.spectrogram(pulse)
            difference = (stage(mel)-baseline).abs().amax(dim=1)[0]
            support = torch.nonzero(difference > difference.max()*1e-5).flatten().numpy()
            first, last = support[[0, -1]]*FRAME_SAMPLES/SAMPLE_RATE*1000
            error = abs((float(first)+float(last))/2-at_ms)
            if error > 16 or first > at_ms or last < at_ms:
                raise ValueError('Convolution impulse clock differs from declared time positions')
            measurements.append(dict(impulse_ms=at_ms, support_ms=[float(first),float(last)], midpoint_error_ms=error))
        began = time.monotonic()
        hidden = encoder(frames=pulse, return_dict=True).last_hidden_state
        if tuple(hidden.shape) != (1, 1024, 768) or not torch.isfinite(hidden).all():
            raise ValueError('Invalid encoder output')
    return dict(convolutions=geometry, mel_shape=list(mel_zero.shape), hidden_shape=list(hidden.shape),
                impulses=measurements, encoder_seconds=time.monotonic()-began,
                notice='Measured positional lattice; attention mixes the full audio window, so this is not note localization accuracy.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('test-data/mapperatorinator/source'))
    parser.add_argument('--checkpoint', type=Path, default=Path('test-data/mapperatorinator/model/gamemode=3'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous geometry audits')
    encoder, _, identity = load_encoder(args.source, args.checkpoint)
    result = dict(identity=identity, geometry=audit_geometry(encoder))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
