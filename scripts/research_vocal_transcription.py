"""Run the author's ROSVOT/RWBD/RMVPE checkpoints on one isolated vocal recording.

Research only: physical audio time, no lyrics/chart labels, no quantization, no
default application change. Run serially after other neural inference finishes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

SOURCE_COMMIT = '3c8332bf43adae35f6e4d64971862f2f6139b310'
RUNTIME_PACKAGES = ('torch','torchaudio','numpy','scipy','librosa','soxr','PyYAML',
                    'pyworld','numba','llvmlite','pretty-midi','setuptools')
FILES = {
    'checkpoints/rmvpe/model.pt': '19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2',
    'checkpoints/rosvot/model.pt': '7501fb5f913d971c2f51bcb3063b930027b03206581820a4d2bfdc394c9c3fcb',
    'checkpoints/rosvot/config.yaml': '2ad2cb756623418c471b7dc2f56175cce88b69a70b4a2c354fa1a78525aa54e2',
    'checkpoints/rwbd/model.pt': '0bc2d42a6d4b7a05436deb937e2deda1c12de49e5687cfda0bdf6a430120dcd2',
    'checkpoints/rwbd/config.yaml': '3bb41f1d9eaa85aa1b3e5b6d94fff4ab4affb39719028ab69b4505974b9a1bc7',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def runtime_versions():
    return dict(python=sys.version, packages={name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES})


def decode_intervals(boundaries, pitches, *, sample_rate, hop_size, duration):
    """Keep adjacent same-pitch attacks; the final interval has no observed tail."""
    if (not boundaries or any(x not in (0, 1) for x in boundaries) or boundaries[0] or boundaries[-1]
            or len(pitches) != sum(boundaries)+1 or not 0 < duration <= 60
            or sample_rate <= 0 or hop_size <= 0):
        raise ValueError('Invalid ROSVOT output clock or boundary sequence')
    if any(type(pitch) is not int or not 0 <= pitch <= 127 for pitch in pitches):
        raise ValueError('Invalid decoded pitch')
    ends = [i for i, boundary in enumerate(boundaries) if boundary] + [len(boundaries)-1]
    rows, begin = [], 0
    for index, (end, pitch) in enumerate(zip(ends, pitches)):
        onset, offset = begin*hop_size/sample_rate, end*hop_size/sample_rate
        if not 0 <= onset < offset < duration + 1e-10:
            raise ValueError('Invalid interval: never silently trim model output')
        rows.append(dict(onset=onset, offset=offset, pitch=pitch, program=-1, is_drum=False,
            rest=pitch == 0, strength=1., strength_semantics='unit_weight_not_confidence',
            rest_semantics='model_zero_or_suppressed_class_not_verified_silence',
            release_reason='input_boundary' if index == len(ends)-1 else 'predicted_joint_note_word_boundary'))
        begin = end
    return rows


def verify_source(source):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != SOURCE_COMMIT or git('diff', 'HEAD', '--name-only'):
        raise ValueError('ROSVOT source differs from the reviewed commit')
    for name, expected in FILES.items():
        if sha(source/name) != expected:
            raise ValueError('ROSVOT checkpoint/config changed: '+name)
    tracked = git('ls-files', '*.py', '*.yaml').splitlines()
    return {name: sha(source/name) for name in tracked}


def load_state(torch, path, key):
    # These official checkpoints contain tensor state dicts, not executable model objects.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint[key]
    if key == 'state_dict':
        if not isinstance(state, dict) or set(state) != {'model'}:
            raise ValueError('Unexpected ROSVOT state-dict namespace')
        state = state['model']
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected only tensor weights')
    return state


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--audio', type=Path, required=True)
    cli.add_argument('--source', type=Path, default=Path('test-data/other-instrument-model-survey-2026-11-10/ROSVOT'))
    cli.add_argument('--dependencies', type=Path, default=Path('test-data/rosvot-runtime'))
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    source, output, audio_path = args.source.resolve(), args.output.resolve(), args.audio.resolve()
    if output.exists():
        raise ValueError('Use a new output directory; preserve previous predictions')
    source_files = verify_source(source)
    if args.dependencies.exists():
        sys.path.insert(0, str(args.dependencies.resolve()))
    sys.path.insert(0, str(source))
    import numpy as np
    import librosa
    import torch
    import torch.nn.functional as F
    import yaml
    from scipy.io import wavfile
    from utils.audio.mel import MelNet
    from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse, boundary2Interval
    from modules.rosvot.rosvot import MidiExtractor, WordbdExtractor
    from modules.pe.rmvpe import RMVPE
    from modules.pe.rmvpe.model import E2E0
    from modules.pe.rmvpe.spec import MelSpectrogram
    from modules.pe.rmvpe.constants import N_MELS, SAMPLE_RATE, WINDOW_LENGTH, MEL_FMIN, MEL_FMAX

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    rate, original = wavfile.read(audio_path)
    if original.ndim != 1 or original.dtype not in (np.int16, np.float32) or not len(original):
        raise ValueError('Expected original mono int16 or float32 WAV')
    signal = original.astype(np.float32)/32768 if original.dtype == np.int16 else original.copy()
    duration = len(signal)/rate
    if not np.isfinite(signal).all() or not .1 <= duration <= 60:
        raise ValueError('Use a complete finite recording of 0.1–60 seconds; no hidden cropping')
    # MelNet itself clips to [-1,1]. Reject out-of-range input instead of
    # silently changing the original floating stem before a comparison.
    if np.max(np.abs(signal)) > 1:
        raise ValueError('Upstream MelNet clips peaks above one; this source needs a separate gain protocol')
    config = yaml.safe_load((source/'checkpoints/rosvot/config.yaml').read_text(encoding='utf-8'))
    word_config = yaml.safe_load((source/'checkpoints/rwbd/config.yaml').read_text(encoding='utf-8'))
    # Match the author's inference CLI overrides, not training YAML thresholds.
    config.update(note_bd_threshold=.85, max_frames=30000)
    word_config.update(word_bd_threshold=.5, max_frames=30000)
    sr, hop = config['audio_sample_rate'], config['hop_size']
    if (sr, hop, config['frames_multiple']) != (24000, 128, 16):
        raise ValueError('Unexpected frozen acoustic configuration')
    if any(word_config[k] != config[k] for k in ('audio_sample_rate','hop_size','fft_size','win_size','audio_num_mel_bins')):
        raise ValueError('Note and word models use different audio clocks')
    resampled = librosa.resample(signal, orig_sr=rate, target_sr=sr, res_type='soxr_hq', fix=True, scale=False)
    real_frames = math.ceil(len(resampled)/hop)
    padded_frames = math.ceil(real_frames/config['frames_multiple'])*config['frames_multiple']
    if real_frames > config['max_frames']:
        raise ValueError('Input would be cropped by upstream max_frames')
    versions = runtime_versions()
    identity = dict(source_commit=SOURCE_COMMIT, source_files=source_files, model_files=FILES,
        runner_sha256=sha(__file__), audio=str(audio_path), audio_sha256=sha(audio_path),
        pcm_sha256=hashlib.sha256(signal.astype('<f4').tobytes()).hexdigest(),
        original_sample_rate=rate, original_samples=len(original), original_dtype=str(original.dtype),
        resampled_pcm_sha256=hashlib.sha256(resampled.astype('<f4').tobytes()).hexdigest(),
        resampled_samples=len(resampled), resampled_sample_rate=sr,
        resampler='librosa/soxr_hq/fix=True/scale=False', normalization=False,
        intra_threads=2, inter_threads=1, note_threshold=.85, word_threshold=.5,
        rmvpe_mode='get_pitch_batch/min_gap=6', strict_weights=True, lyrics_supplied=False,
        inference_scope='full_original_audio', versions=versions)
    output.mkdir(parents=True)
    write = lambda name, value: (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    write('identity.json', identity)
    started = time.monotonic()
    mel_net = MelNet(config).eval()
    model = MidiExtractor(config).eval()
    model.load_state_dict(load_state(torch, source/'checkpoints/rosvot/model.pt', 'state_dict'), strict=True)
    word_model = WordbdExtractor(word_config).eval()
    word_model.load_state_dict(load_state(torch, source/'checkpoints/rwbd/model.pt', 'state_dict'), strict=True)
    # Build exactly RMVPE.__init__'s modules but require strict safe tensor loads.
    # Reuse its original preprocessing/forward/decoding methods unchanged.
    pe = RMVPE.__new__(RMVPE)
    pe.device = torch.device('cpu')
    pe.resample_kernel = {}
    pe.hop_length = 160
    pe.model = E2E0(4, 1, (2, 2)).eval()
    pe.model.load_state_dict(load_state(torch, source/'checkpoints/rmvpe/model.pt', 'model'), strict=True)
    pe.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, 160, None, MEL_FMIN, MEL_FMAX).eval()
    print(json.dumps(dict(stage='loaded', duration_seconds=duration)), flush=True)
    with torch.inference_mode():
        wave = torch.from_numpy(resampled).unsqueeze(0)
        f0s, _ = pe.get_pitch_batch(wave, sample_rate=sr, hop_size=hop, lengths=[real_frames],
                                    fmin=config['f0_min'], fmax=config['f0_max'])
        normalized, unvoiced = norm_interp_f0(f0s[0][:padded_frames])
        f0 = F.pad(torch.tensor(normalized, dtype=torch.float32), (0, padded_frames-len(normalized))).unsqueeze(0)
        uv = F.pad(torch.tensor(unvoiced, dtype=torch.float32), (0, padded_frames-len(unvoiced))).unsqueeze(0)
        pitch = f0_to_coarse(denorm_f0(f0, uv))
        uv = uv.long()
        mel = mel_net(wave)
        mel = F.pad(mel, (0, 0, 0, padded_frames-mel.shape[1]))
        mel[:, real_frames:] = 0
        non_padding = mel.abs().sum(-1) > 0
        word_output = word_model(mel=mel[:, :, :word_config['use_mel_bins']], pitch=pitch,
                                 uv=uv, non_padding=non_padding, train=False)
        word_boundaries = word_output['word_bd_pred']
        result = model(mel=mel[:, :, :config['use_mel_bins']], word_bd=word_boundaries, pitch=pitch,
                       uv=uv, non_padding=non_padding, train=False)
    length = int(result['note_lengths'][0])
    boundary = result['note_bd_pred'][0, :real_frames].numpy().astype(np.int64)
    pitches = result['note_pred'][0, :length].numpy().astype(np.int64)
    events = decode_intervals(boundary.tolist(), pitches.tolist(), sample_rate=sr, hop_size=hop, duration=duration)
    raw_classes = result['note_logits'][0, :length].argmax(-1).tolist()
    for event, raw_class in zip(events, raw_classes):
        event['raw_pitch_class'] = raw_class
        event['model_output_reason'] = ('pitched_interval' if not event['rest'] else
            'zero_class' if raw_class == 0 else 'suppressed_out_of_supported_range')
    official = boundary2Interval(boundary)*hop/sr
    if not np.allclose(official, [[n['onset'], n['offset']] for n in events], atol=0, rtol=0):
        raise ValueError('Adapter timestamps differ from the original decoder')
    np.savez_compressed(output/'activations.npz', f0_hz=f0s[0], note_boundaries=boundary,
        note_pitches=pitches, note_boundary_logits=result['note_bd_logits'][0,:real_frames].numpy(),
        note_pitch_logits=result['note_logits'][0,:length].numpy(),
        word_boundaries=word_boundaries[0,:real_frames].numpy(),
        word_boundary_logits=word_output['word_bd_logits'][0,:real_frames].numpy())
    write('raw-intervals.json', events)
    notes = [n for n in events if not n['rest']]
    write('notes.json', notes)
    report = dict(identity=identity, elapsed_seconds=time.monotonic()-started,
        context_start_seconds=0., context_end_seconds=duration, real_frames=real_frames,
        padded_frames=padded_frames, notes=len(notes), rests=sum(n['rest'] for n in events),
        output_reasons=dict(Counter(n['model_output_reason'] for n in events)),
        release_reasons=dict(Counter(n['release_reason'] for n in notes)),
        notes_sha256=sha(output/'notes.json'), activations_sha256=sha(output/'activations.npz'),
        raw_intervals_sha256=sha(output/'raw-intervals.json'), enabled_in_application=False,
        warnings=['Single vocal note per interval: not a polyphonic choir transcription.',
                  'Zero pitch includes out-of-range class suppression, not just a predicted rest.',
                  'Note/word boundaries are predictions, not verified acoustic releases or LN choices.',
                  'Final interval closes at the input boundary and cannot automatically support an LN.'])
    write('report.json', report)
    print(json.dumps(dict(stage='complete', notes=len(notes), elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
