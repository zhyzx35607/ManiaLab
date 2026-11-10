"""Pinned PESTO frame evidence; no note creation, snapping or LN decisions.

Uses the author's complete signal path, with strict learned-weight loading and
sample-derived frame times. Neural callers must use the single serial CPU slot.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

SOURCE_COMMIT = '62bc0c9702558f19af4593752947fb9db1eadac9'
MODEL_SHA256 = '16c32e06ddd950e3e4866dfa3c7f8a87c4988f8adf43e57977b189f031f26f3e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frame_clock(count, sample_rate, samples, step_ms=10.):
    """CQT centers follow integer sample hops, not rounded requested milliseconds."""
    if (type(count) is not int or count <= 0 or type(sample_rate) is not int or sample_rate <= 0
            or type(samples) is not int or samples <= 0 or not np.isfinite(step_ms) or step_ms <= 0):
        raise ValueError('Invalid frame clock')
    hop = int(step_ms * sample_rate / 1000 + .5)
    if hop <= 0 or count != samples // hop + 1:
        raise ValueError('Unexpected centered CQT frame count')
    centers = np.arange(count, dtype=np.int64) * hop
    return centers / sample_rate, centers < samples, hop


def note_support(notes, frames):
    """Describe agreement with one F0; disagreement cannot disprove a chord note."""
    times, valid = frames['times'], frames['valid']
    rows = []
    for index, note in enumerate(notes):
        if not all(np.isfinite(note[k]) for k in ('onset', 'offset', 'pitch')) or note['offset'] <= note['onset']:
            raise ValueError('Invalid note interval')
        active = valid & (times >= note['onset']) & (times < note['offset'])
        voiced = active & (frames['voicing'] >= .5)
        supported = voiced & (np.abs(frames['pitch_midi'] - note['pitch']) <= .5)
        count = int(active.sum())
        rows.append(dict(index=index, frames=count, voiced_frames=int(voiced.sum()),
            supported_frames=int(supported.sum()), support_fraction=float(supported.sum()/count) if count else None,
            semantics='single_f0_agreement_not_note_correctness_or_salience'))
    return rows


class PestoPredictor:
    def __init__(self, source, dependencies):
        self.source = Path(source).resolve()
        def git(*args):
            return subprocess.check_output(['git', '-C', str(self.source), *args], text=True).strip()
        if git('rev-parse', 'HEAD') != SOURCE_COMMIT or git('diff', 'HEAD', '--name-only'):
            raise ValueError('PESTO source differs from reviewed commit')
        source_files = {name: sha(self.source/name) for name in git('ls-files', '*.py', 'LICENSE*').splitlines()}
        checkpoint = self.source/'pesto/weights/mir-1k_g7.ckpt'
        if sha(checkpoint) != MODEL_SHA256:
            raise ValueError('PESTO checkpoint identity changed')
        if Path(dependencies).exists():
            sys.path.insert(0, str(Path(dependencies).resolve()))
        sys.path.insert(0, str(self.source))
        import torch
        from omegaconf import OmegaConf, DictConfig, ListConfig
        from omegaconf.nodes import AnyNode
        from omegaconf.base import Metadata, ContainerMetadata
        from pesto.data import Preprocessor
        from pesto.model import PESTO, Resnet1d
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        allowed = [AnyNode, DictConfig, ListConfig, Metadata, ContainerMetadata,
                   defaultdict, list, dict, int, Any]
        with torch.serialization.safe_globals(allowed):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        plain = lambda value: OmegaConf.to_container(value) if OmegaConf.is_config(value) else value
        hparams, hcqt = plain(state['hparams']), plain(state['hcqt_params'])
        # Generate deterministic CQT buffers only after learned-weight loading.
        # The upstream loader's broad strict=False fallback is not used.
        self.model = PESTO(Resnet1d(**hparams['encoder']), Preprocessor(hop_size=10., **hcqt),
                           crop_kwargs=hparams['pitch_shift'], reduction=hparams['reduction']).eval()
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.torch = torch
        self.identity = dict(source_commit=SOURCE_COMMIT, source_files=source_files,
            checkpoint_sha256=MODEL_SHA256, model='mir-1k_g7', strict_weights=True,
            runner_sha256=sha(__file__), step_ms=10., reduction=hparams['reduction'],
            chunks=1, normalization=False, resampling=False, intra_threads=2, inter_threads=1,
            versions={p: importlib.metadata.version(p) for p in ('torch', 'torchaudio', 'omegaconf', 'numpy', 'scipy')})

    def predict(self, pcm, sample_rate, output, *, source_identity):
        output = Path(output)
        if output.exists():
            raise ValueError('Preserve prior pitch evidence; use a new directory')
        if (pcm.ndim != 1 or pcm.dtype != np.float32 or not np.isfinite(pcm).all()
                or not 1 <= len(pcm)/sample_rate <= 60):
            raise ValueError('Expected 1–60 seconds of finite mono float32 PCM')
        started = time.monotonic()
        with self.torch.inference_mode():
            raw = self.model(self.torch.from_numpy(pcm.copy()), sr=sample_rate,
                             convert_to_freq=False, return_activations=True)
        pitch, voicing, energy, activations = [value.cpu().numpy() for value in raw]
        if (pitch.ndim != 1 or voicing.shape != pitch.shape or energy.shape != pitch.shape
                or activations.ndim != 2 or activations.shape[0] != len(pitch)
                or any(not np.isfinite(value).all() for value in (pitch, voicing, energy, activations))):
            raise ValueError('Nonfinite or malformed PESTO output')
        times, valid, hop = frame_clock(len(pitch), sample_rate, len(pcm))
        frames = dict(times=times, valid=valid, pitch_midi=pitch, voicing=voicing,
                      energy=energy, activations=activations)
        output.mkdir(parents=True)
        np.savez_compressed(output/'frames.npz', **frames)
        report = dict(identity=self.identity, input=source_identity, sample_rate=sample_rate,
            samples=len(pcm), pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
            actual_hop_samples=hop, actual_hop_ms=hop*1000/sample_rate, frames=len(pitch),
            padding_centers=int((~valid).sum()), seconds=time.monotonic()-started,
            frames_sha256=sha(output/'frames.npz'), confidence_semantics='voicing_not_pitch_accuracy',
            single_f0=True, note_boundaries=False, app_default_enabled=False)
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        return frames, report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--audio', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--source', type=Path, default=Path('test-data/pesto-source'))
    cli.add_argument('--dependencies', type=Path, default=Path('test-data/rosvot-runtime'))
    args = cli.parse_args()
    predictor = PestoPredictor(args.source, args.dependencies)
    import soundfile as sf
    pcm, rate = sf.read(args.audio, dtype='float32', always_2d=True)
    frames, report = predictor.predict(pcm.mean(axis=1), rate, args.output,
        source_identity=dict(audio=str(args.audio.resolve()), sha256=sha(args.audio),
                             original_channels=pcm.shape[1], mono_policy='channel_mean'))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
