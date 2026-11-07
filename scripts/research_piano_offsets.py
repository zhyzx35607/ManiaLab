"""Run an isolated upstream onset/offset piano model on explicit local cases.

Research only: no application import, installation, or automatic download.
Use the separate audio Python environment and explicitly provide a checkout
of qiuqiangkong/piano_transcription_inference plus its extra dependencies.
Predictions are private intermediate files, not distributable training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--dependencies', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size < 160_000_000:
        raise ValueError('Provide the previously verified official checkpoint; automatic download is disabled')
    sys.path[:0] = [str(args.source.resolve()), str(args.dependencies.resolve())]
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    import torch
    from piano_transcription_inference import PianoTranscription

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    initialized = time.monotonic()
    model = PianoTranscription(checkpoint_path=str(args.checkpoint.resolve()), device='cpu')
    initialization_seconds = time.monotonic() - initialized
    args.work.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in json.loads(args.cases.read_text(encoding='utf-8')):
        name = case['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
            raise ValueError('Case names must be ASCII identifiers')
        rate, original = wavfile.read(case['audio'])
        if np.issubdtype(original.dtype, np.integer):
            audio = original.astype(np.float32) / max(abs(np.iinfo(original.dtype).min), np.iinfo(original.dtype).max)
        else:
            audio = original.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        start = float(case.get('start_seconds', 0))
        duration = float(case.get('duration_seconds', 40))
        audio = audio[round(start * rate):round((start + duration) * rate)]
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Empty or invalid audio')
        length = len(audio) / rate
        from math import gcd
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        print('Analyzing ' + name, flush=True)
        begin = time.monotonic()
        prediction = model.transcribe(audio, None)
        seconds = time.monotonic() - begin
        notes = []
        for event in prediction['est_note_events']:
            a = float(event['onset_time'])
            b = min(length, float(event['offset_time']))
            if not 0 <= a < b:
                continue
            notes.append(dict(start=round((start+a)*1000), end=round((start+b)*1000),
                              pitch=int(event['midi_note']), source='pitch', voice='piano-model',
                              strength=float(event['velocity']) / 128, id=f'piano-{len(notes)}'))
        (args.work / (name + '-notes.json')).write_text(json.dumps(notes, indent=2), encoding='utf-8')
        row = dict(name=name, start_seconds=start, duration_seconds=length, seconds=round(seconds, 2),
                   notes=len(notes), over_2s=sum(n['end']-n['start'] >= 2000 for n in notes),
                   source_audio_sha256=hashlib.sha256(Path(case['audio']).read_bytes()).hexdigest())
        rows.append(row)
        print(json.dumps(row), flush=True)
    with args.checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(model='Note_pedal onset/offset regression CRNN; official default thresholds',
        initialization_seconds=round(initialization_seconds, 2), checkpoint_sha256=digest,
        runtime=dict(torch=torch.__version__, numpy=np.__version__, threads=2),
        cases=rows, enabled_in_application=False)
    (args.work / 'inference-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
