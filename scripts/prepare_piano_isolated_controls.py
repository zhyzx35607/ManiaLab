"""Build source-known piano controls from fixed, independently recorded notes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256
from scripts.piano_source_controls import compose_case

PITCHES = (36, 48, 60, 72, 84, 96, 67, 79, 66)
PAIRS = ((36, 48), (48, 60), (60, 72), (72, 84), (84, 96), (48, 67), (60, 79), (48, 66))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def prepare_source(item):
    # aifc is in the project's pinned Python 3.12; no third-party audio decoder.
    import aifc
    path = Path(item['path'])
    if sha256(path) != item['sha256']:
        raise ValueError('Original AIFF changed')
    with aifc.open(str(path), 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (44100, 2, 2, b'NONE'):
            raise ValueError('Expected original 44100Hz stereo 16-bit uncompressed AIFF')
        data = stream.readframes(stream.getnframes())
    raw = np.frombuffer(data, dtype='>i2').astype(np.float64).reshape(-1, 2)/32768
    mono = raw.mean(axis=1)
    block = 220
    power = np.square(mono[:len(mono)//block*block].reshape(-1, block)).mean(axis=1)
    rms = np.sqrt(power)
    threshold = float(rms.max()*.1)
    above = rms >= threshold
    sustained = np.flatnonzero(above[:-2] & above[1:-1] & above[2:])
    if threshold <= 0 or not len(sustained):
        raise ValueError('No three-block envelope anchor')
    anchor = int(sustained[0])*block
    left, right = anchor-4410, anchor+5*44100
    if left < 0 or right > len(mono):
        raise ValueError('Original source lacks the fixed context')
    crop = mono[left:right].copy()
    crop[:441] *= np.linspace(0., 1., 441)
    crop[-2205:] *= np.linspace(1., 0., 2205)
    source = resample_poly(crop, 160, 441).astype(np.float32)
    if source.shape != (81600,) or not np.isfinite(source).all() or np.max(np.abs(source)) >= 1:
        raise ValueError('Invalid prepared component')
    return source, dict(pitch=item['pitch'], original_sha256=item['sha256'], original_url=item['url'],
        original_samples=len(mono), original_sample_rate=44100, original_channels=2,
        mono_rule='arithmetic mean of int16 channels / 32768; no normalization',
        envelope_block_samples=block, envelope_peak_rms=float(rms.max()),
        envelope_threshold=threshold, envelope_required_consecutive_blocks=3,
        original_anchor_sample=anchor, original_anchor_seconds=anchor/44100,
        original_crop_start_sample=left, original_crop_stop_sample=right,
        boundary_fades_samples=[441, 2205],
        resample_up=160, resample_down=441, sample_rate=16000, samples=len(source),
        source_anchor_sample=1600, anchor_semantics='acoustic envelope crossing, not MIDI key-on',
        pcm_sha256=hashlib.sha256(source.astype('<f4').tobytes()).hexdigest(),
        peak_amplitude=float(np.max(np.abs(source))), physical_offset_label_available=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Preserve prior data; use a new test-data folder')
    source = args.source.resolve()
    acquisition, selection = read(source/'source.json'), read(source/'selection.json')
    if acquisition['selection_sha256'] != sha256(source/'selection.json'):
        raise ValueError('Source selection changed')
    if tuple(row['pitch'] for row in acquisition['files']) != PITCHES or tuple(map(tuple, selection['pairs'])) != PAIRS:
        raise ValueError('Fixed source scope changed')
    for page in selection['pages']:
        if sha256(source/page['name']) != page['sha256']:
            raise ValueError('Source page changed')
    paths = dict(runner=Path(__file__), helper=ROOT/'scripts/piano_source_controls.py',
                 protocol=ROOT/'docs/piano-isolated-controls-protocol.md',
                 acquisition=source/'source.json', selection=source/'selection.json')
    identity = {key: sha256(path) for key, path in paths.items()}
    output.mkdir(parents=True)
    for key, path in paths.items():
        shutil.copyfile(path, output/(key+'-snapshot'+path.suffix))
    sources, source_rows, anchors = {}, [], {}
    for item in acquisition['files']:
        samples, row = prepare_source(item)
        path = output/f'source-{item["pitch"]}.wav'
        wavfile.write(path, 16000, samples)
        row.update(wav_file=path.name, wav_sha256=sha256(path))
        sources[item['pitch']], anchors[item['pitch']] = samples, 1600
        source_rows.append(row)
    plans = [dict(name='silence', gain=0, condition='silence', components=[], pair=None)]
    for gain in (1, 4):
        for pitch in PITCHES:
            plans.append(dict(name=f'g{gain}-single-{pitch}', gain=gain, condition='single', pair=None,
                components=[dict(pitch=pitch, anchor_seconds=2., gain=float(gain))]))
        for low, high in PAIRS:
            for condition, high_times, high_scale in (
                    ('simultaneous', (2.,), 1.), ('weak_high', (2.,), .25),
                    ('delayed_high', (2.25,), 1.), ('repeated_high', (2., 2.4), 1.)):
                plans.append(dict(name=f'g{gain}-{low}-{high}-{condition}', gain=gain,
                    condition=condition, pair=[low, high],
                    components=[dict(pitch=low, anchor_seconds=2., gain=float(gain))]+
                               [dict(pitch=high, anchor_seconds=t, gain=gain*high_scale) for t in high_times]))
    rows = []
    for plan in plans:
        mix, metadata = compose_case(sources, plan['components'], sample_rate=16000,
                                    duration_seconds=10., source_anchor_samples=anchors)
        path = output/(plan['name']+'.wav')
        wavfile.write(path, 16000, mix)
        rows.append(dict(**plan, file=path.name, wav_sha256=sha256(path),
                         pcm_sha256=hashlib.sha256(mix.astype('<f4').tobytes()).hexdigest(),
                         expected=[dict(pitch=c['pitch'], anchor_seconds=c['anchor_seconds']) for c in plan['components']],
                         metadata=metadata, peak_amplitude=float(np.max(np.abs(mix)))))
    if len(rows) != 83:
        raise ValueError('Fixed control count changed')
    if any(sha256(path) != identity[key] for key, path in paths.items()):
        raise ValueError('Code/protocol/acquisition changed during preparation')
    report = dict(identity=identity, sources=source_rows, cases=rows,
                  environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
                  sample_rate=16000, duration_seconds=10, anchor_tolerance_seconds=.1,
                  offsets_scored=False, model_run=False, musical_acceptance=False,
                  artifacts_sha256={p.name: sha256(p) for p in output.iterdir()})
    write(output/'manifest.json', report)
    print(json.dumps(dict(cases=len(rows), maximum_peak=max(r['peak_amplitude'] for r in rows),
                         source_anchors={r['pitch']:r['original_anchor_seconds'] for r in source_rows})), flush=True)


if __name__ == '__main__':
    main()
