"""Freeze original-microphone guitar and dual-annotator solo-vocal comparisons.

This only prepares data. No model, chart, timing correction or training is used.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


GUITAR_ERRORS = {'04_BN3-154-E_comp', '04_Jazz1-200-B_comp', '02_Funk2-119-G_comp'}
STYLES = ('BN', 'Funk', 'Jazz', 'Rock', 'SS')


def select_guitar_names(names):
    selected = []
    for player in range(6):
        for style in STYLES:
            for mode in ('comp', 'solo'):
                choices = sorted(n for n in names if re.fullmatch(
                    rf'{player:02}_{style}\d+-[^/]+_{mode}\.jams', n))
                if not choices:
                    raise ValueError(f'Missing player/style/mode: {player}/{style}/{mode}')
                # First filename by a fixed rule, before reading any labels or
                # predictions. A known-bad selected item is reported, not replaced.
                selected.append(choices[0])
    return selected


def guitar_labels(data):
    annotations = [a for a in data['annotations'] if a['namespace'] == 'note_midi']
    strings = [a['annotation_metadata']['data_source'] for a in annotations]
    if sorted(strings) != list('012345'):
        raise ValueError('Expected one note annotation for each guitar string')
    rows = []
    for annotation, string in zip(annotations, strings):
        for note in annotation['data']:
            onset, duration, pitch = (float(note[k]) for k in ('time', 'duration', 'value'))
            rows.append(dict(onset=onset, offset=onset+duration, pitch=round(pitch),
                             midi_continuous=pitch, program=-1, part=int(string), instrument='guitar'))
    return validate_labels(rows)


def vocal_labels(text):
    rows = []
    for line in csv.reader(io.StringIO(text)):
        if not line:
            continue
        onset, frequency, duration = map(float, line)
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError('Invalid singing frequency')
        pitch = 69+12*math.log2(frequency/440)
        rows.append(dict(onset=onset, offset=onset+duration, pitch=round(pitch),
                         midi_continuous=pitch, frequency_hz=frequency, program=-1, instrument='vocals'))
    return validate_labels(rows)


def validate_labels(rows):
    if not rows or any(not all(math.isfinite(r[k]) for k in ('onset', 'offset', 'midi_continuous'))
                       or r['onset'] < 0 or r['offset'] <= r['onset'] for r in rows):
        raise ValueError('Invalid note labels; inspect source rather than silently dropping rows')
    return sorted(rows, key=lambda r: (r['onset'], r['pitch'], r['offset']))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--sources', type=Path, default=Path('test-data/instrument-source-expansion-v1'))
    cli.add_argument('--output', type=Path, default=Path('test-data/guitar-vocal-benchmark-v1'))
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier preparation; choose a new output')
    sources = json.loads((args.sources / 'sources.json').read_text(encoding='utf-8'))
    for source in sources:
        path = Path(source['path'])
        with path.open('rb') as stream:
            md5 = hashlib.file_digest(stream, 'md5').hexdigest()
        if (path.stat().st_size != source['file']['size'] or 'md5:'+md5 != source['file']['checksum']
                or sha(path) != source['sha256']):
            raise ValueError('Archive identity mismatch: ' + str(path))
    args.output.mkdir(parents=True)
    (args.output / 'audio').mkdir()
    (args.output / 'labels').mkdir()
    import numpy as np
    from scipy.io import wavfile
    cases, excluded = [], []

    def save_case(name, dataset, audio_bytes, raw_name, references, metadata):
        audio = args.output / 'audio' / (name + '.wav')
        audio.write_bytes(audio_bytes)
        rate, samples = wavfile.read(audio)
        if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError('Expected mono original PCM: ' + name)
        duration = len(samples) / rate
        seconds = min(20., duration-2.)
        if seconds <= 0:
            raise ValueError('Source too short for a fixed two-second scoring margin')
        label_refs = {}
        for annotator, (rows, origin, raw_bytes) in references.items():
            dest = args.output / 'labels' / (name+'-'+annotator+'.json')
            dest.write_text(json.dumps(rows, indent=2), encoding='utf-8')
            label_refs[annotator] = dict(path=str(dest.resolve()), sha256=sha(dest),
                original_member=origin, original_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                count=len(rows), scored_count=sum(r['onset'] < seconds for r in rows),
                offsets_beyond_audio=sum(r['offset'] >= duration for r in rows),
                scored_offsets_beyond_audio=sum(r['onset'] < seconds and r['offset'] >= duration for r in rows))
        cases.append(dict(name=name, dataset=dataset, original_member=raw_name,
            audio=str(audio.resolve()), audio_sha256=sha(audio), sample_rate=rate,
            samples=len(samples), decoded_dtype=str(samples.dtype), duration_seconds=duration,
            start_ms=0, seconds=seconds, inference_scope='full_original_audio',
            references=label_refs, metadata=metadata))

    with zipfile.ZipFile(args.sources / '3371780/annotation.zip') as labels, \
            zipfile.ZipFile(args.sources / '3371780/audio_mono-mic.zip') as audio:
        selected = select_guitar_names(labels.namelist())
        for name in selected:
            stem = Path(name).stem
            if stem in GUITAR_ERRORS:
                excluded.append(dict(name=stem, reason='upstream_known_annotation_error', replacement=None))
                continue
            raw = labels.read(name)
            data = json.loads(raw)
            member = stem + '_mic.wav'
            save_case('guitar-'+stem, 'GuitarSet', audio.read(member), member,
                {'strings': (guitar_labels(data), name, raw)},
                dict(player=stem[:2], style=stem.split('_')[1], mode=stem.split('_')[-1]))
    with zipfile.ZipFile(args.sources / '5578807/vocadito.zip') as archive:
        members = sorted((n for n in archive.namelist() if re.fullmatch(r'Audio/vocadito_\d+\.wav', n)),
                         key=lambda n: int(re.search(r'_(\d+)\.wav', n)[1]))
        if len(members) != 40:
            raise ValueError('Expected all 40 vocal recordings')
        for member in members:
            name = Path(member).stem
            references = {}
            for annotator in ('A1', 'A2'):
                origin = f'Annotations/Notes/{name}_notes{annotator}.csv'
                raw = archive.read(origin)
                references[annotator] = (vocal_labels(raw.decode('utf-8-sig')), origin, raw)
            save_case(name, 'vocadito', archive.read(member), member, references,
                      dict(annotators=['A1', 'A2'], combine_annotations=False))
    manifest = dict(revision=1, runner_sha256=sha(__file__), sources=sources,
        known_guitar_errors=sorted(GUITAR_ERRORS), excluded=excluded, cases=cases,
        protocol=[
            'Guitar: all six performers, five styles and accompaniment/solo modes; lexically first filename within each group, no result-based replacements.',
            'Vocal: every one of the 40 recordings; score both annotators separately, never pick the more favorable annotation.',
            'Use original mono microphone or vocal PCM; no normalization, onset shift, grid snapping or invented instrument hints.',
            'Predict full original audio; score attacks in [0,min(20,duration-2)) seconds. Keep crossing predictions in interval denominators.',
            'Retain continuous annotated pitch as well as rounded semitone; exact-semitone comparison has tuning/annotation uncertainty.',
            'These annotations do not certify a General MIDI program; retain program=-1 and score pitch/onset/interval, not fabricated exact-program labels.',
            'No claim these publicly used datasets are unknown to pre-trained models. Physical endpoints are not authored LN-release preferences.',
        ])
    target = args.output / 'manifest.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(cases=len(cases), excluded=excluded, manifest_sha256=sha(target),
        scored_tail_boundary_cases=[c['name'] for c in cases
            if any(r['scored_offsets_beyond_audio'] for r in c['references'].values())]), ensure_ascii=False))


if __name__ == '__main__':
    main()
