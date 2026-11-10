"""Prepare every IDMT-SMT-BASS-SINGLE-TRACK recording on its original clock."""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_guitar_vocal_benchmark import read_json, require
from scripts.probe_mapper_adapter import sha

ARCHIVE_SHA256 = 'fdc36effc0377aa065d6cb378bebd54a817e51f4d65c3531c54a1e8848400fcb'
PLUCKING = {'FS', 'PK', 'MU', 'SP', 'ST'}
EXPRESSION = {'NO', 'HA', 'DN', 'BE', 'VI', 'SL'}


def parse_labels(payload, expected_audio):
    root = ET.fromstring(payload)
    require(root.findtext('globalParameter/audioFileName') == expected_audio, 'Annotation/audio mismatch')
    require(root.findtext('globalParameter/instrument') == 'EBSS', 'Unexpected instrument')
    rows = []
    for index, event in enumerate(root.findall('transcription/event')):
        onset, offset, pitch = [float(event.findtext(key)) for key in ('onsetSec', 'offsetSec', 'pitch')]
        require(all(math.isfinite(x) for x in (onset, offset, pitch)) and
                0 <= onset < offset and pitch.is_integer() and 0 <= pitch <= 127, 'Invalid annotated note')
        plucking, expression = event.findtext('excitationStyle'), event.findtext('expressionStyle')
        require(plucking in PLUCKING and expression in EXPRESSION, 'Unknown bass articulation')
        rows.append(dict(onset=onset, offset=offset, pitch=int(pitch), program=-1,
            source_event_index=index, plucking_style=plucking, expression_style=expression,
            fret=int(event.findtext('fretNumber')), string=int(event.findtext('stringNumber'))))
    require(bool(rows), 'Empty annotation')
    # Keep all nominal annotated pitches, including dead notes and harmonics.
    # They are not substituted with estimates from the recording or model.
    return sorted(rows, key=lambda n: (n['onset'], n['pitch'], n['offset'], n['source_event_index']))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=Path('test-data/idmt-smt-bass-lines'))
    cli.add_argument('--output', type=Path, default=Path('test-data/bass-benchmark-v1'))
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve previous benchmark')
    archive = args.source/'IDMT-SMT-BASS-SINGLE-TRACKS.zip'
    record = read_json(args.source/'zenodo-record.json')
    file = next(f for f in record['files'] if f['key'] == archive.name)
    require(sha(archive) == ARCHIVE_SHA256 and archive.stat().st_size == file['size'] and
            'md5:'+hashlib.md5(archive.read_bytes()).hexdigest() == file['checksum'], 'Archive identity differs')
    from scipy.io import wavfile
    import numpy as np
    args.output.mkdir(parents=True)
    (args.output/'audio').mkdir()
    (args.output/'labels').mkdir()
    cases = []
    with zipfile.ZipFile(archive) as zipped:
        names = [f'{i:03}' for i in range(1, 18)]
        require({n for n in zipped.namelist() if n.startswith('audio/') and n.endswith('.wav')}
                == {f'audio/{n}.wav' for n in names}, 'Require all and only 17 source recordings')
        for name in names:
            member = f'audio/{name}.wav'
            path = args.output/member
            path.write_bytes(zipped.read(member))
            rate, wave = wavfile.read(path)
            require(rate == 44100 and wave.ndim == 1 and wave.dtype == np.int16 and len(wave)>2*rate,
                    'Unexpected original bass PCM')
            duration = len(wave)/rate
            require(duration <= 60, 'Recording exceeds current bounded adapter')
            raw = zipped.read(f'annotation/{name}.xml')
            labels = parse_labels(raw, name+'.wav')
            label_path = args.output/'labels'/(name+'.json')
            label_path.write_text(json.dumps(labels, indent=2), encoding='utf-8')
            seconds = min(20., duration-2.)
            cases.append(dict(name=name, audio=str(path.resolve()), audio_sha256=sha(path),
                sample_rate=rate, samples=len(wave), duration_seconds=duration, seconds=seconds,
                labels=str(label_path.resolve()), labels_sha256=sha(label_path),
                original_annotation_sha256=hashlib.sha256(raw).hexdigest(), reference_count=len(labels),
                scoring_reference_count=sum(n['onset']<seconds for n in labels),
                scoring_unobserved_tails=sum(n['onset']<seconds and n['offset']>=duration for n in labels),
                plucking_styles=dict(Counter(n['plucking_style'] for n in labels)),
                expression_styles=dict(Counter(n['expression_style'] for n in labels))))
    manifest = dict(revision=1, runner_sha256=sha(__file__), archive_sha256=ARCHIVE_SHA256,
        source_record='https://zenodo.org/records/7544099', license='CC-BY-NC-ND-4.0',
        source_record_sha256=sha(args.source/'zenodo-record.json'), cases=cases,
        protocol=['Every one of the 17 recordings; original PCM, full recording inference, no normalization.',
            'Score onsets in [0,min(20,duration-2)); retain predictions crossing scoring/context ends.',
            'Unknown reference tails invalidate the entire affected interval metric, not onset scoring.',
            'Compare application bass postprocessing with fixed Tsumugi bass_v2, all classes retained.',
            'Keep nominal pitch for dead notes, harmonics, bends and slides; also report pitch-agnostic onset matching.',
            'Articulation subgroup recall uses one global matching, not separate favorable rematching.',
            'No grid, known tempo, pattern annotation, score or technique labels enter inference.',
            'Single isolated-bass corpus; potential upstream training overlap. No mix/genre generalization or LN acceptance claim.'])
    target = args.output/'manifest.json'
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(cases=len(cases), manifest_sha256=sha(target),
        reference_notes=sum(c['reference_count'] for c in cases),
        unavailable_interval_cases=[c['name'] for c in cases if c['scoring_unobserved_tails']])))


if __name__ == '__main__':
    main()
