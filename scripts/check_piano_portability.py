"""Check all saved upstream decodes and the fixed three portable long windows.

Uses completed inference artifacts only; this does not run either network or
consult reference note labels. Save every comparison, including failed ones.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.piano import decode, MODEL_SHA256
from scripts.probe_mapper_adapter import sha


def compare(reference, predicted):
    order = lambda notes: sorted(notes, key=lambda n: (n['pitch'], n['onset'], n['offset']))
    reference, predicted = order(reference), order(predicted)
    result = dict(reference=len(reference), predicted=len(predicted), passed=False)
    if len(reference) != len(predicted):
        return result
    result['same_pitch_velocity'] = all((a['pitch'], a['velocity']) == (b['pitch'], b['velocity'])
                                        for a, b in zip(reference, predicted))
    for field in ('onset', 'offset'):
        result['max_'+field+'_error_ms'] = max(
            (abs(a[field]-b[field])*1000 for a, b in zip(reference, predicted)), default=0.)
    result['same_round_ms'] = all(round(a[field]*1000) == round(b[field]*1000)
                                 for a, b in zip(reference, predicted) for field in ('onset', 'offset'))
    result['passed'] = (result['same_pitch_velocity'] and result['same_round_ms']
                        and max(result['max_onset_error_ms'], result['max_offset_error_ms']) <= .01)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--specialist', type=Path, default=Path('test-data/smd/selected-v1/benchmark/specialist-v2'))
    cli.add_argument('--portable', type=Path, default=Path('test-data/piano-onnx/portable-v1'))
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous comparisons')
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    official = read(args.specialist/'report.json')
    portable = read(args.portable/'report.json')
    manifest = read(args.specialist.parent/'manifest.json')
    if (official['identity']['runner_sha256'] != sha(args.specialist/'runner-snapshot.py')
            or official['identity']['manifest_sha256'] != sha(args.specialist.parent/'manifest.json')
            or {r['name'] for r in official['cases']} != {r['name'] for r in manifest['cases']}):
        raise ValueError('Incomplete or changed original inference')
    if (portable['identity']['model_sha256'] != MODEL_SHA256
            or portable['identity']['runtime_sha256'] != sha(args.portable/'runtime-snapshot.py')):
        raise ValueError('Portable run identity changed')
    rows = []
    originals = {}
    for case in official['cases']:
        name = case['name']
        path = args.specialist/(name+'-notes.json')
        if sha(path) != case['notes_sha256']:
            raise ValueError('Upstream events changed: '+name)
        originals[name] = read(path)
        activation_path = args.specialist/(name+'-activations.npz')
        with np.load(activation_path, allow_pickle=False) as data:
            predicted = decode(data)
        rows.append(dict(name=name, kind='decoder_on_upstream_activations',
                         activations_sha256=sha(activation_path), notes_sha256=sha(path),
                         release_reasons=dict(Counter(n['release_reason'] for n in predicted)),
                         **compare(originals[name], predicted)))
    # This exact subset was selected before running either backend.
    selected = {manifest['cases'][i]['name'] for i in (0, 5, 10)}
    for case in portable['cases']:
        name = case['name']
        if name not in selected:
            continue
        path = args.portable/(name+'-notes.json')
        original_case = next(r for r in official['cases'] if r['name'] == name)
        if (sha(path) != case['notes_sha256']
                or case['audio_sha256'] != original_case['audio_sha256']
                or case['duration_seconds'] != original_case['duration_seconds']):
            raise ValueError('Different portable input or changed events: '+name)
        rows.append(dict(name=name, kind='portable_full_inference', notes_sha256=sha(path),
                         **compare(originals[name], read(path))))
    if {r['name'] for r in rows if r['kind'] == 'portable_full_inference'} != selected:
        raise ValueError('Portable comparison is incomplete')
    result = dict(runner_sha256=sha(__file__), runtime_sha256=sha('manialab/piano.py'),
                  official_report_sha256=sha(args.specialist/'report.json'),
                  portable_report_sha256=sha(args.portable/'report.json'),
                  rows=rows, passed=all(r['passed'] for r in rows),
                  notice='Numerical/runtime consistency only, not musical acceptance. No network was rerun.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
