"""Re-decode frozen activations; check unchanged events and unsupported tails.

No inference, reference labels, threshold search, or automatic note deletion.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.io import wavfile
from manialab.piano import decode, OUTPUTS, MODEL_SHA256
from scripts.build_provenance import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous semantic checks')
    smd = ROOT/'test-data/smd/selected-v1/benchmark/specialist-v2'
    portable = ROOT/'test-data/piano-onnx/portable-v1'
    jobs = []
    for folder in (smd, portable):
        report = read(folder/'report.json')
        snapshot, key = (('runner-snapshot.py', 'runner_sha256') if folder == smd
                         else ('runtime-snapshot.py', 'runtime_sha256'))
        if sha256(folder/snapshot) != report['identity'][key]:
            raise ValueError('Historical code snapshot changed')
        for case in report['cases']:
            if folder == portable and case['name'] != 'sakura-nova-piano':
                continue
            notes_path = folder/(case['name']+'-notes.json')
            if sha256(notes_path) != case['notes_sha256']:
                raise ValueError('Historical physical notes changed')
            jobs.append((case['name'], folder/(case['name']+'-activations.npz'), notes_path))
    if len(jobs) != 12:
        raise ValueError('Expected eleven SMD cases and the full Sakura piano')
    args.output.mkdir(parents=True)
    sources = [Path(__file__), ROOT/'manialab/piano.py', ROOT/'manialab/domain.py']
    code = {path.name: sha256(path) for path in sources}
    for path in sources:
        shutil.copyfile(path, args.output/(path.stem+'-snapshot.py'))
    results = []
    for name, path, old_path in jobs:
        digest = sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in OUTPUTS}
        events, old = decode(arrays), read(old_path)
        if len(events) != len(old):
            raise ValueError('Physical event count changed')
        fields = ('onset', 'offset', 'pitch', 'velocity')
        order = lambda n: (n['pitch'], n['onset'], n['offset'])
        for now, before in zip(sorted(events, key=order), sorted(old, key=order)):
            if any(now[key] != before[key] for key in fields):
                raise ValueError(f'{name}: physical event changed')
        unsupported = [n for n in events if n['release_reason'] == 'frame_inactive']
        if any(n['release_observed'] for n in unsupported):
            raise ValueError('An inactive frame was claimed as an observed release')
        if sha256(path) != digest:
            raise ValueError('Activations changed during decode')
        details = dict(events=events, inactive_events=unsupported)
        (args.output/(name+'-events.json')).write_text(json.dumps(details, indent=2), encoding='utf-8')
        results.append(dict(name=name, activations_sha256=digest, old_notes_sha256=sha256(old_path),
            events=len(events), inactive_tails=len(unsupported), physical_events_identical=True,
            release_reasons=dict(Counter(n['release_reason'] for n in events))))
        if name == 'sakura-nova-piano':
            # Tie the portable activations to the actual application cache and
            # its complete float PCM, then verify every original candidate.
            root = ROOT/'test-data/sakura-benchmark'
            caches = list((root/'piano-full-pipeline-v1').rglob('piano-specialist-activations.npz'))
            if len(caches) != 1:
                raise ValueError('Ambiguous piano cache')
            rate, pcm = wavfile.read(root/'raw-pcm-audit-v1/piano.wav')
            with np.load(caches[0], allow_pickle=False) as saved:
                identity = json.loads(str(saved['identity']))
                if any(not np.array_equal(arrays[key], saved[key]) for key in OUTPUTS):
                    raise ValueError('Application cache differs from full portable inference')
            if (identity['sample_rate'] != rate or identity['samples'] != len(pcm)
                    or identity['model_sha256'] != MODEL_SHA256 or identity['pcm_sha256'] !=
                    hashlib.sha256(np.asarray(pcm, dtype='<f4').tobytes()).hexdigest()):
                raise ValueError('Cache is not tied to the original piano PCM')
            analysis_path = root/'piano-full-pipeline-v1/specialist-analysis.json'
            frozen = read(ROOT/'test-data/piano-svsep/probe-v3/report.json')
            if sha256(analysis_path) != frozen['identity']['analysis_sha256']:
                raise ValueError('Frozen application candidates changed')
            originals = {n['id']: n for n in read(analysis_path)['candidates'] if n['voice']=='piano'}
            changed = Counter()
            for index, now in enumerate(events):
                before = originals[f'piano:specialist:{index}']
                if ((before['start'], before['end'], before['pitch'], before['strength']) !=
                        (round(now['onset']*1000), round(now['offset']*1000), now['pitch'], now['onset_score'])):
                    raise ValueError('A frozen application candidate changed')
                if before['release_reason'] != now['release_reason']:
                    if now['release_reason'] != 'frame_inactive':
                        raise ValueError('Unexpected metadata change')
                    changed[before['release_reason']] += 1
            results[-1].update(original_pcm_verified=True, application_candidates_identical=True,
                reclassified_from=dict(changed))
    if any(sha256(path) != code[path.name] for path in sources):
        raise ValueError('Code changed during check')
    report = dict(code_sha256=code, cases=results, events=sum(r['events'] for r in results),
        inactive_tails=sum(r['inactive_tails'] for r in results), physical_events_identical=True,
        frame_threshold=.1, onset_threshold=.3, offset_threshold=.3,
        model_rerun=False, reference_labels_read=False, musical_acceptance=False,
        note='Frame absence is uncertainty, not proof of a false onset. No physical events were removed or moved.')
    report['artifacts_sha256'] = {p.name:sha256(p) for p in args.output.iterdir()}
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
