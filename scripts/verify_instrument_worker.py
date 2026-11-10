"""Run the packaged worker through the app adapter against saved real audio.

Fixtures are local research evidence, never bundled with the application.
This proves packaging/IPC parity, not music quality or model generalization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.instrument_bundle import MODELS, VOICE_MODELS, runtime_identity, sha
from manialab.instrument_detector import InstrumentDetector

EVIDENCE = ('notes.json', 'raw-notes.json', 'window-notes.json', 'interval-constructions.json')
HASH_KEYS = ('notes_sha256', 'raw_notes_sha256', 'window_notes_sha256', 'interval_constructions_sha256')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))


def pcm_float(path):
    rate, pcm = wavfile.read(path)
    if np.issubdtype(pcm.dtype, np.signedinteger):
        pcm = pcm.astype(np.float32) / float(2 ** (8 * pcm.dtype.itemsize - 1))
    elif pcm.dtype == np.uint8:
        pcm = (pcm.astype(np.float32) - 128.) / 128.
    elif pcm.dtype != np.float32:
        raise ValueError('Use the frozen float32 or integer PCM fixture')
    return int(rate), pcm


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--fixtures', type=Path, default=ROOT / 'test-data/tsumugi-smoke-v1')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    checked(not os.getenv('MANIALAB_INSTRUMENT_PYTHON'), 'Unset development override for portable verification')
    checked(not args.output.exists(), 'Preserve earlier runs; choose a new output directory')
    args.output.mkdir(parents=True)
    assets = ROOT / 'assets/instruments'
    before = read(assets / 'manifest.json')
    checked(before['worker_sources'] == runtime_identity(ROOT), 'Packaged sources are stale')
    detector = InstrumentDetector(ROOT)
    command, runtime = detector._runtime()
    checked(runtime['kind'] == 'portable', 'Verification must invoke the bundled EXE')
    fixtures = read(args.fixtures / 'manifest.json')
    rows = {row['model']: row for row in fixtures['cases']}
    cases = []
    for voice, model in VOICE_MODELS.items():
        fixture = rows[model]
        audio = Path(fixture['audio'])
        checked(sha(audio) == fixture['audio_sha256'], 'Frozen fixture changed')
        rate, pcm = pcm_float(audio)
        reference = args.fixtures / 'outputs-traced' / model
        reference_report = read(reference / 'report.json')
        reference_identity = reference_report['identity']
        checked(reference_identity['model_sha256'] == MODELS[model], 'Reference model differs')
        pcm_sha = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
        checked(pcm_sha == reference_identity['pcm_sha256'], 'Decoded fixture differs from research PCM')
        for name, key in zip(EVIDENCE, HASH_KEYS):
            checked(sha(reference / name) == reference_report[key], 'Reference evidence changed: ' + name)
        baseline = None
        for release in (False, True):
            arguments = dict(samples=pcm, sample_rate=rate, cache=args.output / ('cache-' + voice),
                stem_identity='portable-fixture:' + fixture['audio_sha256'], voice=voice,
                allow_predicted_releases=release)
            print(json.dumps(dict(stage='start', voice=voice, allow_predicted_releases=release)), flush=True)
            notes, diagnostics = detector.analyze(**arguments)
            checked(diagnostics['cached'] is False, 'First pass unexpectedly used cache')
            output = Path(diagnostics['evidence_directory'])
            result = read(output / 'result.json')
            for key in ('model_sha256', 'pcm_sha256', 'resampled_pcm_sha256', 'original_sample_rate',
                        'original_samples', 'original_channels', 'resampled_sample_rate', 'resampled_samples',
                        'settings', 'model_config', 'source_files'):
                checked(result['identity'][key] == reference_identity[key], 'Runtime identity differs: ' + key)
            for name in EVIDENCE:
                checked(read(output / name) == read(reference / name), 'Packaged prediction differs: ' + name)
            physical = [{k: v for k, v in n.data().items() if k not in ('id', 'release_reason')} for n in notes]
            if baseline is None:
                baseline = physical
                checked(result['predicted_release_hypotheses'] == 0, 'Default enabled model releases')
            else:
                checked(physical == baseline, 'Release option changed physical evidence')
                checked(result['predicted_release_hypotheses'] > 0, 'Fixture did not exercise predicted releases')
            cached, cache_diagnostics = detector.analyze(**arguments)
            checked(cache_diagnostics['cached'] is True and cache_diagnostics['evidence_directory'] == str(output),
                    'Second pass failed to reuse verified cache')
            checked([n.data() for n in notes] == [n.data() for n in cached], 'Cache changed notes')
            case = dict(voice=voice, model=model, fixture_sha256=fixture['audio_sha256'],
                reference_report_sha256=sha(reference / 'report.json'), pcm_sha256=pcm_sha,
                sample_rate=rate, samples=len(pcm), allow_predicted_releases=release, notes=len(notes),
                predicted_releases=result['predicted_release_hypotheses'], result_sha256=sha(output / 'result.json'),
                evidence_sha256=result['evidence_sha256'], raw_evidence_equal=True, cache_reused=True)
            cases.append(case)
            (args.output / f'{voice}-{release}.json').write_text(json.dumps(case, indent=2), encoding='utf-8')
            print(json.dumps(dict(stage='passed', voice=voice, releases=release, notes=len(notes))), flush=True)
    checked(read(assets / 'manifest.json') == before, 'Build manifest changed during verification')
    checked(runtime_identity(ROOT) == before['worker_sources'], 'Worker sources changed during verification')
    report = dict(worker_sha256=before['worker_sha256'], worker_sources=before['worker_sources'],
        verifier_sha256=sha(__file__), detector_sha256=sha(ROOT / 'manialab/instrument_detector.py'),
        models=MODELS, cases=cases, portable_runtime_verified=True, musical_acceptance=False,
        scope='Serial real-audio CPU EXE inference, research parity, both release modes and verified cache; not musical acceptance')
    payload = json.dumps(report, indent=2)
    (args.output / 'report.json').write_text(payload, encoding='utf-8')
    (assets / 'verification.json').write_text(payload, encoding='utf-8')
    manifest = before | dict(portable_runtime_verified=True, verified_models=list(MODELS),
        verification_sha256=sha(assets / 'verification.json'))
    temporary = assets / 'manifest.verified.tmp'
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary.replace(assets / 'manifest.json')
    print(json.dumps(dict(verified=True, cases=len(cases))), flush=True)


if __name__ == '__main__':
    main()
