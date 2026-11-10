"""Compare the fixed Transkun model on historical SMD and the full Nova piano."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import input_pcm, load_model, sha256, transcribe, verify_source
from scripts.evaluate_piano_acoustics import counts, metrics, pairs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def pcm_digest(samples):
    return hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest()


def prepare_input(path, seconds):
    rate, raw = wavfile.read(path)
    if rate != 22050 or raw.ndim != 1 or raw.dtype not in (np.dtype('int16'), np.dtype('float32')):
        raise ValueError('Expected original 22050 Hz mono PCM')
    source = raw[:round(seconds * rate)].astype(np.float32)
    if len(source) != round(seconds * rate):
        raise ValueError('Input shorter than frozen scope')
    if raw.dtype == np.int16:
        source /= 32768
    shared = resample_poly(source, 320, 441).astype(np.float32)
    return shared, input_pcm(shared, 16000)


def onset_comparison(reference, old, new, strict=False):
    old_matches = pairs(reference, old, strict=strict)
    new_matches = pairs(reference, new, strict=strict)
    old_ids = {i for i, _ in old_matches}
    new_ids = {i for i, _ in new_matches}
    bands = {}
    for label, lower, upper in [('low', 21, 47), ('middle', 48, 83), ('high', 84, 108)]:
        ids = {i for i, n in enumerate(reference) if lower <= n['pitch'] <= upper}
        bands[label] = dict(reference=len(ids),
            kong=counts(len(ids & old_ids), len(ids), sum(lower <= n['pitch'] <= upper for n in old)),
            transkun=counts(len(ids & new_ids), len(ids), sum(lower <= n['pitch'] <= upper for n in new)),
            recovered=len(ids & (new_ids - old_ids)), new_missing=len(ids & (old_ids - new_ids)))
    return dict(kong=counts(len(old_ids), len(reference), len(old)),
                transkun=counts(len(new_ids), len(reference), len(new)),
                recovered_ref_indices=sorted(new_ids-old_ids),
                new_missing_ref_indices=sorted(old_ids-new_ids),
                kong_matches=old_matches, transkun_matches=new_matches, bands=bands)


def summarize_music(rows):
    smd = [r for r in rows if r['kind'] == 'smd']
    summary = dict(recordings=len(smd), models={}, recovered={}, new_missing={}, bands={})
    for tolerance in ('50ms', '40ms'):
        comparisons = [r['comparison'][tolerance] for r in smd]
        summary['recovered'][tolerance] = sum(len(c['recovered_ref_indices']) for c in comparisons)
        summary['new_missing'][tolerance] = sum(len(c['new_missing_ref_indices']) for c in comparisons)
        summary['models'][tolerance] = {}
        for engine in ('kong', 'transkun'):
            values = [c[engine] for c in comparisons]
            summary['models'][tolerance][engine] = dict(
                pooled=counts(*(sum(v[k] for v in values) for k in ('matched', 'reference', 'predicted'))),
                macro_f1=float(np.mean([v['f1'] for v in values])),
                per_recording_f1={r['name']:r['comparison'][tolerance][engine]['f1'] for r in smd})
        summary['bands'][tolerance] = {}
        for band in ('low', 'middle', 'high'):
            values = [c['bands'][band] for c in comparisons]
            summary['bands'][tolerance][band] = dict(
                recovered=sum(v['recovered'] for v in values), new_missing=sum(v['new_missing'] for v in values),
                **{engine:counts(*(sum(v[engine][k] for v in values)
                           for k in ('matched', 'reference', 'predicted'))) for engine in ('kong', 'transkun')})
    return summary


def nova_comparison(old, new, text):
    from manialab.domain import parse_osu
    from scripts.probe_mapper_adapter import event_similarity
    human = [n for n in parse_osu(text)['notes'] if n.lane in (4, 5)]
    human_times = [(n.start,) for n in human]
    # Keep the historical non-chaining 30 ms grouping without importing the
    # audio inference dependencies of probe_sakura_sources into this runtime.
    def compare(notes, shift):
        times = sorted(n['onset']*1000-shift for n in notes)
        groups = []
        for t in times:
            if not groups or t-groups[-1][0] > 30:
                groups.append([t])
            else:
                groups[-1].append(t)
        return dict(onset_groups=event_similarity(sorted(set(human_times)),
            [(float(np.median(g)),) for g in groups], [40]),
            note_multiplicity=event_similarity(human_times, [(t,) for t in times], [40]),
            duration_notice='Authored onset/layer diagnostic, not physical pitch or release truth.')
    matched = pairs(old, new)
    return dict(candidates=dict(kong=len(old), transkun=len(new)),
        same_pitch_50ms=dict(matched=len(matched), kong_only=len(old)-len(matched),
                            transkun_only=len(new)-len(matched)),
        author_onset_diagnostic={engine:{str(shift):compare(notes, shift) for shift in (0., 22.5)}
            for engine, notes in [('kong', old), ('transkun', new)]},
        acoustic_pitch_truth_available=False, acoustic_offset_truth_available=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--resume', action='store_true')
    args = cli.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use local test-data output')
    dataset = ROOT/'test-data/smd/selected-v1/benchmark'
    baseline = dataset/'specialist-v2'
    portable = ROOT/'test-data/piano-onnx/portable-v1'
    controls = ROOT/'test-data/piano-transkun-controls-v1'
    source = ROOT/'test-data/piano-acoustic-alternatives-v1'
    paths = dict(manifest=dataset/'manifest.json', baseline=baseline/'report.json',
                 portable=portable/'report.json', controls=controls/'report.json',
                 reference=ROOT/'test-data/sakura-benchmark/reference.osu',
                 protocol=ROOT/'docs/piano-transkun-music-protocol.md', runner=Path(__file__),
                 runtime=ROOT/'scripts/transkun_runtime.py',
                 scoring=ROOT/'scripts/evaluate_piano_acoustics.py',
                 grouping=ROOT/'scripts/probe_sakura_sources.py',
                 author_matching=ROOT/'scripts/probe_mapper_adapter.py',
                 domain=ROOT/'manialab/domain.py')
    fixed = dict(manifest='f1e7bcffc17e9aa5399aa2e923939748e3c9cc2a29e8e1b0a766e2fbc20e3c6b',
        baseline='ab23fa8f30d79e6dcb2a167eeaaada3c47ed3bfdfba55c16e99448125d39cda6',
        portable='4354670c70f39a0420f99a33f1bc9b8c1715d043a19aed363cdd3dd88fb21a30',
        reference='a25e9048747e169568808be790051aed6e6470142ea6a8bf27378d631a2526b3')
    for key, digest in fixed.items():
        if sha256(paths[key]) != digest:
            raise ValueError('Fixed evidence changed: '+key)
    control_report = read(paths['controls'])
    if control_report['unique_control_inferences'] != 91 or not control_report['wrapper_verification']['identical']:
        raise ValueError('Complete controls before real-music inference')
    manifest, old_report = read(paths['manifest']), read(paths['baseline'])
    if old_report['identity']['manifest_sha256'] != fixed['manifest']:
        raise ValueError('Baseline manifest mismatch')
    plans = []
    for case, old in zip(manifest['cases'], old_report['cases'], strict=True):
        if case['name'] != old['name'] or case['audio_sha256'] != old['audio_sha256']:
            raise ValueError('Baseline source mismatch')
        for field in ('audio', 'labels', 'sustain_labels'):
            if sha256(ROOT/case[field]) != case[field+'_sha256']:
                raise ValueError('Source changed: '+case['name'])
        note_path = baseline/(case['name']+'-notes.json')
        if sha256(note_path) != old['notes_sha256']:
            raise ValueError('Baseline notes changed')
        plans.append(dict(case, kind='smd', baseline_notes=str(note_path),
                          baseline_sha256=old['notes_sha256']))
    nova = next(r for r in read(paths['portable'])['cases'] if r['name']=='sakura-nova-piano')
    nova_audio = ROOT/'test-data/sakura-benchmark/raw-pcm-audit-v1/piano.wav'
    nova_notes = portable/'sakura-nova-piano-notes.json'
    if (sha256(nova_audio) != nova['audio_sha256'] or sha256(nova_notes) != nova['notes_sha256']):
        raise ValueError('Nova baseline changed')
    plans.append(dict(name='sakura-nova-piano', kind='nova', audio=str(nova_audio),
        audio_sha256=nova['audio_sha256'], duration_seconds=283.84,
        baseline_notes=str(nova_notes), baseline_sha256=nova['notes_sha256']))
    if len(plans) != 12:
        raise ValueError('Unexpected scope')
    model, model_identity = load_model(source)
    if model_identity != control_report['identity']['model']:
        raise ValueError('Model differs from controls')
    if sha256(paths['runtime']) != control_report['identity']['code']['runtime']:
        raise ValueError('Runtime differs from controls')
    identity = dict(model=model_identity, files={k:sha256(p) for k,p in paths.items()},
                    plans=plans, training_performed=False, application_enabled=False)
    progress = []
    if output.exists():
        if not args.resume or read(output/'identity.json') != identity:
            raise ValueError('Preserve earlier output; resume needs identical evidence')
        progress = read(output/'progress.json') if (output/'progress.json').exists() else []
        if [r['name'] for r in progress] != [p['name'] for p in plans[:len(progress)]]:
            raise ValueError('Resume order changed')
        for row in progress:
            for name, digest in row['artifacts_sha256'].items():
                if sha256(output/name) != digest:
                    raise ValueError('Completed output changed')
    else:
        output.mkdir(parents=True)
        save(output/'identity.json', identity)
        for key in ('runner', 'runtime', 'protocol', 'scoring', 'grouping', 'author_matching', 'domain'):
            shutil.copyfile(paths[key], output/(key+'-snapshot'+paths[key].suffix))
    for plan in plans[len(progress):]:
        name = plan['name']
        shared, pcm = prepare_input(ROOT/plan['audio'], plan['duration_seconds'])
        a, b = output/(name+'-16k.npy'), output/(name+'-44k.npy')
        np.save(a, shared, allow_pickle=False)
        np.save(b, pcm, allow_pickle=False)
        start = time.monotonic()
        try:
            trace = transcribe(model, pcm)
        except Exception as error:
            save(output/'failure.json', dict(name=name, error_type=type(error).__name__, message=str(error)))
            raise
        elapsed = time.monotonic()-start
        old, new = read(plan['baseline_notes']), trace['piano_events']
        result = dict(name=name, kind=plan['kind'], trace=trace, elapsed_seconds=elapsed,
                      pcm16_sha256=pcm_digest(shared), pcm44_sha256=pcm_digest(pcm))
        if plan['kind'] == 'smd':
            scope = lambda notes:[n for n in notes if 0 <= n['onset'] < plan['score_seconds']]
            key = scope(read(ROOT/plan['labels']))
            sustain = scope(read(ROOT/plan['sustain_labels']))
            if [(n['pitch'], n['onset']) for n in key] != [(n['pitch'], n['onset']) for n in sustain]:
                raise ValueError('Reference onset semantics differ')
            old_scored, new_scored = scope(old), scope(new)
            result['comparison'] = {label:onset_comparison(key, old_scored, new_scored, strict)
                                    for label,strict in [('50ms',False), ('40ms',True)]}
            result['metrics'] = {engine:{semantics:{label:metrics(ref, notes, strict=strict)
                for label,strict in [('50ms',False), ('40ms',True)]}
                for semantics,ref in [('key',key), ('sustain',sustain)]}
                for engine,notes in [('kong',old_scored), ('transkun',new_scored)]}
        else:
            result['comparison'] = nova_comparison(old, new, paths['reference'].read_text(encoding='utf-8-sig'))
        result_path = output/(name+'-result.json')
        save(result_path, result)
        progress.append(dict(name=name, artifacts_sha256={p.name:sha256(p) for p in (a,b,result_path)}))
        save(output/'progress.json', progress)
        comparison = result['comparison']
        compact = ({engine:comparison['50ms'][engine] for engine in ('kong','transkun')}
                   if plan['kind']=='smd' else comparison['same_pitch_50ms'])
        print(json.dumps(dict(completed=len(progress),total=len(plans),name=name,seconds=elapsed,
              events=len(new),comparison=compact)), flush=True)
    verify_source(source)
    if any(sha256(p) != identity['files'][k] for k,p in paths.items()):
        raise ValueError('Frozen evidence changed during inference')
    rows = [read(output/(plan['name']+'-result.json')) for plan in plans]
    report = dict(identity=identity, cases=progress, summary=summarize_music(rows),
        nova=rows[-1]['comparison'], musical_acceptance=False, direct_listening_performed=False,
        application_enabled=False, training_performed=False,
        artifacts_sha256={p.name:sha256(p) for p in output.iterdir() if p.name!='report.json'})
    save(output/'report.json', report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
