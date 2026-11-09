"""Compare pinned Transkun with frozen Kong outputs on all 91 audio controls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.piano_source_controls import score_attacks
from scripts.transkun_runtime import load_model, input_pcm, transcribe, verify_source, sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def summarize(rows):
    grouped = {}
    for row in rows:
        key = f'g{row["gain"]}_{row["condition"]}'
        group = grouped.setdefault(key, dict(cases=0, kong=Counter(), transkun=Counter(),
                                             pedal_events=0, new_missing_cases=0, recovered_cases=0))
        group['cases'] += 1
        for engine, score in [('kong', row['baseline_score']), ('transkun', row['score'])]:
            for metric in ('expected', 'predicted', 'matched', 'missing', 'extra'):
                group[engine][metric] += score[metric]
        group['pedal_events'] += len(row['trace']['pedal_events'])
        group['new_missing_cases'] += row['score']['missing'] > row['baseline_score']['missing']
        group['recovered_cases'] += row['score']['missing'] < row['baseline_score']['missing']
    by_name = {r['name']:r for r in rows}
    paired = []
    for mix in [r for r in rows if r['condition']=='weak_high']:
        high = mix['pair'][1]
        solo = by_name[f'g0.25-single-{high}' if mix['gain']==1 else f'g1-single-{high}']
        values = {}
        for engine, field in [('kong','baseline_score'), ('transkun','score')]:
            matched = {m['ref_index'] for m in mix[field]['matches']}
            values[engine] = dict(solo_high_matched=solo[field]['matched']==1,
                                 mixture_high_matched=1 in matched, mixture_low_matched=0 in matched,
                                 solo_extra=solo[field]['extra'], mixture_extra=mix[field]['extra'])
        paired.append(dict(gain=mix['gain'], pair=mix['pair'], solo=solo['name'], mixture=mix['name'], **values))
    new_missing = [r['name'] for r in rows if set(r['score']['missing_ref_indices'])-
                   set(r['baseline_score']['missing_ref_indices'])]
    recovered = sum(len(set(r['baseline_score']['missing_ref_indices'])-
                        set(r['score']['missing_ref_indices'])) for r in rows if r['condition']=='weak_high')
    # This is a control-level research gate, never music or release acceptance.
    checks = dict(weak_mixture_misses_reduced=recovered>0, no_new_misses=not new_missing,
                  total_extra_not_increased=sum(r['score']['extra'] for r in rows)<=
                                           sum(r['baseline_score']['extra'] for r in rows),
                  silence_empty=by_name['silence']['score']['predicted']==0)
    return dict(groups=grouped, matched_level_pairs=paired, new_missing_cases=new_missing,
                recovered_weak_onsets=recovered, control_checks=checks,
                all_control_checks_passed=all(checks.values()), musical_acceptance=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=ROOT/'test-data/piano-acoustic-alternatives-v1')
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--resume', action='store_true')
    args = cli.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use local test-data output')
    dataset = ROOT/'test-data/piano-isolated-controls-data-v1'
    baseline = ROOT/'test-data/piano-isolated-controls-run-v1'
    supplement = ROOT/'test-data/piano-matched-level-controls-v1'
    paths = dict(manifest=dataset/'manifest.json', baseline=baseline/'report.json',
                 supplement=supplement/'report.json', runner=Path(__file__),
                 runtime=ROOT/'scripts/transkun_runtime.py', matching=ROOT/'scripts/piano_source_controls.py',
                 protocol=ROOT/'docs/piano-transkun-controls-protocol.md')
    fixed = dict(manifest='63b6990ef00bba0c10b36fe491b1b527d1b53cbe403db24c85d4f4b7cce486d0',
        baseline='87323da6d3f3c5b0ec83dabb951082d761f72aae18e761245b4679635fe195e3',
        supplement='134b754e05173c8d4b995f06b6015c0a1e1dbc9bf2c9c8ff46033c9e872f8095')
    for key, digest in fixed.items():
        if sha256(paths[key]) != digest:
            raise ValueError('Fixed source changed: '+key)
    manifests = {key:read(paths[key]) for key in fixed}
    for key, folder in [('manifest',dataset), ('baseline',baseline), ('supplement',supplement)]:
        for name, digest in manifests[key]['artifacts_sha256'].items():
            if sha256(folder/name) != digest:
                raise ValueError('Fixed artifact changed: '+name)
    plans = []
    for case in manifests['manifest']['cases']:
        plans.append(dict(name=case['name'], gain=case['gain'], condition=case['condition'], pair=case['pair'],
            expected=case['expected'], file=str(dataset/case['file']), wav_sha256=case['wav_sha256'],
            baseline_result=str(baseline/(case['name']+'-result.json'))))
    for case in read(supplement/'inputs.json'):
        plans.append(dict(name=case['name'], gain=.25, condition='single', pair=None,
            expected=case['expected'], file=str(supplement/case['file']), wav_sha256=case['wav_sha256'],
            baseline_result=str(supplement/(case['name']+'-result.json'))))
    if len(plans)!=91 or len({p['name'] for p in plans})!=91:
        raise ValueError('Expected complete fixed control scope')
    model, model_identity = load_model(source)
    identity = dict(model=model_identity, code={key:sha256(path) for key,path in paths.items()},
                    matching_tolerance_seconds=.1, scope=91)
    progress = []
    if output.exists():
        if not args.resume or read(output/'identity.json')!=identity:
            raise ValueError('Preserve prior output; resume needs identical code, inputs and model')
        if (output/'progress.json').exists():
            progress=read(output/'progress.json')
        for row in progress:
            for file, digest in row['files_sha256'].items():
                if sha256(output/file)!=digest:
                    raise ValueError('Completed output changed')
        if [r['name'] for r in progress]!=[p['name'] for p in plans[:len(progress)]]:
            raise ValueError('Resume order changed')
    else:
        output.mkdir(parents=True)
        save(output/'identity.json', identity)
        save(output/'plans.json', plans)
        for key in ('runner','runtime','matching','protocol'):
            shutil.copyfile(paths[key],output/(key+'-snapshot'+paths[key].suffix))
    for plan in plans[len(progress):]:
        rate, original = wavfile.read(plan['file'])
        if rate!=16000 or original.shape!=(160000,) or original.dtype!=np.float32:
            raise ValueError('Fixed input shape changed')
        pcm = input_pcm(original,rate)
        if pcm.shape!=(441000,):
            raise ValueError('Resampling clock changed')
        pcm_path=output/(plan['name']+'-input.npy')
        np.save(pcm_path,pcm,allow_pickle=False)
        started=time.monotonic()
        trace=transcribe(model,pcm)
        elapsed=time.monotonic()-started
        score=score_attacks(plan['expected'],trace['piano_events'],tolerance_seconds=.1)
        baseline_result=read(Path(plan['baseline_result']))
        row=dict(name=plan['name'],gain=plan['gain'],condition=plan['condition'],pair=plan['pair'],
            expected=plan['expected'],original_wav_sha256=plan['wav_sha256'],
            resampled_pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
            elapsed_seconds=elapsed,trace=trace,score=score,baseline_score=baseline_result['score'])
        result_path=output/(plan['name']+'-result.json')
        save(result_path,row)
        progress.append(dict(name=plan['name'],elapsed_seconds=elapsed,
            files_sha256={p.name:sha256(p) for p in (pcm_path,result_path)}))
        save(output/'progress.json',progress)
        print(json.dumps(dict(completed=len(progress),total=91,name=plan['name'],seconds=elapsed,
             matched=score['matched'],expected=score['expected'],extra=score['extra'],
             pedals=len(trace['pedal_events']))),flush=True)
    verification_path=output/'wrapper-verification.json'
    if not verification_path.exists():
        name='g1-60-72-repeated_high'
        pcm=np.load(output/(name+'-input.npy'),allow_pickle=False)
        started=time.monotonic()
        plain=transcribe(model,pcm,record_windows=False)
        wrapped=read(output/(name+'-result.json'))['trace']
        identical=plain['final_events']==wrapped['final_events']
        save(verification_path,dict(name=name,identical=identical,plain_final_events=plain['final_events'],
             elapsed_seconds=time.monotonic()-started,extra_inference_for_wrapper_verification_only=True))
        if not identical:
            raise ValueError('Recording wrapper changed the official final events')
    elif not read(verification_path)['identical']:
        raise ValueError('Prior wrapper verification failed')
    verify_source(source)
    if any(sha256(path)!=identity['code'][key] for key,path in paths.items()):
        raise ValueError('Frozen code/protocol/input changed during run')
    rows=[read(output/(p['name']+'-result.json')) for p in progress]
    report=dict(identity=identity,cases=progress,summary=summarize(rows),
        unique_control_inferences=91,wrapper_verification=read(verification_path),
        notes_changed=False,application_enabled=False,training_performed=False,
        musical_acceptance=False,direct_listening_performed=False,
        artifacts_sha256={p.name:sha256(p) for p in output.iterdir() if p.name!='report.json'})
    save(output/'report.json',report)
    print(json.dumps(report['summary']['control_checks']),flush=True)


if __name__=='__main__':
    main()
