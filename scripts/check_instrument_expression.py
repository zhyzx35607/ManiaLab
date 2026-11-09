"""Compare full-song accompaniment evidence through actual selection and layout."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.audio import source_evidence,SR
from manialab.arrange import parameters,assign_lanes,layout_metrics
from manialab.domain import UNOBSERVED_RELEASES,validate,export_osu,parse_osu
from manialab.expression import propose_expression,expression_evidence,select_expression
from manialab.timing import chart_clock_notes,align_candidates,grid_report
from scripts.instrument_candidate_adapter import accompaniment_candidates
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input',type=Path,default=Path('test-data/sakura-benchmark/piano-full-pipeline-v1'))
    cli.add_argument('--instrument',type=Path,default=Path('test-data/sakura-benchmark/other-float-yourmt3-v1'))
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve completed runs')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    baseline=read(args.input/'report.json')
    frozen=read(args.input/'specialist-analysis.json')
    report=read(args.instrument/'report.json')
    audit=read(args.instrument/'release-audit.json')
    if baseline['reference_chart_read'] or frozen['source_bleed_rejections']:
        raise ValueError('This controlled comparison requires the untouched, audio-only baseline')
    for name in ('notes','tokens'):
        path=args.instrument/(name+('.json' if name=='notes' else '.npz'))
        if sha(path)!=audit[name+'_sha256'] or sha(path)!=report[name+'_sha256']:
            raise ValueError('Audited model events changed')
    if sha(args.instrument/'runner-snapshot.py')!=report['identity']['runner_sha256']:
        raise ValueError('Inference runner identity changed')
    if (sha(args.instrument/'release-provenance.json')!=audit['provenance_sha256'] or
        sha(args.instrument/'audit-runner-snapshot.py')!=audit['runner_sha256']):
        raise ValueError('Release provenance changed after token audit')
    inputs=report['identity']['pcm_inputs']
    if len(inputs)!=1 or inputs[0]['input_dtype']!='float32':raise ValueError('Expected one original floating source')
    source=Path(inputs[0]['path'])
    if sha(source)!=inputs[0]['file_sha256']:raise ValueError('Inference PCM changed')
    rate,signal=wavfile.read(source)
    source_cache=args.input/'stems-provided'/frozen['separation']['set_sha256']/'other.npy'
    if sha(source_cache)!=frozen['separation']['decoded_files']['other']['npy_sha256']:
        raise ValueError('Application source cache changed')
    original=np.load(source_cache,allow_pickle=False)
    if rate!=SR or not np.array_equal(signal,original):raise ValueError('Not the same source used by the application')
    mix_rate,mix=wavfile.read(source.with_name('mix.wav'))
    if mix_rate!=rate or len(mix)!=len(signal):raise ValueError('Mixture and source clocks differ')
    # Mix identity was frozen by the earlier PCM audit; no newly normalized copy.
    pcm_audit=read(source.parent/'report.json')
    mix_identity=next(row for row in pcm_audit['rows'] if row['role']=='mix')
    if sha(source.with_name('mix.wav'))!=mix_identity['raw_float_sha256']:
        raise ValueError('Original floating mixture changed')
    raw=read(args.instrument/'notes.json')
    notes,trace=accompaniment_candidates(raw,read(args.instrument/'release-provenance.json'),len(signal)*1000/rate)
    filtered,rejected=source_evidence(notes,signal,mix)
    kept={n.id for n in filtered}
    for row in trace:
        if row['decision']=='candidate' and row['id'] not in kept:row['decision']='below_source_energy_floor'
    revised=copy.deepcopy(frozen)
    revised['candidates']=sorted([n for n in revised['candidates'] if n['voice']!='other']+
        [n.data() for n in filtered],key=lambda n:(n['start'],n['pitch'],n['voice']))
    revised['phrase_candidates']=[n for n in revised['phrase_candidates'] if n['voice']!='other']
    revised['instrument_engine']='research-yourmt3'
    revised['structure_frozen_for_comparison']=True
    for summary in revised['voices']:
        if summary['voice']=='other':
            summary.update(candidates=len(filtered),detector='yourmt3-research',cached=True,
                rejected_low_energy_pitch=rejected,release_refinement=audit,articulation_rejections=[],
                sustains_over_800ms=sum(n.end-n.start>=800 for n in filtered))
    assert all(n==next(m for m in revised['candidates'] if m['id']==n['id'])
               for n in frozen['candidates'] if n['voice']!='other')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    snapshots={}
    for name in ('scripts/instrument_candidate_adapter.py','manialab/expression.py','manialab/arrange.py',
                 'manialab/domain.py','manialab/timing.py','manialab/audio.py'):
        target=args.output/(Path(name).stem+'-snapshot.py')
        shutil.copyfile(name,target);snapshots[name]=sha(target)
    save=lambda name,value:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    save('instrument-trace.json',trace)
    save('instrument-analysis.json',revised)
    identity=dict(runner_sha256=sha(__file__),snapshots=snapshots,
        protocol_sha256=sha('docs/accompaniment-transcription-protocol.md'),
        original_analysis_sha256=sha(args.input/'specialist-analysis.json'),
        inference_report_sha256=sha(args.instrument/'report.json'),inference=report['identity'],
        release_audit=audit,generation_clock=baseline['identity']['generation_clock'],
        source_pcm_equal=True,reference_chart_read=False,unit_event_weights_are_not_confidence=True)
    cases=[]
    for backend,keys in [('basic-other',10),*[('instrument-other',k) for k in range(6,11)]]:
        began=time.monotonic();name=f'{backend}-{keys}k'
        analysis=copy.deepcopy(frozen if backend=='basic-other' else revised)
        settings=parameters(baseline['identity']['settings']|{'keys':keys})
        clock=identity['generation_clock']
        plan=propose_expression(analysis,settings)
        evidence,metadata,decisions=expression_evidence(analysis,plan,settings)
        aligned,alignment=align_candidates(chart_clock_notes(evidence,clock['audio_shift_ms']),clock,confirmed=True)
        selected,selection=select_expression(aligned,metadata,decisions,settings,60000/clock['bpm'])
        originals={n.id:n for n in aligned}
        assert all((n.start,n.end,n.voice,n.pitch)==(originals[n.id].start,originals[n.id].end,
                   originals[n.id].voice,originals[n.id].pitch) for n in selected)
        placed=assign_lanes(selected,settings,beat_ms=60000/clock['bpm'])
        structural=validate(placed,keys);grid=grid_report(placed,clock)
        assert structural['valid'] and grid['off_grid']==0 and len(placed)==len(selected)
        assert all(n.voice!='vocals' for n in placed)
        assert all(not n.hold or n.release_reason not in UNOBSERVED_RELEASES for n in placed)
        content=export_osu(placed,keys,Path('vendor/mapperatorinator/template.osu'),
            title='Sakura accompaniment evidence comparison',audio='audio.mp3',bpm=clock['bpm'],offset=clock['offset'],
            version=name+' research',grid_division=clock['division'])
        assert validate(parse_osu(content)['notes'],keys)['valid']
        (args.output/(name+'.osu')).write_text(content,encoding='utf-8')
        save(name+'-plan.json',plan);save(name+'-selection.json',selection);save(name+'-notes.json',[n.data() for n in placed])
        cases.append(dict(name=name,elapsed_seconds=time.monotonic()-began,structural=structural,grid=grid,
            alignment=alignment,selection_optimal=selection['attack_optimal'],
            source_candidates=dict(Counter(n['voice'] for n in analysis['candidates'])),
            source_counts=dict(Counter(n.voice for n in placed)),
            source_attacks=dict(Counter(v for v,t in {(n.voice,n.start) for n in placed})),
            source_holds=dict(Counter(n.voice for n in placed if n.hold)),
            retained_large_attack_shifts=dict(Counter(n.voice for n in placed if abs(n.start-n.evidence_start)>30)),
            primary_voice_by_section=[r['main_voice'] for r in plan['sections']],
            omission_reasons=selection['omission_reasons'],layout=layout_metrics(placed,keys,60000/clock['bpm'])))
        save('progress.json',cases)
        print(json.dumps({k:cases[-1][k] for k in ('name','elapsed_seconds','source_counts','source_holds',
            'retained_large_attack_shifts')}),flush=True)
    shutil.copyfile(args.input/'audio.mp3',args.output/'audio.mp3')
    save('report.json',dict(identity=identity,cases=cases,raw_events=len(raw),source_energy_rejections=rejected,
        instrument_programs=report['by_program'],adapter_decisions=dict(Counter(r['decision'] for r in trace)),
        endpoint_counts=audit['endpoint_counts'],default_enabled=False,portable_runtime_verified=False,
        visual_check=False,musical_acceptance=False,
        notice='Controlled research replacement of other source only. Unit event weights are not calibrated confidence. Manual timing; no author-reference scoring.'))


if __name__=='__main__':main()
