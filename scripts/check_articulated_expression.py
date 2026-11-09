"""Compare frozen full-song evidence under three explicit expression policies."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters,assign_lanes,layout_metrics
from manialab.domain import Note,UNOBSERVED_RELEASES,validate,export_osu,parse_osu
from manialab.expression import propose_expression,expression_evidence,select_expression,validate_expression
from manialab.piano import decode,OUTPUTS
from scripts.probe_mapper_adapter import sha
from manialab.timing import chart_clock_notes,align_candidates,grid_report
from scripts.research_expression_policies import propose


def smd_checks(output):
    """Keep every strong attack and raw interval; do not score authored style."""
    root=Path('test-data/smd/selected-v1/benchmark')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(root/'manifest.json');inference=read(root/'specialist-v2/report.json')
    if inference['identity']['manifest_sha256']!=sha(root/'manifest.json'):
        raise ValueError('Frozen piano dataset changed')
    rows=[]
    for case in manifest['cases']:
        name=case['name'];source=root/'specialist-v2'/(name+'-activations.npz')
        expected=next(r for r in inference['cases'] if r['name']==name)
        if sha(root/'specialist-v2'/(name+'-notes.json'))!=expected['notes_sha256']:
            raise ValueError('Piano event archive changed')
        with np.load(source,allow_pickle=False) as d:
            events=decode({key:d[key] for key in OUTPUTS})
        frozen=read(root/'specialist-v2'/(name+'-notes.json'))
        key=lambda n:(n['pitch'],n['onset'],n['offset'])
        assert sorted(map(key,events))==sorted(map(key,frozen))
        notes=[Note(round(n['onset']*1000),round(n['offset']*1000),pitch=n['pitch'],strength=n['onset_score'],
            voice='piano',source='pitch',id=f'p:{i}',release_reason=n['release_reason'])
            for i,n in enumerate(events) if 0<=n['onset']<case['score_seconds']]
        analysis=dict(duration=case['duration_seconds'],candidates=[n.data() for n in notes],
            structure=dict(segments=[dict(start_ms=0,end_ms=round(case['score_seconds']*1000))]))
        saved=copy.deepcopy(analysis);settings=parameters();plan=propose(analysis,settings,articulated=True)
        proposed,_,trace=expression_evidence(analysis,plan,settings)
        strong={n.id:n for n in notes if n.strength>=.35}
        assert {n.id for n in proposed}==set(strong)
        assert all(n.start==strong[n.id].start and n.pitch==strong[n.id].pitch and
            n.end in (n.start,strong[n.id].end) for n in proposed)
        assert all(not n.hold or n.release_reason not in UNOBSERVED_RELEASES for n in proposed)
        assert analysis==saved
        row=dict(name=name,activation_sha256=sha(source),raw_events=len(notes),
            strong_attacks=len(strong),expressed_attacks=len(proposed),holds=sum(n.hold for n in proposed),
            articulation_body=sum(r['reason']=='articulated_attack_run' for r in trace.values()),
            original_candidate_data_unchanged=True,all_strong_attacks_retained=True,
            original_or_tap_end_only=True,unsafe_tail_holds=0)
        rows.append(row)
    (output/'smd-invariants.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return rows


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input',type=Path,default=Path('test-data/sakura-benchmark/instrument-expression-v1'))
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--independent',action='store_true')
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve all earlier comparisons')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    previous=read(args.input/'report.json')
    basic_path=Path('test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json')
    if sha(basic_path)!=previous['identity']['original_analysis_sha256']:
        raise ValueError('Frozen baseline changed')
    analyses={'basic-other':read(basic_path),'instrument-other':read(args.input/'instrument-analysis.json')}
    baseline_report=read(basic_path.parent/'report.json')
    args.output.mkdir(parents=True)
    save=lambda name,value:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    snapshots={}
    for name in ('manialab/expression.py','manialab/expression_articulation.py','scripts/research_expression_policies.py','manialab/arrange.py',
                 'manialab/domain.py','manialab/timing.py','manialab/piano.py'):
        dest=args.output/(Path(name).stem+'-snapshot.py');shutil.copyfile(name,dest);snapshots[name]=sha(dest)
    identity=dict(runner_sha256=sha(__file__),snapshots=snapshots,
        protocol_sha256=sha('docs/independent-expression-protocol.md' if args.independent else 'docs/articulated-expression-protocol.md'),
        previous_report_sha256=sha(args.input/'report.json'),
        analyses_sha256={'basic-other':sha(basic_path),'instrument-other':sha(args.input/'instrument-analysis.json')},
        generation_clock=previous['identity']['generation_clock'],reference_chart_read=False,network_rerun=False)
    cases=[]
    specs=[(source,mode,10) for source in analyses for mode in ('old-plan','activity-only','articulation')]
    specs += [('instrument-other','articulation',keys) for keys in range(6,10)]
    if args.independent:
        specs=[('basic-other','independent',10),*[('instrument-other','independent',keys) for keys in range(6,11)]]
    for source,mode,keys in specs:
        name=f'{source}-{mode}-{keys}k';start=time.monotonic()
        analysis=analyses[source];original=copy.deepcopy(analysis)
        settings=parameters(baseline_report['identity']['settings']|{'keys':keys})
        if mode=='old-plan':
            plan=read(args.input/(source+'-10k-plan.json'))
        else:
            plan=propose(analysis,settings,articulated=mode in ('articulation','independent'),independent=mode=='independent')
        validate_expression(plan,analysis)
        evidence,metadata,trace=expression_evidence(analysis,plan,settings)
        decision_snapshot=copy.deepcopy(trace)
        clock=identity['generation_clock']
        aligned,alignment=align_candidates(chart_clock_notes(evidence,clock['audio_shift_ms']),clock,confirmed=True)
        selected,selection=select_expression(aligned,metadata,trace,settings,60000/clock['bpm'])
        placed=assign_lanes(selected,settings,beat_ms=60000/clock['bpm'])
        structural=validate(placed,keys);grid=grid_report(placed,clock)
        assert structural['valid'] and grid['off_grid']==0 and len(placed)==len(selected)
        aligned_by_id={n.id:n for n in aligned}
        assert all((n.start,n.end,n.voice,n.pitch)==(aligned_by_id[n.id].start,aligned_by_id[n.id].end,
            aligned_by_id[n.id].voice,aligned_by_id[n.id].pitch) for n in placed)
        assert all(not n.hold or n.release_reason not in UNOBSERVED_RELEASES for n in placed)
        assert all(n.voice!='vocals' for n in placed) and analysis==original
        if mode=='old-plan':
            old=read(args.input/(source+'-10k-notes.json'))
            fields=lambda n:(n['id'],n['start'],n['end'],n['pitch'],n['voice'],n['lane'])
            assert sorted(map(fields,old))==sorted(fields(n.data()) for n in placed)
        content=export_osu(placed,keys,Path('vendor/mapperatorinator/template.osu'),
            title='Sakura articulated expression comparison',audio='audio.mp3',bpm=clock['bpm'],offset=clock['offset'],
            version=name+' research',grid_division=clock['division'])
        assert validate(parse_osu(content)['notes'],keys)['valid']
        (args.output/(name+'.osu')).write_text(content,encoding='utf-8')
        save(name+'-plan.json',plan);save(name+'-selection.json',selection);save(name+'-notes.json',[n.data() for n in placed])
        grouped=Counter((n.voice,n.start) for n in placed)
        row=dict(name=name,source=source,mode=mode,keys=keys,elapsed_seconds=time.monotonic()-start,
            structural=structural,grid=grid,alignment=alignment,selection_optimal=selection['attack_optimal'],
            source_counts=dict(Counter(n.voice for n in placed)),
            source_attacks=dict(Counter(v for v,t in grouped)),
            source_holds=dict(Counter(n.voice for n in placed if n.hold)),
            expressed_source_attacks=dict(Counter(v for v,t in {(n.voice,n.start) for n in evidence})),
            articulation_runs=len({(r['section'],r['voice'],r['articulation']['run']) for r in decision_snapshot.values()
                if r.get('articulation',{}).get('tap')}),
            expression_reasons=dict(Counter(r['reason'] for r in decision_snapshot.values())),
            retained_large_attack_shifts=dict(Counter(n.voice for n in placed if abs(n.start-n.evidence_start)>30)),
            main_voices=[r['main_voice'] for r in plan['sections']],
            omission_reasons=selection['omission_reasons'],layout=layout_metrics(placed,keys,60000/clock['bpm']),
            original_candidates_unchanged=True,retained_endpoints_preserved_after_selection=True,
            old_output_exactly_reproduced=mode=='old-plan')
        cases.append(row);save('progress.json',cases)
        print(json.dumps({k:row[k] for k in ('name','source_counts','source_holds','retained_large_attack_shifts','articulation_runs')}),flush=True)
    independent=smd_checks(args.output)
    shutil.copyfile(args.input/'audio.mp3',args.output/'audio.mp3')
    save('report.json',dict(identity=identity,cases=cases,smd=independent,visual_check=False,musical_acceptance=False,
        notice='Fixed rule comparison on detected evidence. Explicit musical-expression choice; no claim of physical key-release inference or musical acceptance. Manual chart timing.'))


if __name__=='__main__':main()
