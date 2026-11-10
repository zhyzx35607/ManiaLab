"""Exercise chord depth on frozen full-song evidence, without reference charts."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters,assign_lanes,layout_metrics
from manialab.domain import validate,export_osu,parse_osu
from manialab.expression import expression_evidence,select_expression
from manialab.timing import chart_clock_notes,align_candidates,grid_report
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input',type=Path,default=Path('test-data/sakura-benchmark/piano-full-pipeline-v1'))
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve completed runs')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    baseline=read(args.input/'report.json')
    if baseline['reference_chart_read']:raise ValueError('Input comparison read a reference chart')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    shutil.copyfile('manialab/expression.py',args.output/'expression-snapshot.py')
    save=lambda name,value:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    identity=dict(runner_sha256=sha(__file__),expression_sha256=sha('manialab/expression.py'),
        baseline_report_sha256=sha(args.input/'report.json'),reference_chart_read=False,
        generation_clock=baseline['identity']['generation_clock'],inputs={})
    cases=[]
    for engine,keys in [('general',10),*[('specialist',k) for k in range(6,11)]]:
        begin=time.monotonic();name=f'{engine}-{keys}k'
        analysis_path=args.input/(engine+'-analysis.json')
        plan_path=args.input/(engine+'-expression.json')
        identity['inputs'][engine]=dict(analysis_sha256=sha(analysis_path),plan_sha256=sha(plan_path))
        analysis,plan=read(analysis_path),read(plan_path)
        unchanged=copy.deepcopy(analysis)
        settings=parameters(baseline['identity']['settings']|{'keys':keys})
        clock=identity['generation_clock']
        evidence,meta,trace=expression_evidence(analysis,plan,settings)
        candidates,alignment=align_candidates(chart_clock_notes(evidence,clock['audio_shift_ms']),clock,confirmed=True)
        selected,selection=select_expression(candidates,meta,trace,settings,60000/clock['bpm'])
        originals={n.id:n for n in candidates}
        assert all((n.start,n.end,n.pitch,n.voice)==(originals[n.id].start,originals[n.id].end,
            originals[n.id].pitch,originals[n.id].voice) for n in selected)
        chosen={n.id:n for n in selected}
        for decision in selection['decisions']:
            if decision['reason']=='represented_by_chord':
                assert decision['represented_by']
                for identifier in decision['represented_by']:
                    assert identifier in chosen and chosen[identifier].voice==decision['voice']
                    assert chosen[identifier].start==decision['represented_at']
        output=assign_lanes(selected,settings,beat_ms=60000/clock['bpm'])
        report=validate(output,keys);grid=grid_report(output,clock)
        assert report['valid'] and grid['off_grid']==0 and len(output)==len(selected)
        assert all(n.voice!='vocals' for n in output)
        assert all(not n.hold or n.release_reason not in ('duration_limit','input_boundary') for n in output)
        content=export_osu(output,keys,Path('vendor/mapperatorinator/template.osu'),
            title='Sakura source expression comparison',audio='audio.mp3',bpm=clock['bpm'],offset=clock['offset'],
            version=name+' reduced chords research',grid_division=clock['division'])
        assert validate(parse_osu(content)['notes'],keys)['valid']
        (args.output/(name+'.osu')).write_text(content,encoding='utf-8')
        save(name+'-notes.json',[n.data() for n in output]);save(name+'-selection.json',selection)
        assert analysis==unchanged and sha(analysis_path)==identity['inputs'][engine]['analysis_sha256']
        row=dict(name=name,settings=settings,elapsed_seconds=time.monotonic()-begin,
            structural=report,grid=grid,alignment=alignment,selection_optimal=selection['attack_optimal'],
            source_counts=dict(Counter(n.voice for n in output)),
            source_attacks=dict(Counter(v for v,t in {(n.voice,n.start) for n in output})),
            source_holds=dict(Counter(n.voice for n in output if n.hold)),
            omission_reasons=selection['omission_reasons'],
            reduced_chord_groups=len({(d['voice'],d['represented_at']) for d in selection['decisions'] if d['reason']=='represented_by_chord'}),
            layout=layout_metrics(output,keys,60000/clock['bpm']),
            raw_evidence_unchanged=True,retained_endpoints_unchanged=True,representation_links_valid=True)
        cases.append(row);save('progress.json',cases)
        print(json.dumps(row,ensure_ascii=True),flush=True)
    shutil.copyfile(args.input/'audio.mp3',args.output/'audio.mp3')
    save('report.json',dict(identity=identity,cases=cases,
        baseline={k:{field:v[field] for field in ('structural','source_counts','source_holds','omission_reasons')}
                  for k,v in baseline['models'].items()},
        network_rerun=False,visual_check=False,musical_acceptance=False,
        notice='Same saved audio evidence and section plan. Known timing for generation. Tests capacity/lineage and source retention, not agreement with an authored chart or subjective music quality.'))


if __name__=='__main__':main()
