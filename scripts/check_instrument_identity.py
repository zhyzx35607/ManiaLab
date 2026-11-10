"""Replay frozen accompaniment with/without preserved model instrument IDs."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.instrument_candidate_adapter import accompaniment_candidates
from manialab.arrange import parameters,assign_lanes
from manialab.domain import UNOBSERVED_RELEASES,export_osu,parse_osu,validate
from manialab.expression import expression_evidence,select_expression,signature
from manialab.timing import chart_clock_notes,align_candidates,grid_report


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();output=args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new research output')
    baseline=ROOT/'test-data/sakura-benchmark/instrument-expression-v1'
    model=ROOT/'test-data/sakura-benchmark/other-float-yourmt3-v1'
    audit=read(model/'release-audit.json');report=read(model/'report.json')
    for name,suffix in [('notes','.json'),('tokens','.npz')]:
        digest=sha256(model/(name+suffix))
        if digest!=audit[name+'_sha256'] or digest!=report[name+'_sha256']:
            raise ValueError('Original predictions changed')
    if sha256(model/'release-provenance.json')!=audit['provenance_sha256']:
        raise ValueError('Endpoint evidence changed')
    original=read(baseline/'instrument-analysis.json')
    model_notes,_=accompaniment_candidates(read(model/'notes.json'),read(model/'release-provenance.json'),original['duration']*1000)
    ids={n.id:n.instrument_id for n in model_notes}
    if {n['id'] for n in original['candidates'] if n['voice']=='other'}!=set(ids):
        raise ValueError('Frozen candidate selection differs')
    config=read(baseline/'report.json')['identity']
    clock=config['generation_clock'];settings=parameters(config['settings'] if 'settings' in config else
        read(ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/report.json')['identity']['settings'])
    frozen_plan=read(baseline/'instrument-other-10k-plan.json')
    output.mkdir(parents=True)
    paths=[Path(__file__),ROOT/'scripts/instrument_candidate_adapter.py',
           *[ROOT/'manialab'/name for name in ('domain.py','expression.py','expression_articulation.py','arrange.py','layout.py','timing.py','recommendations.py')]]
    for path in paths:
        (output/(path.stem+'-snapshot.py')).write_bytes(path.read_bytes())
    identity=dict(inputs={str(p.relative_to(ROOT)):sha256(p) for p in [
        baseline/'instrument-analysis.json',baseline/'report.json',baseline/'instrument-other-10k-plan.json',
        baseline/'instrument-other-10k-notes.json',model/'notes.json',model/'release-provenance.json']},
        code={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        clock=clock,settings=settings,new_audio_inference=False,reference_chart_read=False,
        changed_field='instrument_id only; identical frozen section policies')
    save(output/'identity.json',identity)
    cases=[];flat_notes={}
    for enabled,keys in [(False,10),*[(True,k) for k in range(6,11)]]:
        started=time.monotonic();name=('identified' if enabled else 'flat')+f'-{keys}k'
        analysis=copy.deepcopy(original)
        for n in analysis['candidates']:
            if n['voice']=='other':n['instrument_id']=ids[n['id']] if enabled else ''
        plan=copy.deepcopy(frozen_plan);plan['evidence_signature']=signature(analysis)
        p=parameters(settings|{'keys':keys})
        evidence,meta,trace=expression_evidence(analysis,plan,p)
        aligned,alignment=align_candidates(chart_clock_notes(evidence,clock['audio_shift_ms']),clock,confirmed=True)
        selected,selection=select_expression(aligned,meta,trace,p,60000/clock['bpm'])
        originals={n.id:n for n in aligned}
        if any((n.start,n.end,n.instrument_id)!=(originals[n.id].start,originals[n.id].end,originals[n.id].instrument_id) for n in selected):
            raise ValueError('Selection changed candidate timing or identity')
        if not enabled:
            prior=read(baseline/'instrument-other-10k-notes.json')
            if sorted((n.id,n.start,n.end) for n in selected)!=sorted((n['id'],n['start'],n['end']) for n in prior):
                raise ValueError('Flat control does not reproduce archived selected events')
        placed=assign_lanes(selected,p,beat_ms=60000/clock['bpm'])
        if any((n.start,n.end,n.instrument_id)!=(originals[n.id].start,originals[n.id].end,originals[n.id].instrument_id) for n in placed):
            raise ValueError('Layout changed timing or instrument identity')
        if any(n.hold and n.release_reason in UNOBSERVED_RELEASES for n in placed):
            raise ValueError('Unobserved release became LN')
        structural=validate(placed,keys);grid=grid_report(placed,clock)
        if not structural['valid'] or grid['off_grid']:
            raise ValueError('Invalid chart')
        content=export_osu(placed,keys,ROOT/'vendor/mapperatorinator/template.osu',
            title='Instrument identity regression',audio='audio.mp3',version=name+' research',
            bpm=clock['bpm'],offset=clock['offset'],grid_division=clock['division'])
        reread=parse_osu(content)['notes']
        if (not validate(reread,keys)['valid'] or
            sorted((n.start,n.end,n.lane) for n in reread)!=sorted((n.start,n.end,n.lane) for n in placed)):
            raise ValueError('Export round trip failed')
        (output/(name+'.osu')).write_text(content,encoding='utf-8')
        save(output/(name+'-notes.json'),[n.data() for n in placed])
        save(output/(name+'-selection.json'),selection)
        if not enabled:flat_notes={n.id:n for n in placed}
        current={n.id:n for n in placed}
        comparison=None
        if enabled and keys==10:
            comparison=dict(
                added_by_voice=dict(Counter(current[i].voice for i in current.keys()-flat_notes.keys())),
                removed_by_voice=dict(Counter(flat_notes[i].voice for i in flat_notes.keys()-current.keys())),
                shared_timing_changed=sum((current[i].start,current[i].end)!=(flat_notes[i].start,flat_notes[i].end)
                    for i in current.keys()&flat_notes.keys()),
                notice='Identity grouping also changes capacity competition; more notes do not establish better musical selection.')
        summary=dict(name=name,keys=keys,seconds=time.monotonic()-started,
            notes=len(placed),ln=sum(n.hold for n in placed),
            source_counts=dict(Counter(n.voice for n in placed)),
            instrument_counts=dict(Counter(n.instrument_id for n in placed if n.voice=='other')),
            omitted=selection['omission_reasons'],alignment=alignment,
            duplicates_by_voice=dict(Counter(r['voice'] for r in selection['decisions'] if r.get('reason')=='duplicate_pitch')),
            flat_control_comparison=comparison,
            large_shifts=selection['selected_large_attack_shifts'],optimal=selection['attack_optimal'],
            valid=structural['valid'],grid_off=grid['off_grid'])
        cases.append(summary);save(output/'progress.json',cases)
        print(json.dumps(summary),flush=True)
    save(output/'report.json',dict(identity_sha256=sha256(output/'identity.json'),cases=cases,
        complete=True,model_training=False,new_audio_inference=False,musical_acceptance=False,
        direct_listening_performed=False,
        artifacts_sha256={p.name:sha256(p) for p in output.iterdir()}))


if __name__=='__main__':main()
