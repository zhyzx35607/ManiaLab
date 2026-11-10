"""Offline, fixed-protocol onset recovery from audited Tsumugi V1 filtered groups.

No neural inference. Recovered events have offset == onset and never grant LN use.
The cohort command requires all 100 completed cases before reading any labels.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_tsumugi_releases as release_audit
from scripts import evaluate_guitar_vocal_benchmark as cohort_evaluation
from scripts import evaluate_instrument_benchmark as onset_evaluation
from scripts.research_tsumugi_transcription import convert_notes

PROTOCOL_SHA256 = 'cb53f530e0ebc63d70789421589de598280f9af7561c09c5c80340f9a45c2976'
DEDUPLICATION_MS = 50
DEFAULT_SOURCE = Path('test-data/other-instrument-model-survey-2026-11-10/tsumugi')
TOLERANCES = {'standard':50,'fixed_40_80':40}
read_json, require, sha = release_audit.read_json, release_audit.require, release_audit.sha


def data_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def recover_attacks(audited, constructions, baseline, classes):
    """Pure recovery. Its caller must obtain ``audited`` by verified replay."""
    require(all(audited.get(k) is True for k in ('official_finalize_equal','saved_raw_notes_equal','window_snapshots_equal')),
            'Require verified official/raw/window replay')
    identity=audited['identity']
    sr,total=identity['resampled_sample_rate'],identity['resampled_samples']
    require(type(sr) is int and sr>0 and type(total) is int and total>0,'Invalid recorded clock')
    require(identity['model_config']['semi_crf_version']=='v1','Only V1 recovery is defined')
    raw=[item['note'] for item in audited['notes']]
    expected=convert_notes(raw,sample_rate=sr,total_samples=total,classes=classes)
    require(baseline==expected,'Saved application events differ from audited raw conversion')
    sources=audited['filtered_early_unclosed_onset_source_indices']
    require(len(sources)==len(set(sources)),'Duplicate filtered source index')
    filtered={source for group in audited['filtered_groups'] for source in group['sources']}
    expected_sources={i for i in filtered if constructions[i]['constructed_note']['has_onset']
                      and not constructions[i]['constructed_note']['has_offset']}
    require(set(sources)==expected_sources,'Recovery source list omits or adds an unclosed onset')
    require(all(type(i) is int and 0<=i<len(constructions) and i in filtered for i in sources),
            'Recovery source is not in an audited filtered group')
    for source in sources:
        row=constructions[source]
        n=row['constructed_note']
        require(n is not None and n['has_onset'] is True and n['has_offset'] is False,
                'Recovery requires an original unclosed onset')
        require(type(row['v1_slot_index']) is int and 0<=row['v1_slot_index']<identity['model_config']['num_pitch_slots'],
                'Invalid recovered slot')
    ordered=sorted(sources,key=lambda i:(constructions[i]['constructed_note']['start_sample'],
        constructions[i]['constructed_note']['pitch'],constructions[i]['v1_slot_index'],i))
    accepted,decisions=[],[]
    for source in ordered:
        row=constructions[source]
        original=copy.deepcopy(row['constructed_note'])
        note=copy.deepcopy(original)
        note['slot_index']=row['v1_slot_index']
        # Validation/conversion preserves the construction's original model class.
        event=convert_notes([note],sample_rate=sr,total_samples=total,classes=classes)[0]
        def near(other):
            return (note['pitch']==other['pitch'] and note['slot_index']==other['slot_index']
                and abs(note['start_sample']-other['start_sample'])*1000<=DEDUPLICATION_MS*sr)
        existing=[i for i,n in enumerate(raw) if near(n)]
        previous=[item['source_index'] for item in accepted if near(item['raw_note'])]
        evidence=dict(source_index=source,original_model_interval=original,
            arguments=copy.deepcopy(row['arguments']),v1_slot_index=row['v1_slot_index'],
            sample_rate=sr,original_interval_is_not_confirmed_release=True)
        if existing:
            decisions.append(dict(**evidence,decision='duplicate_existing_onset',existing_note_indices=existing))
            continue
        if previous:
            decisions.append(dict(**evidence,decision='duplicate_recovered_onset',recovered_source_indices=previous))
            continue
        event.update(offset=event['onset'],release_reason='unresolved_or_postprocessed',
            event_kind='onset_only',application_ln_eligible=False,recovery_source_index=source,
            model_has_offset_after_stitching=False)
        accepted.append(dict(source_index=source,raw_note=note,event=event))
        decisions.append(dict(**evidence,decision='recovered_onset_only'))
    events=[item['event'] for item in accepted]
    counts=Counter(d['decision'] for d in decisions)
    return dict(baseline_notes=copy.deepcopy(baseline),recovered_attacks=events,
        notes=copy.deepcopy(baseline)+events,decisions=decisions,
        counts=dict(input_candidates=len(sources),baseline_notes=len(baseline),recovered_attacks=len(events),
            duplicate_existing_onsets=counts['duplicate_existing_onset'],
            duplicate_recovered_onsets=counts['duplicate_recovered_onset'],total_notes=len(baseline)+len(events)),
        baseline_modified=False,enabled_in_application=False,application_ln_eligible=False)


def load_classes(source,identity):
    """Pinned upstream mapping without importing its model or package initializers."""
    relative='instrument_agnostic_amt/taxonomy/instrument_merge.json'
    require(sha(Path(source)/relative)==identity['source_files'][relative],'Instrument mapping changed')
    classes=sorted(read_json(Path(source)/relative)['labels'])
    for name in ('melody','vocal_harmony','wind_chimes'):
        if name not in classes:
            classes.append(name)
    require(len(classes)==identity['model_config']['num_instrument_classes'],'Instrument class count differs')
    return classes


def recover_case(input_dir,source):
    folder=Path(input_dir)
    audited=release_audit.audit(folder,source)
    classes=load_classes(source,audited['identity'])
    result=recover_attacks(audited,read_json(folder/'interval-constructions.json'),
        read_json(folder/'notes.json'),classes)
    return dict(revision=1,protocol_sha256=PROTOCOL_SHA256,runner_sha256=sha(__file__),
        auditor_sha256=sha(release_audit.__file__),verified_replay_sha256=data_sha(audited),
        input=str(folder.resolve()),identity=audited['identity'],report_sha256=audited['report_sha256'],
        evidence_sha256=audited['evidence_sha256'],audit_counts=audited['counts'],
        deduplication_ms=DEDUPLICATION_MS,**result)


def onset_metrics(reference,predicted,*,begin,end,context_start,context_end):
    """Keep zero-length attacks valid while never scoring or reporting offsets."""
    require(all(cohort_evaluation.finite(x) for x in (begin,end,context_start,context_end))
        and 0<=context_start<=begin<end<=context_end,'Invalid scoring/context clock')
    cohort_evaluation.validate_notes(reference,reference=True)
    for note in predicted:
        require(all(cohort_evaluation.finite(note.get(k)) for k in ('onset','offset','pitch','program')),
                'Nonfinite/missing prediction')
        require(note['onset']>=0 and note['offset']>=note['onset'],'Invalid prediction interval')
        require(type(note['pitch']) is int and 0<=note['pitch']<=127 and type(note['program']) is int
                and -1<=note['program']<=128 and type(note.get('is_drum')) is bool,'Invalid prediction labels')
        if note['offset']==note['onset']:
            require(note.get('event_kind')=='onset_only' and note.get('release_reason')=='unresolved_or_postprocessed'
                    and note.get('application_ln_eligible') is False,'Zero duration requires explicit onset-only semantics')
    ri=[i for i,n in enumerate(reference) if begin<=n['onset']<end]
    pi=[i for i,n in enumerate(predicted) if begin<=n['onset']<end]
    refs=[reference[i] for i in ri]
    preds=[predicted[i] for i in pi]
    metrics={}
    for name in TOLERANCES:
        value=onset_evaluation.match(refs,preds,offset=False,strict=name=='fixed_40_80')
        # The frozen matcher computes an offset diagnostic even when matching
        # onsets. It must not escape into this onset-only evaluation.
        value.pop('mean_release_error_ms')
        value.update(true_positives=value['matched'],false_positives=value['predicted']-value['matched'],
                     false_negatives=value['reference']-value['matched'])
        metrics[name]=value
    return metrics,dict(reference_indices=ri,predicted_indices=pi,reference_total=len(reference),
        predicted_total=len(predicted),reference=len(refs),predicted=len(preds),
        predicted_drums=sum(n['is_drum'] for n in preds),
        onset_only_predictions=sum(n.get('event_kind')=='onset_only' for n in preds),
        begin_seconds=begin,end_seconds=end,context_start_seconds=context_start,context_end_seconds=context_end,
        clock='original_audio_seconds',all_classes_retained=True,interval_scoring=False)


def aggregate_onsets(rows):
    result={}
    for name in TOLERANCES:
        values=[r['metrics'][name] for r in rows]
        nref=sum(v['reference'] for v in values)
        npred=sum(v['predicted'] for v in values)
        matched=sum(v['matched'] for v in values)
        result[name]=dict(total_cases=len(rows),full_cohort_available=True,
            micro=dict(reference=nref,predicted=npred,matched=matched,true_positives=matched,
                false_positives=npred-matched,false_negatives=nref-matched,
                precision=matched/npred if npred else 0.,recall=matched/nref if nref else 0.,
                f1=2*matched/(nref+npred) if nref+npred else 0.),
            macro={k:sum(v[k] for v in values)/len(values) if values else 0. for k in ('precision','recall','f1')})
    return result


def completed_inputs(root,predictions):
    """Preflight the full completion chain before touching reference note files."""
    root,predictions=Path(root),Path(predictions)
    completed_path=predictions/'completed.json'
    require(completed_path.is_file(),'All 100 Tsumugi cases must finish before cohort scoring')
    manifest=read_json(root/'manifest.json')
    cases=cohort_evaluation.verify_manifest(manifest)
    plan=read_json(predictions/'plan.json')
    completed=read_json(completed_path)
    comparison=read_json(predictions/'comparison.json')
    require(completed.get('cases')==100 and comparison.get('cases')==100
        and comparison.get('annotations')==140,'Incomplete cohort denominator')
    require(completed.get('identity')==plan==comparison.get('identity'),'Completion identity differs')
    cohort_evaluation.verified_hash(root/'manifest.json',plan.get('manifest_sha256'))
    cohort_evaluation.verified_hash(predictions/'plan.json',completed.get('plan_sha256'))
    cohort_evaluation.verified_hash(predictions/'comparison.json',completed.get('comparison_sha256'))
    cohort_evaluation.verified_hash('scripts/run_tsumugi_guitar_vocal_comparison.py',plan.get('runner_sha256'))
    cohort_evaluation.verified_hash('scripts/research_tsumugi_transcription.py',plan.get('adapter_sha256'))
    cohort_evaluation.verified_hash(cohort_evaluation.__file__,plan.get('evaluator_sha256'))
    cohort_evaluation.verified_hash(onset_evaluation.__file__,plan.get('matcher_sha256'))
    require(plan.get('stem_filter')=='all' and plan.get('normalization') is False
        and plan.get('inference_scope')=='full_original_audio','Original input/class protocol differs')
    rows=comparison.get('rows',[])
    names=[c['name'] for c in cases]
    require(len(rows)==100 and [r['name'] for r in rows]==names,'Comparison must contain every frozen case in order')
    for case,row in zip(cases,rows):
        path=predictions/case['name']
        cohort_evaluation.verified_hash(path/'report.json',row.get('report_sha256'))
        cohort_evaluation.verified_hash(path/'notes.json',row.get('notes_sha256'))
        require(set(row.get('references',{}))==set(case['references']),'Missing annotation in completed comparison')
    return cases,plan,comparison


def evaluate_cohort(root,predictions,source):
    cases,plan,comparison=completed_inputs(root,predictions)
    groups={engine:{name:[] for name in ('GuitarSet/strings','vocadito/A1','vocadito/A2')}
            for engine in ('baseline','recovered')}
    rows=[]
    for case,original in zip(cases,comparison['rows']):
        result=recover_case(Path(predictions)/case['name'],source)
        identity=result['identity']
        model=plan['selection'][case['dataset']]
        expected=dict(source_commit=plan['source_commit'],source_files=plan['source_files'],
            runner_sha256=plan['adapter_sha256'],model_revision=plan['model_revision'],model=model,
            model_sha256=plan['models'][model],audio=str(Path(case['audio']).resolve()),audio_sha256=case['audio_sha256'],
            original_sample_rate=case['sample_rate'],original_samples=case['samples'],original_channels=1,
            resampled_sample_rate=22050,resampled_samples=(case['samples']*22050+case['sample_rate']-1)//case['sample_rate'],
            normalization=False,inference_scope='full_original_audio',stem_filter='all',evidence_revision=2)
        require(all(identity.get(k)==v for k,v in expected.items()),'Recovery input differs from frozen cohort: '+case['name'])
        require(case['start_ms']==0 and case['duration_seconds']==case['samples']/case['sample_rate']
                and abs(case['seconds']-min(20.,case['duration_seconds']-2.))<=1e-10,'Frozen scoring clock differs')
        references={}
        for annotator,entry in case['references'].items():
            cohort_evaluation.verified_hash(entry['path'],entry['sha256'])
            reference=read_json(entry['path'])
            scores={}
            for engine,predicted in (('baseline',result['baseline_notes']),('recovered',result['notes'])):
                metrics,scope=onset_metrics(reference,predicted,begin=0,end=case['seconds'],
                    context_start=0,context_end=case['duration_seconds'])
                require(entry['count']==len(reference) and entry['scored_count']==scope['reference'],'Label denominator differs')
                scores[engine]=dict(metrics=metrics,scope=scope)
                groups[engine][case['dataset']+'/'+annotator].append(dict(name=case['name'],metrics=metrics))
            for tolerance in TOLERANCES:
                frozen=original['references'][annotator]['tsumugi']['metrics'][tolerance]['pitched_onsets']
                require(all(scores['baseline']['metrics'][tolerance][key]==frozen[key]
                    for key in ('reference','predicted','matched','precision','recall','f1','mean_onset_error_ms')),
                    'Baseline onset scores differ from completed frozen comparison')
            references[annotator]=dict(labels_sha256=entry['sha256'],**scores)
        rows.append(dict(name=case['name'],dataset=case['dataset'],recovery=result,references=references))
    return dict(revision=1,complete=True,cases=100,annotations=140,protocol_sha256=PROTOCOL_SHA256,
        runner_sha256=sha(__file__),auditor_sha256=sha(release_audit.__file__),matcher_sha256=sha(onset_evaluation.__file__),
        manifest_sha256=sha(Path(root)/'manifest.json'),plan_sha256=sha(Path(predictions)/'plan.json'),
        completed_sha256=sha(Path(predictions)/'completed.json'),comparison_sha256=sha(Path(predictions)/'comparison.json'),
        tolerances_ms=TOLERANCES,interval_scoring=False,enabled_in_application=False,rows=rows,
        groups={engine:{name:aggregate_onsets(items) for name,items in value.items()} for engine,value in groups.items()},
        limits=['Unknown releases are zero-length onset events, never complete intervals or validated LN releases.',
                'All predicted classes remain in denominators; drum predictions cannot match pitched references.',
                'A1 and A2 are independent full groups; no favorable annotator selection.',
                'Recovery is a fixed 50ms deduplication ablation, not proof that every recovered attack is audible.'])


def write_new(path,result):
    path=Path(path)
    require(not path.exists(),'Preserve earlier output; choose a new output file')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest='mode',required=True)
    single=sub.add_parser('single',help='Recover one complete audited case without labels')
    single.add_argument('--input',type=Path,required=True)
    cohort=sub.add_parser('cohort',help='Score all completed 100 cases; reject partial experiments')
    cohort.add_argument('--root',type=Path,default=Path('test-data/guitar-vocal-benchmark-v3'))
    cohort.add_argument('--predictions',type=Path,default=Path('test-data/tsumugi-guitar-vocal-comparison-v1'))
    for command in (single,cohort):
        command.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
        command.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    require(not args.output.exists(),'Preserve earlier output; choose a new output file')
    result=(recover_case(args.input,args.source) if args.mode=='single'
            else evaluate_cohort(args.root,args.predictions,args.source))
    write_new(args.output,result)
    print(json.dumps(result['counts'] if args.mode=='single' else result['groups']),flush=True)


if __name__=='__main__':
    main()
