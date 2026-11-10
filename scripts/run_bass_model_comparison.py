"""Serial local-only bass comparison: all 17 application cases, then 17 Tsumugi.

Nothing loads a model at import or --help time. The run command is deliberately
explicit; application-stage is a serial child used only after the plan freezes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from math import gcd
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_guitar_vocal_benchmark as evaluation
from scripts.prepare_bass_benchmark import PLUCKING,EXPRESSION
from scripts.research_tsumugi_transcription import SOURCE_COMMIT,MODEL_REVISION,MODELS,verify_source

read_json,require,sha,verified_hash=evaluation.read_json,evaluation.require,evaluation.sha,evaluation.verified_hash
MANIFEST_SHA256='5e575417b05674ba23a0d1614643197a97ace1a0d1d533fbdd689987fed3d46f'
APP_FILES=('manialab/audio.py','manialab/articulation.py','manialab/domain.py','manialab/timing.py')
APP_PACKAGES=('numpy','scipy','onnxruntime','imageio-ffmpeg')
APP_ARTIFACTS={'notes.json':'notes_sha256','activations.npz':'activations_sha256',
    'candidates.json':'candidates_sha256','articulation-suppressed.json':'articulation_suppressed_sha256'}
TSUMUGI_ARTIFACTS={'notes.json':'notes_sha256','raw-notes.json':'raw_notes_sha256',
    'window-notes.json':'window_notes_sha256','interval-constructions.json':'interval_constructions_sha256'}
TSUMUGI_SETTINGS=dict(window_ms=8000,stride_ms=4000,track_batch_size=352,window_batch_size=1,
    merge_gap_ms=None,merge_onset_ms=50.,silence_gate_rms_dbfs=-72.,note_bias=0.,disable_tqdm=True,
    use_boundary_head=True,instrument_probability_mode='sigmoid',semi_crf_backend='torch',
    semi_crf_sparse_decode=False,semi_crf_sparse_topk_per_start=16,semi_crf_sparse_score_threshold=None,
    semi_crf_sparse_max_span_frames=None,instrument_pair_infer_topk=256,instrument_pair_gate_threshold=-3.,
    instrument_pair_max_pairs=512,allowed_instrument_ids=None)


def write_new(path,value):
    path=Path(path)
    require(not path.exists(),'Preserve existing evidence: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def app_versions():
    return dict(python=sys.version,packages={name:importlib.metadata.version(name) for name in APP_PACKAGES})


def validate_manifest(manifest):
    require(manifest.get('revision')==1,'Unsupported bass manifest')
    cases=manifest.get('cases',[])
    require(len(cases)==17 and [c.get('name') for c in cases]==[f'{i:03}' for i in range(1,18)],
            'Require all 17 bass cases in frozen order')
    for case in cases:
        require(case.get('sample_rate')==44100 and type(case.get('samples')) is int and case['samples']>88200,
                'Invalid original bass clock')
        duration=case['samples']/case['sample_rate']
        require(case.get('duration_seconds')==duration and duration<=60
                and case.get('seconds')==min(20.,duration-2.),'Changed bass scoring clock')
    return cases


def validate_predictions(notes):
    require(isinstance(notes,list),'Predictions must be a list')
    for n in notes:
        require(all(evaluation.finite(n.get(k)) for k in ('onset','offset','pitch','program'))
            and 0<=n['onset']<=n['offset'],'Invalid prediction interval')
        require(type(n['pitch']) is int and 0<=n['pitch']<=127 and type(n['program']) is int
            and -1<=n['program']<=128 and type(n.get('is_drum')) is bool,'Invalid prediction labels')


def global_matches(reference,predicted,*,strict=False,pitched=True,offset=False):
    """One assignment for the complete scoring window, never per technique."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not reference or not predicted:
        return []
    ref=np.asarray([[n['onset'],n['offset'],n['pitch']] for n in reference])
    pred=np.asarray([[n['onset'],n['offset'],n['pitch']] for n in predicted])
    delta=np.abs(ref[:,None,:]-pred[None,:,:])
    allowed=delta[:,:,0]<=(.04 if strict else .05)
    if pitched:
        allowed&=(delta[:,:,2]==0)&np.asarray([not n['is_drum'] for n in predicted])[None,:]
    if offset:
        tolerance=np.full(len(ref),.08) if strict else np.maximum(.05,.2*(ref[:,1]-ref[:,0]))
        allowed&=(delta[:,:,1]<=tolerance[:,None])&np.asarray([n['offset']>n['onset'] for n in predicted])[None,:]
    costs=np.concatenate((np.where(allowed,delta[:,:,0]/(len(ref)+1),1e6),np.ones((len(ref),len(ref)))),axis=1)
    rows,cols=linear_sum_assignment(costs)
    return [[int(i),int(j)] for i,j in zip(rows,cols) if j<len(pred) and allowed[i,j]]


def match_counts(reference,predicted,pairs):
    nr,npred,matched=len(reference),len(predicted),len(pairs)
    return dict(reference=nr,predicted=npred,matched=matched,true_positives=matched,
        false_positives=npred-matched,false_negatives=nr-matched,
        precision=matched/npred if npred else 0.,recall=matched/nr if nr else 0.,
        f1=2*matched/(nr+npred) if nr+npred else 0.,
        mean_onset_error_ms=sum(abs(reference[i]['onset']-predicted[j]['onset'])*1000 for i,j in pairs)/matched if matched else None)


def technique_recall(reference,pairs):
    matched={i for i,_ in pairs}
    result={}
    for field,allowed in (('plucking_style',PLUCKING),('expression_style',EXPRESSION)):
        require(all(n.get(field) in allowed for n in reference),'Invalid bass technique label')
        result[field]={}
        for value in sorted(allowed):
            selected=[i for i,n in enumerate(reference) if n[field]==value]
            hits=sum(i in matched for i in selected)
            result[field][value]=dict(reference=len(selected),matched=hits,
                false_negatives=len(selected)-hits,recall=hits/len(selected) if selected else None)
    return result


def score_case(reference,predicted,*,end,context_end):
    evaluation.validate_notes(reference,reference=True)
    validate_predictions(predicted)
    # Frozen evaluator receives real positive intervals only. Zero-length
    # candidates remain in every prediction denominator and can match onsets.
    positive=[n for n in predicted if n['offset']>n['onset']]
    metrics,scope=evaluation.window_metrics(reference,positive,begin=0,end=end,
        context_start=0,context_end=context_end)
    ri=[i for i,n in enumerate(reference) if 0<=n['onset']<end]
    pi=[i for i,n in enumerate(predicted) if 0<=n['onset']<end]
    refs,preds=[reference[i] for i in ri],[predicted[i] for i in pi]
    scope.update(predicted_indices=pi,predicted_total=len(predicted),predicted=len(preds),
        predicted_drums=sum(n['is_drum'] for n in preds),
        predicted_zero_length=sum(n['offset']==n['onset'] for n in preds),
        predicted_crosses_scoring_end=sum(n['offset']>=end for n in preds),
        predicted_crosses_context_end=sum(n['offset']>=context_end for n in preds))
    matching,agnostic,techniques={},{},{}
    for tolerance,strict in (('standard',False),('fixed_40_80',True)):
        pitched=global_matches(refs,preds,strict=strict)
        rhythm=global_matches(refs,preds,strict=strict,pitched=False)
        metrics[tolerance]['pitched_onsets']=dict(available=True,**match_counts(refs,preds,pitched))
        intervals=metrics[tolerance]['complete_intervals']
        interval_pairs=None
        if intervals['available']:
            interval_pairs=global_matches(refs,preds,strict=strict,offset=True)
            require(len(interval_pairs)==intervals['matched'],'Interval matcher parity failed')
            intervals.update(predicted=len(preds),precision=intervals['matched']/len(preds) if preds else 0.,
                f1=2*intervals['matched']/(len(refs)+len(preds)) if refs or preds else 0.,
                true_positives=intervals['matched'],false_positives=len(preds)-intervals['matched'],
                false_negatives=len(refs)-intervals['matched'])
        else:
            intervals['predicted']=len(preds)
        agnostic[tolerance]=match_counts(refs,preds,rhythm)
        def mapped(pairs):
            return None if pairs is None else [[ri[i],pi[j]] for i,j in pairs]
        matching[tolerance]=dict(nominal_pitched_onsets=mapped(pitched),pitch_agnostic_onsets=mapped(rhythm),
            nominal_complete_intervals=mapped(interval_pairs),indices='original reference/prediction arrays')
        techniques[tolerance]=dict(nominal_pitched_onsets=technique_recall(refs,pitched),
                                   pitch_agnostic_onsets=technique_recall(refs,rhythm))
    return dict(metrics=metrics,interval_scope=scope,pitch_agnostic=agnostic,
        matched_indices=matching,technique_recall=techniques)


def aggregate_extra(rows):
    output={}
    for tolerance in ('standard','fixed_40_80'):
        values=[row['pitch_agnostic'][tolerance] for row in rows]
        nr=sum(v['reference'] for v in values)
        npred=sum(v['predicted'] for v in values)
        hits=sum(v['matched'] for v in values)
        techniques={}
        for mode in ('nominal_pitched_onsets','pitch_agnostic_onsets'):
            techniques[mode]={}
            for field,styles in (('plucking_style',PLUCKING),('expression_style',EXPRESSION)):
                techniques[mode][field]={}
                for style in sorted(styles):
                    cells=[r['technique_recall'][tolerance][mode][field][style] for r in rows]
                    count=sum(c['reference'] for c in cells)
                    matched=sum(c['matched'] for c in cells)
                    techniques[mode][field][style]=dict(reference=count,matched=matched,
                        false_negatives=count-matched,recall=matched/count if count else None)
        output[tolerance]=dict(cases=len(rows),pitch_agnostic=dict(reference=nr,predicted=npred,matched=hits,
            true_positives=hits,false_positives=npred-hits,false_negatives=nr-hits,
            precision=hits/npred if npred else 0.,recall=hits/nr if nr else 0.,
            f1=2*hits/(nr+npred) if nr+npred else 0.),technique_recall=techniques)
    return output


def verify_output(folder,expected,artifacts):
    folder=Path(folder)
    report=read_json(folder/'report.json')
    require(report.get('identity')==expected,'Existing output identity differs: '+str(folder))
    for filename,key in artifacts.items():
        verified_hash(folder/filename,report.get(key))
    notes=read_json(folder/'notes.json')
    validate_predictions(notes)
    require(report.get('notes')==len(notes),'Output count differs')
    require(report.get('context_start_seconds')==0 and
        report.get('context_end_seconds')==expected['original_samples']/expected['original_sample_rate'],'Output clock differs')
    return report,notes


def verify_application(folder,plan,case):
    report=read_json(Path(folder)/'report.json')
    expected=dict(plan['application'],manifest_sha256=plan['manifest_sha256'],runner_sha256=plan['runner_sha256'],
        audio=str(Path(case['audio']).resolve()),audio_sha256=case['audio_sha256'],original_sample_rate=case['sample_rate'],
        original_samples=case['samples'],original_channels=1,
        resampled_samples=(case['samples']*22050+case['sample_rate']-1)//case['sample_rate'])
    require(all(report['identity'].get(k)==v for k,v in expected.items()),'Application source/runtime/settings identity differs')
    require(all(evaluation.digest(report['identity'].get(k)) for k in ('pcm_sha256','resampled_pcm_sha256')),
            'Missing PCM evidence identity')
    require(report.get('resampled_duration_seconds')==expected['resampled_samples']/22050,'Application resampled clock differs')
    return verify_output(folder,report['identity'],APP_ARTIFACTS)


def application_pipeline(audio,steady,signal,frames,onsets,times,settings):
    """Call the actual application bass sequence; injected modules allow pure tests."""
    attacks,_=audio.rhythmic_evidence(signal,estimate=False)
    raw_attacks=[n.data() for n in attacks]
    attacks,rejected_attacks=audio.source_evidence(attacks,signal,signal)
    attacks=[replace(n,voice='bass',id='bass:'+n.id) for n in attacks]
    suppressed=steady(frames,onsets,times,signal,audio.SR,settings['onset_threshold'],settings['frame_threshold'])
    args=(frames,onsets,times,len(signal)/audio.SR,settings['onset_threshold'],settings['frame_threshold'],attacks)
    unsuppressed=audio.pitched_notes(*args,suppressed=())
    pitched=audio.pitched_notes(*args,suppressed=suppressed)
    before_source=[n.data() for n in pitched]
    pitched,rejected_pitch=audio.source_evidence(pitched,signal,signal)
    before_release=[n.data() for n in pitched]
    pitched,releases=audio.refine_releases(pitched,signal)
    before_fusion=[n.data() for n in pitched]
    pitched,unmatched,fused=audio.fuse_attacks(pitched,attacks)
    pitched=[replace(n,voice='bass',id='bass:'+n.id) for n in pitched]
    evidence=dict(raw_attacks=raw_attacks,filtered_attacks=[n.data() for n in attacks],
        pitched_without_articulation_suppression=[n.data() for n in unsuppressed],
        pitched_before_source_filter=before_source,pitched_before_release_refinement=before_release,
        pitched_before_attack_fusion=before_fusion,final_candidates=[n.data() for n in pitched],
        unmatched_attacks=[n.data() for n in unmatched],rejected_low_energy_attacks=rejected_attacks,
        rejected_low_energy_pitch=rejected_pitch,release_refinement=releases,fused_attacks=fused)
    events=[dict(onset=n.start/1000,offset=n.end/1000,pitch=n.pitch,program=-1,is_drum=False,
        strength=n.strength,release_reason=n.release_reason,voice='bass') for n in pitched]
    validate_predictions(events)
    return events,evidence,suppressed


def current_app_identity(plan,case,pcm,signal):
    return dict(plan['application'],manifest_sha256=plan['manifest_sha256'],runner_sha256=plan['runner_sha256'],
        audio=str(Path(case['audio']).resolve()),audio_sha256=case['audio_sha256'],
        original_sample_rate=case['sample_rate'],original_samples=case['samples'],original_channels=1,
        pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),resampled_samples=len(signal),
        resampled_pcm_sha256=hashlib.sha256(signal.astype('<f4').tobytes()).hexdigest())


def application_stage(root,output):
    import numpy as np
    import onnxruntime as ort
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from manialab import audio
    from manialab.articulation import steady_bass_onsets
    root,output=Path(root),Path(output)
    plan=read_json(output/'plan.json')
    verified_hash(root/'manifest.json',plan['manifest_sha256'])
    require(sha(__file__)==plan['runner_sha256'] and app_versions()==plan['application']['versions'],'Application runner/runtime changed')
    for path,digest in plan['application']['source_files'].items(): verified_hash(path,digest)
    verified_hash(plan['application']['model'],plan['application']['model_sha256'])
    cases=validate_manifest(read_json(root/'manifest.json'))
    outputs={}
    session=None
    for index,case in enumerate(cases):
        verified_hash(case['audio'],case['audio_sha256'])
        rate,wave=wavfile.read(case['audio'])
        require(rate==case['sample_rate'] and len(wave)==case['samples'] and wave.ndim==1 and wave.dtype==np.int16,
                'Original PCM format/length differs')
        pcm=wave.astype(np.float32)/32768
        common=gcd(rate,audio.SR)
        signal=resample_poly(pcm,audio.SR//common,rate//common).astype(np.float32)
        require(len(signal)==(len(pcm)*audio.SR+rate-1)//rate and np.isfinite(signal).all(),'Resampled clock differs')
        expected=current_app_identity(plan,case,pcm,signal)
        folder=output/'application'/case['name']
        if not (folder/'report.json').exists():
            require(not folder.exists(),'Inspect partial application case before retry: '+case['name'])
            folder.mkdir(parents=True)
            if session is None:
                options=ort.SessionOptions()
                options.intra_op_num_threads=2
                options.inter_op_num_threads=1
                session=ort.InferenceSession(plan['application']['model'],sess_options=options,providers=['CPUExecutionProvider'])
                audio._session=session
            started=time.monotonic()
            frames,onsets,times=audio.infer(signal,Path(plan['application']['model']))
            np.savez_compressed(folder/'activations.npz',frames=frames,onsets=onsets,times=times)
            events,evidence,suppressed=application_pipeline(audio,steady_bass_onsets,signal,frames,onsets,times,plan['application']['settings'])
            write_new(folder/'notes.json',events)
            write_new(folder/'candidates.json',evidence)
            write_new(folder/'articulation-suppressed.json',suppressed)
            report=dict(identity=expected,notes=len(events),elapsed_seconds=time.monotonic()-started,
                context_start_seconds=0.,context_end_seconds=len(wave)/rate,
                resampled_duration_seconds=len(signal)/audio.SR,
                release_reasons=dict(Counter(n['release_reason'] for n in events)),
                **{key:sha(folder/filename) for filename,key in APP_ARTIFACTS.items()})
            write_new(folder/'report.json',report)
        verify_output(folder,expected,APP_ARTIFACTS)
        outputs[case['name']]=dict(report_sha256=sha(folder/'report.json'))
        print(json.dumps(dict(stage='application',index=index+1,total=17,case=case['name'])),flush=True)
    completion=dict(identity=plan['application'],plan_sha256=sha(output/'plan.json'),cases=17,case_reports=outputs)
    marker=output/'application'/'completed.json'
    if marker.exists(): require(read_json(marker)==completion,'Application completion changed')
    else: write_new(marker,completion)


def expected_tsumugi(plan,case):
    frozen=plan['tsumugi']
    return dict(source_commit=SOURCE_COMMIT,source_files=frozen['source_files'],runner_sha256=frozen['adapter_sha256'],
        model_revision=MODEL_REVISION,model='bass_v2',model_sha256=MODELS['bass_v2'],audio=str(Path(case['audio']).resolve()),
        audio_sha256=case['audio_sha256'],original_sample_rate=case['sample_rate'],original_samples=case['samples'],original_channels=1,
        resampled_sample_rate=22050,resampled_samples=(case['samples']*22050+case['sample_rate']-1)//case['sample_rate'],
        resampler='torchaudio.functional.resample/default',normalization=False,mono_policy='duplicate_to_stereo',
        inference_scope='full_original_audio',intra_threads=2,inter_threads=1,amp=False,compile=False,seed=42,
        stem_filter='all',versions=frozen['versions'],settings=TSUMUGI_SETTINGS,evidence_revision=2)


def verify_tsumugi(folder,plan,case):
    report=read_json(Path(folder)/'report.json')
    expected=expected_tsumugi(plan,case)
    require(all(report['identity'].get(k)==v for k,v in expected.items()),'Tsumugi source/runtime/settings identity differs')
    require(report['identity'].get('model_config',{}).get('semi_crf_version')=='v1'
        and report['identity'].get('settings',{}).get('allowed_instrument_ids') is None,'Tsumugi must decode all V1 classes')
    require(read_json(Path(folder)/'identity.json')==report['identity'],'Tsumugi report/identity mismatch')
    return verify_output(folder,report['identity'],TSUMUGI_ARTIFACTS)


def collect_scores(root,output,plan,cases):
    require([c['name'] for c in cases]==[f'{i:03}' for i in range(1,18)],'Never score a partial bass cohort')
    for case in cases:
        require(all((Path(output)/engine/case['name']/'report.json').is_file()
                    for engine in ('application','tsumugi')),'Both complete 17-case stages are required before scoring')
    completed=read_json(Path(output)/'application/completed.json')
    require(completed.get('cases')==17 and completed.get('identity')==plan['application']
        and set(completed.get('case_reports',{}))=={c['name'] for c in cases},'Application completion is incomplete')
    verified_hash(Path(output)/'plan.json',completed['plan_sha256'])
    rows=[]
    for case in cases:
        verified_hash(Path(output)/'application'/case['name']/'report.json',
                      completed['case_reports'][case['name']]['report_sha256'])
        reference=read_json(case['labels'])
        verified_hash(case['labels'],case['labels_sha256'])
        require(len(reference)==case['reference_count'] and sum(n['onset']<case['seconds'] for n in reference)==case['scoring_reference_count'],
                'Frozen reference denominator changed')
        require(sum(n['onset']<case['seconds'] and n['offset']>=case['duration_seconds'] for n in reference)
                ==case['scoring_unobserved_tails'],'Frozen unobservable reference count differs')
        engines={}
        for engine in ('application','tsumugi'):
            folder=Path(output)/engine/case['name']
            report=read_json(folder/'report.json')
            if engine=='tsumugi': report,notes=verify_tsumugi(folder,plan,case)
            else: report,notes=verify_application(folder,plan,case)
            engines[engine]=dict(report_sha256=sha(folder/'report.json'),notes_sha256=sha(folder/'notes.json'),
                **score_case(reference,notes,end=case['seconds'],context_end=case['duration_seconds']))
        rows.append(dict(name=case['name'],labels_sha256=case['labels_sha256'],engines=engines))
    require(len(rows)==17,'Never aggregate a partial bass cohort')
    groups={engine:dict(nominal_pitched=evaluation.aggregate([dict(name=r['name'],metrics=r['engines'][engine]['metrics']) for r in rows]),
        **aggregate_extra([r['engines'][engine] for r in rows])) for engine in ('application','tsumugi')}
    return dict(revision=1,identity=plan,cases=17,rows=rows,groups=groups,enabled_in_application=False,
        local_evaluation_only=True,license='CC-BY-NC-ND-4.0',limitations=[
            'Nominal pitch for dead notes, harmonics and slides can be ambiguous; pitch-agnostic results are reported separately.',
            'Technique recall uses each complete-window assignment; unmatched predictions have no fabricated technique attribution or subgroup F1.',
            'Zero-length predictions cannot match full intervals but remain in all prediction denominators.',
            'All model classes and boundary predictions are retained; no physical interval establishes authored LN quality.',
            'Small isolated-source public corpus with possible upstream training overlap; local evaluation only.'])


def run(args):
    root,output=args.root.resolve(),args.output.resolve()
    verified_hash(root/'manifest.json',MANIFEST_SHA256)
    cases=validate_manifest(read_json(root/'manifest.json'))
    require(not (output/'completed.json').exists(),'Preserve completed bass comparison')
    for case in cases: verified_hash(case['audio'],case['audio_sha256'])
    sources=verify_source(args.source)
    verified_hash(args.source/'checkpoints/best_model_bass_v2.pth',MODELS['bass_v2'])
    app_runtime=json.loads(subprocess.check_output([str(args.application_python),str(Path(__file__).resolve()),'application-runtime'],text=True))
    model_runtime=json.loads(subprocess.check_output([str(args.tsumugi_python),'-c',
        'import sys,json;sys.path.insert(0,sys.argv[1]);from scripts.research_tsumugi_transcription import runtime_versions;print(json.dumps(runtime_versions()))',
        str(args.dependencies.resolve())],text=True))
    plan=dict(revision=1,manifest_sha256=MANIFEST_SHA256,runner_sha256=sha(__file__),
        evaluator_sha256=sha(evaluation.__file__),matcher_sha256=sha('scripts/evaluate_instrument_benchmark.py'),
        order='all 17 application cases, then all 17 Tsumugi cases',
        application=dict(model=str(args.model.resolve()),model_sha256=sha(args.model),source_files={p:sha(p) for p in APP_FILES},
            settings=dict(onset_threshold=.45,frame_threshold=.3),versions=app_runtime,
            resampler='scipy.signal.resample_poly/default',resampled_sample_rate=22050,pcm_conversion='int16/32768 float32',
            normalization=False,inference_scope='full_original_audio',intra_threads=2,inter_threads=1,provider='CPUExecutionProvider'),
        tsumugi=dict(source_commit=SOURCE_COMMIT,source_files=sources,adapter_sha256=sha('scripts/research_tsumugi_transcription.py'),
            model_revision=MODEL_REVISION,model='bass_v2',model_sha256=MODELS['bass_v2'],versions=model_runtime,
            settings=TSUMUGI_SETTINGS,stem_filter='all'),
        execution=dict(application_python=str(args.application_python.resolve()),tsumugi_python=str(args.tsumugi_python.resolve()),
            source=str(args.source.resolve()),dependencies=str(args.dependencies.resolve())))
    path=output/'plan.json'
    if path.exists(): require(read_json(path)==plan,'Frozen bass comparison plan changed')
    else:
        require(not output.exists() or not any(output.iterdir()),'Inspect incomplete destination before creating plan')
        write_new(path,plan)
    env=os.environ|{'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUTF8':'1'}
    subprocess.run([str(args.application_python),str(Path(__file__).resolve()),'application-stage','--root',str(root),
        '--output',str(output)],check=True,env=env)
    completion=read_json(output/'application/completed.json')
    require(completion.get('cases')==17 and completion.get('identity')==plan['application']
        and set(completion.get('case_reports',{}))=={c['name'] for c in cases},'Application stage incomplete')
    verified_hash(path,completion['plan_sha256'])
    for case in cases: verified_hash(output/'application'/case['name']/'report.json',completion['case_reports'][case['name']]['report_sha256'])
    for index,case in enumerate(cases):
        verified_hash(case['audio'],case['audio_sha256'])
        folder=output/'tsumugi'/case['name']
        if not (folder/'report.json').exists():
            require(not folder.exists(),'Inspect partial Tsumugi case before retry: '+case['name'])
            with (output/('tsumugi-'+case['name']+'.log')).open('x',encoding='utf-8') as log:
                subprocess.run([str(args.tsumugi_python),'scripts/research_tsumugi_transcription.py','--audio',case['audio'],
                    '--model','bass_v2','--stem','all','--source',str(args.source),'--dependencies',str(args.dependencies),
                    '--output',str(folder)],check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
        verify_tsumugi(folder,plan,case)
        print(json.dumps(dict(stage='tsumugi',index=index+1,total=17,case=case['name'])),flush=True)
    result=collect_scores(root,output,plan,cases)
    write_new(output/'comparison.json',result)
    write_new(output/'completed.json',dict(identity=plan,cases=17,plan_sha256=sha(path),comparison_sha256=sha(output/'comparison.json')))
    print(json.dumps(dict(stage='complete',groups=result['groups'])),flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest='mode',required=True)
    sub.add_parser('application-runtime')
    child=sub.add_parser('application-stage')
    child.add_argument('--root',type=Path,required=True)
    child.add_argument('--output',type=Path,required=True)
    command=sub.add_parser('run')
    command.add_argument('--root',type=Path,default=Path('test-data/bass-benchmark-v1'))
    command.add_argument('--output',type=Path,required=True)
    command.add_argument('--model',type=Path,default=Path('assets/basic-pitch.onnx'))
    command.add_argument('--application-python',type=Path,default=Path('.venv/Scripts/python.exe'))
    command.add_argument('--tsumugi-python',type=Path,default=Path('.venv-audio/Scripts/python.exe'))
    command.add_argument('--source',type=Path,default=Path('test-data/other-instrument-model-survey-2026-11-10/tsumugi'))
    command.add_argument('--dependencies',type=Path,default=Path('test-data/rosvot-runtime'))
    args=cli.parse_args()
    if args.mode=='application-runtime': print(json.dumps(app_versions()))
    elif args.mode=='application-stage': application_stage(args.root,args.output)
    else: run(args)


if __name__=='__main__': main()
