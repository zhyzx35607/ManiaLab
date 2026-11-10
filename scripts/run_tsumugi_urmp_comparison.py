"""Compare fixed Tsumugi other_v1_5 to completed URMP baselines on all 35 inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_guitar_vocal_benchmark import read_json,require,verified_hash,aggregate
from scripts.evaluate_instrument_benchmark import window_metrics
from scripts.research_tsumugi_transcription import SOURCE_COMMIT,MODEL_REVISION,MODELS,sha,verify_source


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/urmp/comparison-extension'))
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--python',type=Path,default=Path('.venv-audio/Scripts/python.exe'))
    cli.add_argument('--source',type=Path,default=Path('test-data/other-instrument-model-survey-2026-11-10/tsumugi'))
    cli.add_argument('--dependencies',type=Path,default=Path('test-data/rosvot-runtime'))
    args=cli.parse_args()
    require(not (args.output/'completed.json').exists(),'Preserve completed comparison')
    completed=read_json(args.root/'completed.json')
    verified_hash(args.root/'comparison.json',completed['report_sha256'])
    frozen=read_json(args.root/'comparison.json')
    manifest=read_json(args.root/'manifest.json')
    verified_hash(args.root/'manifest.json',frozen['manifest_sha256'])
    require(completed['cases']==len(manifest['cases'])==len(frozen['rows'])==35,'Require all 35 inputs')
    require({c['name'] for c in manifest['cases']}=={r['name'] for r in frozen['rows']},'Baseline cases differ')
    source_files=verify_source(args.source)
    model='other_v1_5'
    verified_hash(args.source/'checkpoints'/('best_model_'+model+'.pth'),MODELS[model])
    runtime=json.loads(subprocess.check_output([str(args.python),'-c',
        "import sys,json;sys.path.insert(0,sys.argv[1]);from scripts.research_tsumugi_transcription import runtime_versions;print(json.dumps(runtime_versions()))",
        str(args.dependencies.resolve())],text=True))
    identity=dict(source_commit=SOURCE_COMMIT,source_files=source_files,model_revision=MODEL_REVISION,
        model=model,model_sha256=MODELS[model],adapter_sha256=sha('scripts/research_tsumugi_transcription.py'),
        runner_sha256=sha(__file__),manifest_sha256=sha(args.root/'manifest.json'),
        baseline_completed_sha256=sha(args.root/'completed.json'),baseline_comparison_sha256=sha(args.root/'comparison.json'),
        matcher_sha256=sha('scripts/evaluate_instrument_benchmark.py'),runtime=runtime,
        stem_filter='all',normalization=False,scoring_seconds=30,context_seconds=40)
    args.output.mkdir(parents=True,exist_ok=True)
    plan=args.output/'plan.json'
    if plan.exists():
        require(read_json(plan)==identity,'Frozen URMP Tsumugi plan changed')
    else:
        require(not any(args.output.iterdir()),'Inspect incomplete destination before starting')
        plan.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    from scipy.io import wavfile
    import numpy as np
    old_rows={r['name']:r for r in frozen['rows']}
    rows=[]
    groups={engine:{} for engine in ('application','yourmt3','tsumugi')}
    for index,case in enumerate(manifest['cases']):
        print(json.dumps(dict(stage='tsumugi_urmp',case=case['name'],index=index+1,total=35)),flush=True)
        verified_hash(case['audio'],case['audio_sha256'])
        verified_hash(case['labels_path'],case['labels_sha256'])
        original_rate,pcm=wavfile.read(case['audio'])
        require(pcm.ndim==1 and pcm.dtype==np.int16 and original_rate==16000,'Unexpected URMP PCM')
        sample_count=min(len(pcm),40*original_rate)
        clip=args.output/(case['name']+'.wav')
        if clip.exists():
            rate,saved=wavfile.read(clip)
            require(rate==original_rate and saved.dtype==pcm.dtype and np.array_equal(saved,pcm[:sample_count]),'Input clip differs')
        else:
            wavfile.write(clip,original_rate,pcm[:sample_count])
        clip_hash=sha(clip)
        folder=args.output/case['name']
        if not (folder/'report.json').exists():
            require(not folder.exists(),'Inspect partial Tsumugi output before retry')
            with (args.output/(case['name']+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run([str(args.python),'scripts/research_tsumugi_transcription.py','--audio',str(clip),
                    '--model',model,'--stem','all','--source',str(args.source),'--dependencies',str(args.dependencies),
                    '--output',str(folder)],check=True,stdout=log,stderr=subprocess.STDOUT,
                    env=os.environ|{'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUTF8':'1'})
        report=read_json(folder/'report.json')
        expected=dict(source_commit=SOURCE_COMMIT,source_files=source_files,runner_sha256=identity['adapter_sha256'],
            model_revision=MODEL_REVISION,model=model,model_sha256=MODELS[model],audio=str(clip.resolve()),
            audio_sha256=clip_hash,original_sample_rate=original_rate,original_samples=sample_count,original_channels=1,
            resampled_sample_rate=22050,resampled_samples=(sample_count*22050+original_rate-1)//original_rate,
            resampler='torchaudio.functional.resample/default',normalization=False,mono_policy='duplicate_to_stereo',
            inference_scope='full_original_audio',intra_threads=2,inter_threads=1,amp=False,compile=False,seed=42,
            stem_filter='all',versions=runtime,evidence_revision=2)
        require(all(report['identity'].get(k)==v for k,v in expected.items()),'Tsumugi identity differs')
        require(report['context_start_seconds']==0 and report['context_end_seconds']==sample_count/original_rate,'Input clock changed')
        for name in ('notes','raw_notes','window_notes','interval_constructions'):
            verified_hash(folder/(name.replace('_','-')+'.json'),report[name+'_sha256'])
        old=old_rows[case['name']]
        require(old['interval_scope']['begin_seconds']==0 and old['interval_scope']['end_seconds']==30 and
                old['interval_scope']['context_start_seconds']==0 and
                old['interval_scope']['context_end_seconds']==sample_count/original_rate,'Baseline clock differs')
        predicted=read_json(folder/'notes.json')
        require(len(predicted)==report['notes'],'Prediction count differs')
        metrics,scope=window_metrics(read_json(case['labels_path']),predicted,begin=0,end=30,
            context_start=0,context_end=sample_count/original_rate)
        metrics={t:{m:v for m,v in values.items() if 'instrument' not in m} for t,values in metrics.items()}
        row=dict(name=case['name'],kind=case['kind'],instrument=case.get('instrument'),
            source_sha256=case['audio_sha256'],clip_sha256=clip_hash,source_samples=len(pcm),clip_samples=sample_count,
            report_sha256=sha(folder/'report.json'),notes_sha256=report['notes_sha256'],interval_scope=scope,
            metrics=metrics,application=old['application_detector']['metrics'],yourmt3=old['metrics'])
        rows.append(row)
        keys=[case['kind']]+(['instrument/'+case['instrument']] if case.get('instrument') else [])
        for engine,values in (('application',row['application']),('yourmt3',row['yourmt3']),('tsumugi',metrics)):
            values={('fixed_40_80' if t=='strict' else t):{m:dict(available=True,**v) for m,v in group.items()
                    if 'instrument' not in m} for t,group in values.items()}
            for key in keys:
                groups[engine].setdefault(key,[]).append(dict(name=case['name'],metrics=values))
    result=dict(revision=1,identity=identity,cases=35,rows=rows,
        groups={engine:{key:aggregate(group) for key,group in values.items()} for engine,values in groups.items()},
        enabled_in_application=False,limitations=[
            'Same 7 performances appear in solo and mixture inputs; not 35 independent songs.',
            'Public corpus may be upstream validation/training; not an unseen blind test.',
            'No instrument-class filtering or GM accuracy claims; all predicted notes retained.',
            'Physical model note tails do not specify authored LN gestures.'])
    target=args.output/'comparison.json'
    with target.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2,allow_nan=False))
    (args.output/'completed.json').write_text(json.dumps(dict(identity=identity,cases=35,
        plan_sha256=sha(plan),comparison_sha256=sha(target)),indent=2),encoding='utf-8')
    print(json.dumps(dict(stage='complete',groups=result['groups'])),flush=True)


if __name__=='__main__':main()
