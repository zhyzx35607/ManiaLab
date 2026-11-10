"""Review all fixed candidate gaps against PCM, spectra and existing context."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_expression_current_replay import read,save,sha,ANALYSIS,ANALYSIS_SHA
from scripts.piano_gap_audio import measure_gap

AUDIT=ROOT/'test-data/piano-inactivity-audit-v1'
PREVIOUS=ROOT/'test-data/piano-inactivity-context-v1'
SMD=ROOT/'test-data/smd/selected-v1/benchmark'
GAP_SHA='80c0a9be2657c1333b8ee72f5ce7fd1d723df02a21c52058bb3af2d715b4f7b7'


def plot_record(signal,rate,activations,record,arrays,path):
    import matplotlib.pyplot as plt
    measurement=record['measurement'];a,b=record['interval_ms']
    begin,end=max(0,a-1000),min(len(signal)*1000/rate,b+1000)
    first,last=round(begin*rate/1000),round(end*rate/1000)
    fig,axes=plt.subplots(5,1,figsize=(13,11),sharex=True,
        gridspec_kw={'height_ratios':[1,1.5,1.5,1,1]})
    axes[0].plot((first+np.arange(last-first)[::16])/rate,signal[first:last:16],lw=.5)
    axes[0].set_ylabel('Original PCM')
    axes[0].set_title(f"{record['name']} | {a/1000:.3f}-{b/1000:.3f}s | {record['category']}\n"
        'Held keys / pedal labels do not establish audible sustain',loc='left',fontsize=10)
    common=max(float(arrays[f'magnitude_{size}'].max(initial=0)) for size in (2048,8192))
    for axis,size in zip(axes[1:3],(2048,8192)):
        times,freq=arrays[f'times_{size}'],arrays[f'frequencies_{size}']
        if len(times):
            db=20*np.log10(np.maximum(arrays[f'magnitude_{size}'],1e-12)/max(common,1e-12))
            axis.pcolormesh(times,69+12*np.log2(freq/440),db,shading='auto',vmin=-75,vmax=0,cmap='magma')
        axis.set_ylim(20,105);axis.set_ylabel(f'{size} FFT\nMIDI frequency')
        axis.set_title(f"{size/rate*1000:.1f} ms window; complete gap windows: "
            f"{measurement['windows'][str(size)]['region_frames']['gap']}",fontsize=8,loc='left')
    lo,hi=max(0,int(begin//10)),min(len(activations['frame_output']),int(np.ceil(end/10)))
    times=np.arange(lo,hi)/100
    axes[3].plot(times,activations['frame_output'][lo:hi].max(axis=1),label='Max frame / all pitches')
    axes[3].plot(times,activations['reg_onset_output'][lo:hi].max(axis=1),label='Max onset / all pitches')
    axes[3].axhline(.1,lw=.4,ls='--');axes[3].axhline(.3,lw=.4,ls='--')
    axes[3].legend(fontsize=8);axes[3].set_ylabel('Model activity')
    for note in record['reference_context']:
        axes[4].plot([note['onset'],note['offset']],[note['pitch']]*2,color='#f59e0b',lw=4,alpha=.6)
        axes[4].plot([note['onset'],note['key_offset']],[note['pitch']]*2,color='#0369a1',lw=1.5)
    for n in record['raw_context']:
        axes[4].plot([n['start']/1000,n['end']/1000],[n['pitch']-.2]*2,color='#9ca3af',lw=.5)
    axes[4].set_ylabel('Key blue / pedal gold\nraw candidates gray');axes[4].set_xlabel('Original audio time (s)')
    for axis in axes:
        axis.axvspan(a/1000,b/1000,color='green',alpha=.12)
        axis.set_xlim(begin/1000,end/1000)
    fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)


def stats(values):
    measured=[float(x) for x in values if x is not None]
    return dict(count=len(values),measured=len(measured),minimum=min(measured) if measured else None,
        median=float(np.median(measured)) if measured else None,maximum=max(measured) if measured else None)


def run(output,dependencies):
    output=output.resolve();sys.path.insert(0,str(dependencies.resolve()))
    import matplotlib
    matplotlib.use('Agg')
    assert not output.exists() and output.is_relative_to(ROOT/'test-data')
    assert sha(AUDIT/'all-smd-gaps.json')==GAP_SHA
    assert sha(AUDIT/'report.json')=='780141c50038d0117e6e4dacb6be7fae9fe1795c4caf8b61fcb62879f99fcdc0'
    audit=read(AUDIT/'report.json');previous=read(PREVIOUS/'report.json')
    assert sha(PREVIOUS/'report.json')==audit['source_report_sha256']
    assert sha(PREVIOUS/'execution-plan.json')==previous['execution_plan_sha256']
    prior_plan=read(PREVIOUS/'execution-plan.json')
    inputs={AUDIT/'all-smd-gaps.json':GAP_SHA,AUDIT/'report.json':sha(AUDIT/'report.json'),
        PREVIOUS/'report.json':sha(PREVIOUS/'report.json'),PREVIOUS/'execution-plan.json':sha(PREVIOUS/'execution-plan.json')}
    manifest=read(SMD/'manifest.json');gaps=read(AUDIT/'all-smd-gaps.json');jobs=[]
    for group in gaps:
        if not group['all_eligible_candidate_gaps']:continue
        case=next(c for c in manifest['cases'] if c['name']==group['name'])
        audio=ROOT/case['audio'];labels=ROOT/case['sustain_labels']
        activation=SMD/'specialist-v2'/f"{case['name']}-activations.npz"
        events=PREVIOUS/'smd'/case['name']/'candidates.json'
        expected=next(c for c in previous['smd'] if c['name']==case['name'])
        for path,digest in ((audio,case['audio_sha256']),(labels,case['sustain_labels_sha256']),
            (activation,prior_plan['inputs'][activation.relative_to(ROOT).as_posix()]),
            (events,expected['artifacts']['candidates.json'])):
            assert sha(path)==digest;inputs[path]=digest
        jobs.append(dict(name=case['name'],audio=audio,labels=labels,activation=activation,events=events,
            intervals=[m for gap in group['all_eligible_candidate_gaps'] for m in gap['measurements']],
            seconds=case['duration_seconds']))
    assert sum(len(j['intervals']) for j in jobs)==104
    assert sha(PREVIOUS/'review.json')==previous['review_sha256']
    inputs[PREVIOUS/'review.json']=previous['review_sha256']
    reviews=read(PREVIOUS/'review.json');assert len(reviews)==6
    piano=ROOT/'test-data/sakura-benchmark/raw-pcm-audit-v1/piano.wav'
    caches=[ROOT/p for p in prior_plan['inputs'] if p.endswith('piano-specialist-activations.npz')]
    assert len(caches)==1
    for path in (piano,caches[0],ANALYSIS):
        digest=prior_plan['inputs'][path.relative_to(ROOT).as_posix()]
        assert sha(path)==digest;inputs[path]=digest
    assert sha(ANALYSIS)==ANALYSIS_SHA
    jobs.append(dict(name='sakura',audio=piano,labels=None,activation=caches[0],events=ANALYSIS,seconds=283.84,
        intervals=[dict(start_ms=r['candidate_gap'][0],end_ms=r['candidate_gap'][1],
            key_activity_ms=None,pedal_activity_ms=None,previous_low_interval=r['model_low_interval']) for r in reviews]))
    sources=['scripts/review_piano_gap_audio.py','scripts/piano_gap_audio.py','scripts/piano_band_evidence.py',
        'scripts/check_expression_current_replay.py','docs/piano-gap-audio-protocol.md']
    source_hashes={s:sha(ROOT/s) for s in sources}
    selected=[]
    for job in jobs[:-1]:
        conflicts=[(i,r) for i,r in enumerate(job['intervals']) if r['pedal_activity_ms']>0]
        if not conflicts:continue
        index=max(conflicts,key=lambda x:(x[1]['pedal_activity_ms'],-x[0]))[0]
        selected.append(f"{job['name']}-{index:03d}")
        first=next((i for i,r in enumerate(job['intervals']) if r['pedal_activity_ms']==0),None)
        if first is not None:selected.append(f"{job['name']}-{first:03d}")
    selected+=['sakura-000','sakura-001']
    save(output/'execution-plan.json',dict(inputs={p.relative_to(ROOT).as_posix():h for p,h in inputs.items()},
        sources=source_hashes,selected_visual_records=selected,neural_inference=False,
        all_smd_intervals=104,sakura_intervals=6,automatic_threshold_search=False))
    for name in sources:
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    records=[]
    for job in jobs:
        rate,raw=wavfile.read(job['audio']);assert rate==22050 and raw.ndim==1
        raw=raw[:round(job['seconds']*rate)]
        signal=raw.astype(np.float64)/(32768 if raw.dtype==np.int16 else 1)
        assert raw.dtype in (np.int16,np.float32)
        with np.load(job['activation'],allow_pickle=False) as values:
            activations={key:values[key] for key in ('frame_output','reg_onset_output')}
        assert len(activations['frame_output'])==round(job['seconds']*100)
        labels=read(job['labels']) if job['labels'] else []
        notes=read(job['events'])
        if job['name']=='sakura':notes=[n for n in notes['candidates'] if n['voice']=='piano']
        for index, interval in enumerate(job['intervals']):
            a,b=interval['start_ms'],interval['end_ms'];identity=f"{job['name']}-{index:03d}"
            reference=[l for l in labels if l['onset']*1000<b and l['offset']*1000>a]
            preceding=[n for n in notes if n['start']<a and n['end']>a-500]
            pitches={n['pitch'] for n in reference}|{n['pitch'] for n in preceding}
            measurement,arrays=measure_gap(signal,rate,a,b,pitches)
            category=('unknown_reference' if job['name']=='sakura' else 'key_overlap' if interval['key_activity_ms']>0
                else 'pedal_only' if interval['pedal_activity_ms']>0 else 'no_reference_overlap')
            record=dict(id=identity,name=job['name'],interval_ms=[a,b],category=category,
                source_interval=interval,reference_active=reference,preceding_candidates=preceding,
                measurement=measurement,
                reference_context=[l for l in labels if l['onset']*1000<b+1000 and l['offset']*1000>a-1000],
                raw_context=[n for n in notes if n['start']<b+1000 and n['end']>a-1000])
            folder=output/identity;folder.mkdir()
            save(folder/'record.json',record)
            np.savez_compressed(folder/'arrays.npz',**arrays)
            first=max(0,round((a-1000)*rate/1000));last=min(len(raw),round((b+1000)*rate/1000))
            wavfile.write(folder/'original.wav',rate,raw[first:last])
            sr,back=wavfile.read(folder/'original.wav');assert sr==rate and np.array_equal(back,raw[first:last])
            plot_record(signal,rate,activations,record,arrays,folder/'evidence.png')
            records.append(dict(id=identity,name=job['name'],interval_ms=[a,b],category=category,
                key_activity_ms=interval['key_activity_ms'],pedal_activity_ms=interval['pedal_activity_ms'],
                waveform_gap_pre_db=measurement['waveform_gap_pre_db'],
                gap_rms=float(np.sqrt(measurement['waveform']['gap']['power'])),
                pre_rms=float(np.sqrt(measurement['waveform']['pre']['power'])),
                complete_gap_frames={size:row['region_frames']['gap'] for size,row in measurement['windows'].items()},
                reference_pitches=sorted({l['pitch'] for l in reference}),
                artifacts={p.name:sha(p) for p in folder.iterdir()}))
        print(f"{job['name']}: measured and saved {len(job['intervals'])} fixed gaps",flush=True)
    save(output/'records.json',records)
    cards=[]
    for record in records:
        label=html.escape(f"{record['id']} · {record['interval_ms'][0]/1000:.3f}–{record['interval_ms'][1]/1000:.3f}s · {record['category']}")
        identity=record['id']
        cards.append(f'<section><h2>{label}</h2><audio controls preload="none" src="{identity}/original.wav"></audio>'
                     f'<img loading="lazy" src="{identity}/evidence.png" alt="原声、频谱、模型活动与参考保持区间"></section>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>钢琴低活动原声对照</title>'
    page+='<style>body{max-width:1250px;margin:32px auto;font:16px sans-serif;background:#f7fafc;color:#17212b}section{background:white;padding:16px;margin:24px 0}img{display:block;width:100%;height:auto}h2{font-size:18px}audio{width:100%}</style>'
    page+='<h1>钢琴低活动原声对照</h1><p>原声未做响度归一化。低活动、按键/踏板保持与可闻持续是不同概念；本页不判定物理松键或制谱质量。</p>'
    page+=''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    aggregates=[]
    for category in sorted({r['category'] for r in records}):
        group=[r for r in records if r['category']==category]
        aggregates.append(dict(category=category,intervals=len(group),
            waveform_gap_pre_db=stats([r['waveform_gap_pre_db'] for r in group]),
            gap_rms=stats([r['gap_rms'] for r in group]),
            unmeasurable={str(s):sum(r['complete_gap_frames'][str(s)]==0 for r in group) for s in (512,2048,8192)}))
    assert all(sha(p)==h for p,h in inputs.items()) and source_hashes=={s:sha(ROOT/s) for s in sources}
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    save(output/'report.json',dict(execution_plan_sha256=sha(output/'execution-plan.json'),records_sha256=sha(output/'records.json'),
        index_sha256=sha(output/'index.html'),intervals=len(records),aggregates=aggregates,
        source_samples_preserved=True,neural_inference=False,threshold_fitted=False,
        musical_acceptance=False,default_enabled=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--plot-dependencies',type=Path,required=True)
    args=cli.parse_args();run(args.output,args.plot_dependencies)
