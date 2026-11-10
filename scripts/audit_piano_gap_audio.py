"""Independently recheck sample windows, powers, clips and selected FFTs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_expression_current_replay import read,save,sha


def integer_limits(a,b,rate,count):
    assert int(a)==a and int(b)==b
    return max(0,(int(a)*rate+999)//1000), min(count,int(b)*rate//1000)


def run(source,output):
    source,output=source.resolve(),output.resolve()
    assert not output.exists()
    report,plan=read(source/'report.json'),read(source/'execution-plan.json')
    assert sha(source/'execution-plan.json')==report['execution_plan_sha256']
    assert sha(source/'records.json')==report['records_sha256']
    assert sha(source/'index.html')==report['index_sha256']
    checks=3
    for name,digest in plan['inputs'].items():
        assert sha(ROOT/name)==digest;checks+=1
    for name,digest in plan['sources'].items():
        assert sha(source/'source'/name)==digest;checks+=1
    manifest=read(ROOT/'test-data/smd/selected-v1/benchmark/manifest.json')
    waveforms={}
    for case in manifest['cases']:
        if case['name'] not in {r['name'] for r in read(source/'records.json')}:continue
        path=ROOT/case['audio'];assert sha(path)==case['audio_sha256']
        rate,raw=wavfile.read(path);raw=raw[:round(case['duration_seconds']*rate)]
        waveforms[case['name']]=(rate,raw,raw.astype(np.float64)/32768)
    rate,raw=wavfile.read(ROOT/'test-data/sakura-benchmark/raw-pcm-audit-v1/piano.wav')
    waveforms['sakura']=(rate,raw,raw.astype(np.float64))
    records=read(source/'records.json');band_regions=0;fft_frames=0;max_error=0.;selection=set(plan['selected_visual_records'])
    for row in records:
        folder=source/row['id']
        for name,digest in row['artifacts'].items():
            assert sha(folder/name)==digest;checks+=1
        record=read(folder/'record.json');m=record['measurement'];rate,raw,signal=waveforms[row['name']]
        a,b=record['interval_ms'];start=max(0,round((a-1000)*rate/1000));end=min(len(raw),round((b+1000)*rate/1000))
        sr,clip=wavfile.read(folder/'original.wav');assert sr==rate and np.array_equal(clip,raw[start:end])
        limits={name:integer_limits(left,right,rate,len(signal)) for name,(left,right) in m['regions_ms'].items()}
        for name,(left,right) in limits.items():
            expected=float(np.dot(signal[left:right],signal[left:right])/(right-left)) if right>left else None
            actual=m['waveform'][name]['power']
            if expected is None:assert actual is None
            else:assert np.isclose(expected,actual,rtol=1e-12,atol=1e-16)
            assert [m['waveform'][name]['start_sample'],m['waveform'][name]['end_sample']]==[left,right]
        if m['waveform']['pre']['power'] and m['waveform']['gap']['power']:
            assert np.isclose(10*np.log10(m['waveform']['gap']['power']/m['waveform']['pre']['power']),m['waveform_gap_pre_db'])
        with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:
            for size_text,window in m['windows'].items():
                size=int(size_text);starts=arrays[f'starts_{size}']
                np.testing.assert_array_equal(arrays[f'times_{size}'],(starts+size/2)/rate)
                masks={name:(starts>=left)&(starts+size<=right) for name,(left,right) in limits.items()}
                assert window['region_frames']=={name:int(mask.sum()) for name,mask in masks.items()}
                for band in window['bands']:
                    power=arrays[band['power_array']]
                    assert len(power)==len(starts) and np.all(power>=0) and np.isfinite(power).all()
                    for name,mask in masks.items():
                        actual=band['regions'][name]['power'];band_regions+=1
                        if mask.any() and band['bins']:
                            assert np.isclose(actual,np.sum(power[mask])/np.count_nonzero(mask),rtol=1e-12,atol=1e-16)
                        else:assert actual is None
                if row['id'] in selection:
                    window_values=.5-.5*np.cos(2*np.pi*np.arange(size)/size)
                    direct=np.stack([np.abs(np.fft.rfft(signal[t:t+size]*window_values))/window_values.sum() for t in starts],axis=1)
                    frequencies=np.fft.rfftfreq(size,1/rate);keep=(frequencies>0)&(frequencies<=5000)
                    np.testing.assert_allclose(arrays[f'magnitude_{size}'],direct[keep],rtol=1e-11,atol=1e-14)
                    max_error=max(max_error,float(np.abs(arrays[f'magnitude_{size}']-direct[keep]).max(initial=0)))
                    for band in window['bands']:
                        expected=np.square(direct[band['bins']]).sum(axis=0)
                        np.testing.assert_allclose(arrays[band['power_array']],expected,rtol=1e-11,atol=1e-15)
                    fft_frames+=len(starts)
    save(output/'report.json',dict(source_report_sha256=sha(source/'report.json'),files_verified=checks,
        interval_count=len(records),sample_identical_audio_clips=len(records),band_regions_checked=band_regions,
        direct_fft_records=len(selection),direct_fft_frames=fft_frames,max_fft_absolute_error=max_error,
        thresholds_fitted=False,neural_inference=False,musical_acceptance=False))
    print(read(output/'report.json'))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();run(args.source,args.output)
