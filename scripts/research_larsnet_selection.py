"""Add frozen separated-stem energy to the matched-candidate drum control."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from scripts.probe_mapper_adapter import sha
from scripts.research_drum_mixture import linear_control, overlap_recalls, TRAIN_END, CALIBRATION_END
from scripts.research_source_attack_classes import CLASSES, drum_event_counts, evaluate


def stem_features(run, times, identity):
    features = []
    for role in ('kick', 'snare', 'toms', 'hihat', 'cymbals'):
        path = run / (role+'.wav')
        if sha(path) != identity['stems'][role]['sha256']:
            raise ValueError('Frozen drum separation changed')
        rate, audio = wavfile.read(path)
        if rate != 44100 or audio.shape != (identity['identity']['samples'],2) or not np.isfinite(audio).all():
            raise ValueError('Unexpected separated stem clock')
        energy = np.mean(audio.astype(float)**2, axis=1)
        cumulative = np.r_[0., np.cumsum(energy)]
        vectors = []
        for low, high in ((-45,-5),(-5,25),(25,70),(70,140)):
            lo = np.clip(np.ceil((times+low)*rate/1000).astype(int),0,len(audio))
            hi = np.clip(np.ceil((times+high)*rate/1000).astype(int),0,len(audio))
            vectors.append((cumulative[hi]-cumulative[lo])/np.maximum(1,hi-lo))
        features.append(np.asarray(vectors).T)
    power = np.stack(features, axis=1)
    scale = np.maximum(power.sum(axis=1).max(axis=1),1e-12)
    return np.c_[np.log1p(power/scale[:,None,None]*100).reshape(len(times),20),
        np.log1p(np.maximum(0,power[:,:,1]-power[:,:,0])/scale[:,None]*100),
        np.log1p(power*1e6).reshape(len(times),20)]


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier selection experiments')
    root=args.root;read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    source=root/'raw-pcm-audit-v1/drums.wav';reference_path=root/'reference.osu'
    candidates_path=root/'detector-baseline/drums-notes.json'
    baseline=read(root/'drum-mixture-v1/report.json');separation=read(args.run/'report.json')
    for path,key in [(source,'source_audio_sha256'),(reference_path,'reference_sha256'),(candidates_path,'candidates_sha256')]:
        if sha(path)!=baseline['identity'][key]:raise ValueError('Matched control identity changed')
    if separation['identity']['original_audio_sha256']!=baseline['identity']['original_sha256']:
        raise ValueError('Separation used a different original source')
    rate,audio=wavfile.read(source)
    times=np.asarray([n['start'] for n in read(candidates_path)],float);chart_times=times-22.5
    reference=list(drum_event_counts(parse_osu(reference_path.read_text(encoding='utf-8-sig'))['notes']))
    linear,control,control_selection=linear_control(audio,rate,times,reference)
    evidence=stem_features(args.run,times,separation)
    enriched,predictions,selection=linear_control(audio,rate,times,reference,extra_features=evidence)
    segments={}
    for name,start,end in [('fit',0,TRAIN_END),('calibration',TRAIN_END,CALIBRATION_END),('later_sections',CALIBRATION_END,len(audio)/rate*1000)]:
        segments[name]={}
        for label,scores,thresholds in [('linear_float',control,linear['thresholds']),('separated_evidence',predictions,enriched['thresholds'])]:
            segments[name][label]=dict(scores=evaluate(reference,chart_times,scores,thresholds,start,end),
                author_overlap=overlap_recalls(reference,chart_times,scores,thresholds,start,end))
        if segments[name]['linear_float']!=baseline['segments'][name]['linear_float']:
            raise ValueError('Matched float baseline failed to reproduce')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'model.npz',**enriched,evidence=predictions,stem_features=evidence,times=times)
    events=[dict(start_ms=float(t),kind=kind,score=float(predictions[i,c])) for i,t in enumerate(chart_times)
        for c,kind in enumerate(CLASSES) if predictions[i,c]>=enriched['thresholds'][c]]
    (args.output/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    (args.output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    result=dict(source_identity=baseline['identity'],separation_sha256=sha(args.run/'report.json'),
        protocol_sha256=sha('docs/larsnet-selection-protocol.md'),runner_sha256=sha(__file__),
        control_code_sha256=sha('scripts/research_drum_mixture.py'),segments=segments,
        model_sha256=sha(args.output/'model.npz'),events_sha256=sha(args.output/'events.json'),
        enabled_in_application=False,notice='Same-song development of authored selection, not exhaustive drum truth or cross-song validation.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(later=segments['later_sections'],thresholds=enriched['thresholds'].tolist())),flush=True)


if __name__=='__main__':main()
