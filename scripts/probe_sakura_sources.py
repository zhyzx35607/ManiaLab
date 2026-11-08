"""Diagnose source-aware onset coverage on mapper-confirmed Nova stems.

The reference is a musical transcription with limited multiplicity, not MIDI
truth. No reference timings are used by the audio detector. Both raw and a
predeclared 22.5ms chart-clock shift are reported, without timing-grid snapping.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.audio import infer, pitched_notes, rhythmic_evidence, source_evidence, refine_releases, fuse_attacks
from manialab.domain import parse_osu, note_from
from scripts.probe_mapper_adapter import event_similarity


def grouped_onsets(notes, shift=0., tolerance=30):
    """Collapse instrument chord attacks, never merge a chain across a window."""
    groups = []
    for n in sorted(notes, key=lambda n:n.start):
        t = n.start-shift
        if not groups or t-groups[-1][0] > tolerance:
            groups.append([t])
        else:
            groups[-1].append(t)
    return [(float(np.median(g)),) for g in groups]


def match_source(reference, predicted, shift):
    return dict(onset_groups=event_similarity(grouped_onsets(reference, tolerance=0), grouped_onsets(predicted, shift), [40]),
        note_multiplicity=event_similarity([(n.start,) for n in reference], [(n.start-shift,) for n in predicted], [40]),
        duration_notice='Reference taps have no acoustic release labels; pitch-count matches are not equivalent to authored layer multiplicity.')


def pcm(path):
    rate, samples = wavfile.read(path)
    if rate != 22050:
        raise ValueError('Prepared preview sample clock changed')
    return samples.astype(np.float32)/32768


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/sakura-benchmark'))
    args = cli.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    sources={r['id']:r for r in manifest['audio']}
    reference=parse_osu((root/'reference.osu').read_text(encoding='utf-8-sig'))['notes']
    model=Path('assets/basic-pitch.onnx')
    mix=pcm(root/sources['mix']['preview'])
    output=root/'detector-baseline';output.mkdir(exist_ok=True)
    all_notes={};report=dict(sources={},comparison={},automatic_snapping=False,default_model_changed=False)
    for voice in ('drums','piano','other','guitar','bass','vocals'):
        row=sources[voice];source=root/row['path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Bundled source changed')
        samples=pcm(root/row['preview'])
        if len(samples)!=len(mix):raise ValueError('Source clocks differ')
        began=time.monotonic()
        attacks,_=rhythmic_evidence(samples,estimate=False)
        attacks,_=source_evidence(attacks,samples,mix)
        if voice=='drums':
            notes=attacks
        else:
            cache=output/(voice+'-activations.npz')
            identity=dict(audio=row['sha256'],model=hashlib.sha256(model.read_bytes()).hexdigest(),revision=1)
            if cache.exists():
                with np.load(cache,allow_pickle=False) as saved:
                    if json.loads(str(saved['identity']))!=identity:raise ValueError('Stale detector cache')
                    frames,onsets,times=saved['frames'],saved['onsets'],saved['times']
            else:
                frames,onsets,times=infer(samples,model)
                np.savez_compressed(cache,frames=frames,onsets=onsets,times=times,identity=json.dumps(identity))
            notes=pitched_notes(frames,onsets,times,len(samples)/22050,attacks=attacks)
            notes,_=source_evidence(notes,samples,mix)
            notes,_=refine_releases(notes,samples)
            notes,_,_=fuse_attacks(notes,attacks)
        notes=[replace(n,voice=voice,id=voice+':'+n.id) for n in notes]
        all_notes[voice]=notes
        (output/(voice+'-notes.json')).write_text(json.dumps([n.data() for n in notes]),encoding='utf-8')
        report['sources'][voice]=dict(candidates=len(notes),seconds=round(time.monotonic()-began,2),audio_sha256=row['sha256'])
        print(json.dumps(dict(source=voice,**report['sources'][voice])),flush=True)
    for voice,lanes in dict(drums=range(4),piano=range(4,6),other=range(6,10)).items():
        human=[n for n in reference if n.lane in lanes]
        predicted=all_notes[voice] if voice!='other' else sum([all_notes[v] for v in ('other','guitar','bass')],[])
        report['comparison'][voice]={str(shift):match_source(human,predicted,shift) for shift in (0.,22.5)}
        report['comparison'][voice]['segments']=[dict(start_ms=s['start_ms'],end_ms=s['end_ms'],
            **match_source([n for n in human if s['start_ms']<=n.start<s['end_ms']],
                           [n for n in predicted if s['start_ms']<=n.start-22.5<s['end_ms']],22.5))
            for s in manifest['segments']]
    report['notice']='User-confirmed source labels; no vocal notes in the reference. Other is a union of guitar/bass/residual, not a violin-only label. No inference tuning on reference times.'
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({voice:{k:v for k,v in value.items() if k!='segments'} for voice,value in report['comparison'].items()}),flush=True)


if __name__=='__main__':main()
