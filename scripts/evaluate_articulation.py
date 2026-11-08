"""Check low-pitch reattack suppression against isolated tones and real rests."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.io import wavfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.audio import infer,pitched_notes,rhythmic_evidence,refine_releases,fuse_attacks
from manialab.articulation import steady_bass_onsets
from manialab.domain import note_from
from scripts.evaluate_continuity_controls import synthesize
from scripts.evaluate_synthetic_mix import pitched_match


def compare(folder, audio, truth):
    rate,w=wavfile.read(audio);w=w.astype(np.float32)/32768
    cache=folder/'activations.npz'
    if cache.exists():
        d=np.load(cache);f,o,t=d['frames'],d['onsets'],d['times']
    else:
        f,o,t=infer(w,Path('assets/basic-pitch.onnx'));np.savez_compressed(cache,frames=f,onsets=o,times=t)
    attacks,_=rhythmic_evidence(w,estimate=False)
    removed=steady_bass_onsets(f,o,t,w)
    result={'suppressed':len(removed)}
    for mode,suppressed in [('baseline',[]),('probe',removed)]:
        notes=pitched_notes(f,o,t,len(w)/rate,attacks=attacks,suppressed=suppressed)
        notes,_=refine_releases(notes,w);notes,_,_=fuse_attacks(notes,attacks)
        result[mode]=dict(attacks=pitched_match(notes,truth),intervals=pitched_match(notes,truth,True),
            long_notes_over_800ms=sum(n.end-n.start>=800 and n.pitch in {x.pitch for x in truth} for n in notes))
    return result


def main():
    root=Path('test-data/articulation-audit');root.mkdir(exist_ok=True)
    isolated=Path('test-data/isolated-source-audit')
    truth=[note_from(n) for n in json.loads((isolated/'ground-truth.json').read_text())]
    report=dict(isolated={},repeated=[],enabled_in_application=False)
    for v in ('bass','lead','pad'):
        row=compare(isolated/v,isolated/v/'preview.wav',[n for n in truth if n.voice==v])
        report['isolated'][v]=row;print(v,json.dumps(row),flush=True)
    for pitch in (28,36,44,48):
        for gap in (10,30,80):
            for distractor in (False,True):
                work=root/f'{pitch}-{gap}-{int(distractor)}'
                truth=synthesize(work,pitch,gap,distractor)
                row=dict(pitch=pitch,gap_ms=gap,distractor=distractor,**compare(work,work/'repeated.wav',truth))
                report['repeated'].append(row)
                (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(pitch,gap,distractor,row['suppressed'],row['baseline']['attacks']['matched'],row['probe']['attacks']['matched'],flush=True)
    report['regressions']=[r for r in report['repeated'] if r['probe']['attacks']['matched']<r['baseline']['attacks']['matched'] or r['probe']['long_notes_over_800ms']>r['baseline']['long_notes_over_800ms']]
    (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
