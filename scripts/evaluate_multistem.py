"""Compare the same local input with frozen four/six-source worker outputs.

The controlled mixture has known pitches and intervals. Natural-music outputs
are inventory and timings only, never automatically labelled transcription truth.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from manialab.audio import analyze_deep
from manialab.arrange import parameters
from manialab.domain import note_from
from scripts.evaluate_synthetic_mix import pitched_match


def run(source,work,truth=None):
    output=dict(audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                protocol=__doc__,modes={})
    for profile,directory in [('four','stems'),('six','stems-six')]:
        folder=work/directory
        metadata=json.loads((folder/'separation.json').read_text(encoding='utf-8'))
        metadata.update(audio_sha256=output['audio_sha256'],profile=profile,cached=True)
        class Frozen:
            def separate(self,*args):return folder,metadata
        analysis=analyze_deep(source,work,ROOT/'assets/basic-pitch.onnx',parameters(),Frozen())
        (work/f'{profile}-analysis.json').write_text(json.dumps(analysis,ensure_ascii=False),encoding='utf-8')
        notes=[note_from(n) for n in analysis['candidates']]
        row=dict(voices=dict(Counter(n.voice for n in notes)),holds=sum(n.hold for n in notes),
            separation_seconds=metadata['elapsed_seconds'],analysis_seconds=analysis['elapsed_seconds'],
            tempo={k:analysis['tempo'].get(k) for k in ['bpm','offset','stable','coverage','median_error_ms']},
            source_model_sha256=metadata['model_sha256'])
        if truth:
            reference=[note_from(n) for n in json.loads(truth.read_text(encoding='utf-8'))]
            row.update(pitched_onsets=pitched_match(notes,reference),pitched_intervals=pitched_match(notes,reference,True))
        output['modes'][profile]=row
        print(json.dumps({profile:row}),flush=True)
        (work/'comparison.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('work',type=Path)
    parser.add_argument('--truth',type=Path)
    args=parser.parse_args();run(args.source,args.work,args.truth)
