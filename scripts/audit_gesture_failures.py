"""Inspect frozen long-LN failures without modifying predictions or thresholds."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.anchor_gestures import AnchorContext, probabilities
from manialab.acoustic_anchors import AcousticAnchorContext
from manialab.release_candidates import ReleaseContext, AcousticReleaseContext
from manialab.domain import parse_osu
from scripts.anchor_acoustic_inputs import load_evidence
from scripts.probe_mapper_adapter import sha
from scripts.train_anchor_gestures import source_chart


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    output=args.run/'long-hold-diagnosis.json'
    if output.exists():raise ValueError('Preserve prior diagnosis')
    model=json.loads(args.model.read_text(encoding='utf-8'))
    report=json.loads((args.run/'report.json').read_text(encoding='utf-8'))
    if report['identity']['model_sha256']!=sha(args.model):raise ValueError('Wrong model for generated run')
    root=Path('test-data/sakura-benchmark');chart=source_chart(root/'detected-onset-sketch-v1/sketch.osu')
    independent=model['revision'].startswith('independent-release-')
    ratio=961/2806
    if model.get('acoustic_feature_sha256'):
        cls=AcousticReleaseContext if independent else AcousticAnchorContext
        evidence,_=load_evidence(root);context=cls(chart['notes'],10,chart['timing'],ratio,evidence)
    else:context=(ReleaseContext if independent else AnchorContext)(chart['notes'],10,chart['timing'],ratio)
    pred=probabilities(context,model,report['calibration']['ln_bias'])
    human=parse_osu((root/'finished.osu').read_text(encoding='utf-8-sig'))['notes']
    generated=parse_osu((args.run/'planned.osu').read_text(encoding='utf-8-sig'))['notes']
    rows=[]
    for n in human:
        if n.end-n.start<2000:continue
        nearest=int(np.argmin(abs(context.times-n.start)))
        row=dict(start_ms=n.start,end_ms=n.end,region='later' if n.start>=188286 else 'development',
            head_distance_ms=int(abs(context.times[nearest]-n.start)),input_count=int(context.counts[nearest]),
            source_counts=dict(Counter(x.voice for x in context.notes[nearest])))
        options=context.endpoint_times(nearest,pred[nearest]['options']);prob=pred[nearest]['probabilities']
        endpoint=int(np.argmin(abs(options[1:-1]-n.end)))+1
        rank=int(np.sum(prob>prob[endpoint]))+1
        selected=[x.end for x in generated if abs(x.start-n.start)<=30]
        row.update(candidate_error_ms=int(abs(options[endpoint]-n.end)),candidate_probability=float(prob[endpoint]),
            candidate_rank=rank,expected_count=float(prob[endpoint]*context.counts[nearest]),
            tap_probability=float(prob[0]),unresolved_probability=float(prob[-1]),
            predicted_ends_ms=selected,
            top_options=[dict(time_ms=int(options[j]),probability=float(prob[j])) for j in np.argsort(-prob)[:5]],
            endpoint_match=any(abs(end-n.end)<=60 for end in selected))
        rows.append(row)
    output.write_text(json.dumps(dict(model_sha256=sha(args.model),chart_sha256=sha(args.run/'planned.osu'),rows=rows),indent=2),encoding='utf-8')
    print(json.dumps(dict(total=len(rows),later=[dict(start=r['start_ms'],candidate_error=r['candidate_error_ms'],
        rank=r['candidate_rank'],expected_count=r['expected_count'],matched=r['endpoint_match']) for r in rows if r['region']=='later'])),flush=True)


if __name__=='__main__':main()
