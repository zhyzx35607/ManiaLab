"""Frozen actual generation comparison for rhythm and Mapper-audio evidence."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, validate
from manialab.release_candidates import ReleaseContext
from manialab.mapper_release_evidence import MapperReleaseContext
from manialab.anchor_gestures import probabilities
from manialab.gesture_flow import plan_flow
from scripts.prepare_mapper_releases import load_evidence
from scripts.probe_mapper_adapter import sha
from scripts.evaluate_anchor_gestures import greedy_lanes, head_scoped_comparison, nearest_baseline


class ContextView:
    """Choose planning heads while retaining full-chart rhythmic observations."""
    def __init__(self, context, indices):
        self.context, self.indices = context, list(indices)
        self.times = context.times[self.indices]; self.counts = context.counts[self.indices]
        self.notes = [context.notes[i] for i in self.indices]
        self.keys, self.ln_condition = context.keys, context.ln_condition
        self.revision, self.feature_names = context.revision, context.feature_names
        self.endpoint_source = context.endpoint_source

    def options(self,i):return self.context.options(self.indices[i])
    def endpoint_times(self,i,options):return self.context.endpoint_times(self.indices[i],options)
    def features(self,i,options=None):return self.context.features(self.indices[i],options)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','model','control','pretrained','output'):cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--split',choices=('validation','test'),required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier generation comparisons')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(args.dataset/'manifest.json');model=read(args.model);control=read(args.control);initial=read(args.pretrained)
    if (model['dataset_sha256']!=sha(args.dataset/'manifest.json')
            or control['parent_dataset_sha256']!=sha(args.dataset/'manifest.json')
            or initial['dataset_sha256']!=manifest['pretraining_dataset_sha256']):
        raise ValueError('Models do not have compatible data and song splits')
    if (model['embedding_feature_sha256']!=sha('manialab/mapper_release_evidence.py')
            or manifest['loader_sha256']!=sha('scripts/prepare_mapper_releases.py')):
        raise ValueError('Frozen embedding implementation changed')
    for trained in (model,control,initial):
        if trained['release_feature_sha256']!=sha('manialab/release_candidates.py'):raise ValueError('Candidate features changed')
    args.output.mkdir(parents=True);rows=[];excluded=[]
    for record in manifest['records']:
        if record['split']!=args.split:continue
        if sha(record['chart'])!=record['chart_sha256']:raise ValueError('Chart changed')
        chart=parse_osu(Path(record['chart']).read_text(encoding='utf-8-sig'))
        ratio=sum(n.hold for n in chart['notes'])/len(chart['notes'])
        notes=[replace(n,end=n.start,lane=0,voice='mix',raw='') for n in chart['notes']]
        rhythm=ReleaseContext(notes,chart['keys'],chart['timing'],ratio)
        acoustic=MapperReleaseContext(notes,chart['keys'],chart['timing'],ratio,load_evidence(record))
        supported=[i for i in range(len(acoustic.times)) if acoustic.supported(i)]
        if not supported:raise ValueError('Prepared chart lost all audio support')
        first,last=record['score_bounds']
        left_limit=max(first,float(rhythm.time_at(rhythm.beats[supported[0]]+24)))
        right_limit=min(last,float(rhythm.time_at(rhythm.beats[supported[-1]]-24)))
        if right_limit-left_limit<1000:
            excluded.append(dict(id=record['id'],reason='Insufficient complete audio for 24-beat planning margins'));continue
        start=int(max(left_limit,(left_limit+right_limit-16000)/2));end=int(min(right_limit,start+16000))
        lo=float(rhythm.beat_at(start))-24;hi=float(rhythm.beat_at(end))+24
        indices=np.flatnonzero((rhythm.beats>=lo)&(rhythm.beats<=hi)).tolist()
        if not all(acoustic.supported(i) for i in indices):raise ValueError('Audio support gap in planning context')
        rview,aview=ContextView(rhythm,indices),ContextView(acoustic,indices)
        output=args.output/str(record['id']);output.mkdir()
        generated={};decisions={}
        for name,context,trained in (('embedding',aview,model),('rhythm_control',rview,control),
                                      ('pretrained',rview,initial),('next_anchor',rview,None)):
            predicted=probabilities(context,trained) if trained else nearest_baseline(context)
            events,solved=plan_flow(context,predicted,hierarchical=True)
            generated[name]=greedy_lanes(events,chart['keys'])
            if Counter(n.start for n in events)!=Counter(n.start for i in indices for n in rhythm.notes[i]):
                raise ValueError('Fixed attack counts changed')
            path=output/(name+'-events.json');path.write_text(json.dumps([n.data() for n in generated[name]]),encoding='utf-8')
            decisions[name]=dict(solver=solved,events_sha256=sha(path),validation=validate(generated[name],chart['keys']))
        frozen=dict(id=record['id'],group=record['group'],keys=chart['keys'],bounds=[start,end],
                    planning_bounds=[int(rview.times[0]),int(rview.times[-1])],variants=decisions)
        (output/'decisions.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
        result=frozen|dict(scores={name:head_scoped_comparison(chart['notes'],events,start,end) for name,events in generated.items()})
        (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');rows.append(result)
        print(json.dumps(dict(id=record['id'],bounds=[start,end],f1={n:s['complete_holds']['f1'] for n,s in result['scores'].items()})),flush=True)
    report=dict(dataset_sha256=sha(args.dataset/'manifest.json'),model_sha256=sha(args.model),control_sha256=sha(args.control),
        pretrained_sha256=sha(args.pretrained),runner_sha256=sha(__file__),flow_sha256=sha('manialab/gesture_flow.py'),
        split=args.split,hierarchical=True,ln_bias=0.,rows=rows,exclusions=excluded,enabled_in_application=False,
        notice='Supplied authored attacks/Timing/ratio; same examples and song splits; not automatic transcription or musical acceptance.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
