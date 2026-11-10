"""Compare paired-audio expression to matched rhythm-only and simple controls."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, validate
from manialab.release_candidates import ReleaseContext
from manialab.stem_release_evidence import StemReleaseContext
from manialab.anchor_gestures import probabilities
from manialab.gesture_flow import plan_flow
from scripts.paired_stem_inputs import load_observations
from scripts.probe_mapper_adapter import sha
from scripts.evaluate_anchor_gestures import greedy_lanes, head_scoped_comparison, candidate_ceiling, nearest_baseline


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','model','control','pretrained','output'):cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--split',choices=('validation','test'),required=True)
    cli.add_argument('--hierarchical',action='store_true')
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier comparisons')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(args.dataset/'manifest.json'); model=read(args.model); control=read(args.control); initial=read(args.pretrained)
    if (model['dataset_sha256']!=sha(args.dataset/'manifest.json')
            or control['parent_dataset_sha256']!=sha(args.dataset/'manifest.json')
            or initial['dataset_sha256']!=manifest['pretraining_dataset_sha256']):
        raise ValueError('Models do not correspond to the same paired examples and song splits')
    if (model['stem_feature_sha256']!=sha('manialab/stem_release_evidence.py')
            or manifest['loader_sha256']!=sha('scripts/paired_stem_inputs.py')):
        raise ValueError('Frozen acoustic implementation changed')
    for trained in (model,control,initial):
        if trained['release_feature_sha256']!=sha('manialab/release_candidates.py'):
            raise ValueError('Frozen release features changed')
    args.output.mkdir(parents=True); rows=[]
    for record in manifest['records']:
        if record['split']!=args.split:continue
        chart_path=Path(record['chart'])
        if sha(chart_path)!=record['chart_sha256']:raise ValueError('Frozen target bytes changed')
        chart=parse_osu(chart_path.read_text(encoding='utf-8-sig'))
        ratio=sum(n.hold for n in chart['notes'])/len(chart['notes'])
        notes=[replace(n,end=n.start,lane=0,voice='mix',raw='') for n in chart['notes']]
        full=ReleaseContext(notes,chart['keys'],chart['timing'],ratio)
        first,last=record['score_bounds'];start=max(first,(first+last-32000)//2);end=min(last,start+32000)
        lo=max(0,int(full.times.searchsorted(start))-1);hi=min(len(full.times)-1,int(full.times.searchsorted(end)))
        left=max(0,int(full.beats.searchsorted(full.beats[lo]-24))-4)
        right=min(len(full.times)-1,int(full.beats.searchsorted(full.beats[hi]+24))+4)
        notes=[n for n in notes if full.times[left]<=n.start<=full.times[right]]
        rhythm=ReleaseContext(notes,chart['keys'],chart['timing'],ratio)
        evidence=load_observations(Path(record['observations']),record['identity'])
        acoustic=StemReleaseContext(notes,chart['keys'],chart['timing'],ratio,evidence,record['identity']['audio_start_ms'])
        output=args.output/str(record['beatmap_id']);output.mkdir()
        generated={};reports={}
        variants=(('acoustic',acoustic,model),('rhythm_control',rhythm,control),('pretrained',rhythm,initial),('next_anchor',rhythm,None))
        for name,context,trained in variants:
            predicted=probabilities(context,trained) if trained else nearest_baseline(context)
            events,solved=plan_flow(context,predicted,hierarchical=args.hierarchical)
            generated[name]=greedy_lanes(events,chart['keys'])
            path=output/(name+'-events.json');path.write_text(json.dumps([n.data() for n in generated[name]]),encoding='utf-8')
            if Counter(n.start for n in events)!=Counter(n.start for n in notes):raise ValueError('Attack counts changed')
            reports[name]=dict(solver=solved,events_sha256=sha(path),validation=validate(generated[name],chart['keys']))
        # The target chart was parsed to obtain onsets and the supplied LN
        # condition; its tails do not enter predictions or window selection.
        frozen=dict(id=record['beatmap_id'],group=record['group'],keys=chart['keys'],bounds=[start,end],variants=reports)
        (output/'decisions.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
        result=frozen|dict(scores={name:head_scoped_comparison(chart['notes'],events,start,end) for name,events in generated.items()},
                           candidate_coverage=candidate_ceiling(rhythm,chart['notes'],start,end))
        (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');rows.append(result)
        print(json.dumps(dict(id=record['beatmap_id'],f1={name:score['complete_holds']['f1'] for name,score in result['scores'].items()})),flush=True)
    report=dict(dataset_sha256=sha(args.dataset/'manifest.json'),model_sha256=sha(args.model),control_sha256=sha(args.control),
        pretrained_sha256=sha(args.pretrained),runner_sha256=sha(__file__),flow_sha256=sha('manialab/gesture_flow.py'),
        split=args.split,hierarchical=args.hierarchical,ln_bias=0.,rows=rows,enabled_in_application=False,
        notice='Fixed authored attacks and supplied Timing/LN condition; paired-audio development comparison, not transcription or source ownership.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
