"""Compare physical drum classes to author choices without treating them as exhaustive MIDI truth."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from scripts.probe_mapper_adapter import event_similarity,sha
from scripts.research_source_attack_classes import drum_event_counts


def score(reference,events,shift,start,end):
    report={}
    for kind in ('kick','snare','hi-hat'):
        human=[(t,) for t,c in reference if c==kind and start<=t<end]
        predicted=[(e['start_ms']-shift,) for e in events if e['kind']==kind and start<=e['start_ms']-shift<end]
        report[kind]=event_similarity(human,predicted,[40])
    # Author says lane 4 also contains occasional toms: show a separate union
    # diagnostic, never relabel every lane-4 note as a definite hi-hat.
    human=[(t,) for t,c in reference if c=='hi-hat' and start<=t<end]
    predicted=[(t,) for t in sorted({e['start_ms']-shift for e in events
        if e['kind'] in ('hi-hat','tom') and start<=e['start_ms']-shift<end})]
    report['lane4_hihat_or_tom_diagnostic']=event_similarity(human,predicted,[40])
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--probe',type=Path,required=True)
    args=cli.parse_args()
    reference_path=args.root/'reference.osu'
    reference=list(drum_event_counts(parse_osu(reference_path.read_text(encoding='utf-8-sig'))['notes']))
    events=json.loads((args.probe/'events.json').read_text())
    inference=json.loads((args.probe/'report.json').read_text())
    duration=inference['duration_seconds']*1000
    report=dict(reference_sha256=sha(reference_path),inference=inference,
        segments={name:{label:score(reference,events,shift,start,end) for label,shift in [('raw',0),('chart_lead_22_5ms',22.5)]}
            for name,start,end in [('all',0,duration),('fit_region',0,162340),
                                  ('calibration_region',162340,188286),('later_sections',188286,duration)]},
        limits=['Reference is authored selection, not exhaustive drum MIDI truth.',
                'Lane 4 primarily hi-hat, with tom and other exceptions; separate union is diagnostic only.',
                'No retraining or threshold search on this song. Same-song development comparison.',
                'Raw timing and fixed 22.5ms author lead reported separately; no beat-grid snapping.'])
    (args.probe/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['segments']['later_sections']),flush=True)


if __name__=='__main__':main()
