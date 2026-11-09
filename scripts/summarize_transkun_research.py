"""Publish aggregate control/music results without audio or note-level labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.evaluate_piano_acoustics import counts


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified_report(folder):
    report=read(folder/'report.json')
    for name,digest in report['artifacts_sha256'].items():
        if sha256(folder/name)!=digest:
            raise ValueError('Evidence changed: '+name)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--music',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    controls=verified_report(args.controls)
    music=verified_report(args.music)
    if (controls['unique_control_inferences']!=91 or not controls['wrapper_verification']['identical']
            or len(music['cases'])!=12):
        raise ValueError('Complete both frozen runs')
    control_rows=[read(args.controls/(r['name']+'-result.json')) for r in controls['cases']]
    music_rows=[read(args.music/(r['name']+'-result.json')) for r in music['cases']]
    totals={engine:{key:sum(r[field][key] for r in control_rows)
        for key in ('expected','predicted','matched','missing','extra')}
        for engine,field in [('kong','baseline_score'),('transkun','score')]}
    conditions={}
    for row in control_rows:
        label='quiet_single' if row['gain']==.25 else row['condition']
        group=conditions.setdefault(label,dict(cases=0,kong=Counter(),transkun=Counter()))
        group['cases']+=1
        for engine,field in [('kong','baseline_score'),('transkun','score')]:
            group[engine].update({key:row[field][key] for key in ('expected','matched','missing','extra')})
    public_rows=[]
    for row in music_rows:
        if row['kind']!='smd':
            continue
        public_rows.append(dict(name=row['name'],elapsed_seconds=row['elapsed_seconds'],
            onsets={label:{engine:row['comparison'][label][engine] for engine in ('kong','transkun')}
                    for label in ('50ms','40ms')},
            recovered={label:len(row['comparison'][label]['recovered_ref_indices']) for label in ('50ms','40ms')},
            new_missing={label:len(row['comparison'][label]['new_missing_ref_indices']) for label in ('50ms','40ms')},
            interval_metrics={engine:{semantics:{label:row['metrics'][engine][semantics][label]['intervals']
                for label in ('50ms','40ms')} for semantics in ('key','sustain')}
                for engine in ('kong','transkun')},
            full_piano_events=len(row['trace']['piano_events']),
            pedal_events=len(row['trace']['pedal_events']),
            piano_onsets_outside_input=row['trace']['piano_onsets_outside_input']))
    intervals={}
    for engine in ('kong','transkun'):
        intervals[engine]={}
        for semantics in ('key','sustain'):
            intervals[engine][semantics]={}
            for tolerance in ('50ms','40ms'):
                rows=[r['interval_metrics'][engine][semantics][tolerance] for r in public_rows]
                intervals[engine][semantics][tolerance]=counts(*(
                    sum(r[k] for r in rows) for k in ('matched','reference','predicted')))
    result=dict(source=dict(controls_report_sha256=sha256(args.controls/'report.json'),
                            music_report_sha256=sha256(args.music/'report.json'),
                            summarizer_sha256=sha256(Path(__file__)),
                            model=music['identity']['model'],protocols={
                                'controls':controls['identity']['code']['protocol'],
                                'music':music['identity']['files']['protocol']}),
        controls=dict(cases=91,totals=totals,conditions=conditions,
            new_misses=sum(len(set(r['score']['missing_ref_indices'])-set(r['baseline_score']['missing_ref_indices']))
                          for r in control_rows),
            recovered=sum(len(set(r['baseline_score']['missing_ref_indices'])-set(r['score']['missing_ref_indices']))
                          for r in control_rows),
            checks=controls['summary']['control_checks'],
            seconds=sum(r['elapsed_seconds'] for r in control_rows),
            wrapper_verification_passed=True),
        music=dict(summary=music['summary'],intervals=intervals,recordings=public_rows,
                   seconds=sum(r['elapsed_seconds'] for r in music_rows)),
        nova=music['nova'],training_performed=False,application_enabled=False,
        musical_acceptance=False,direct_listening_performed=False,
        interpretation='Historical real-music comparison and isolated-source controls; no replacement or LN acceptance.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise ValueError('Preserve prior summary')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(controls=totals,music=music['summary']['models']['50ms'],nova=music['nova'])),flush=True)


if __name__=='__main__':
    main()
