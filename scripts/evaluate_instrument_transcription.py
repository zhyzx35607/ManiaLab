"""Score frozen instrument probes after inference; no reference-based filtering."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import Note,parse_osu,note_from
from scripts.probe_mapper_adapter import event_similarity,sha
from scripts.probe_sakura_sources import match_source

DRUMS={'kick':{35,36},'snare':{38,40},'hi_hat':{22,26,42,44,46}}
DRUM_DIAGNOSTICS={'side_stick_only':({37,2},{1,2}),
    'snare_and_side_stick':({37,2,38,40},{1,2}),
    'ride_only':({51,53,59},{3}),
    'tom_only':({41,43,45,47,48,50},{3}),
    'hi_hat_ride_tom':({22,26,42,44,46,51,53,59,41,43,45,47,48,50},{3})}
FAMILIES={'piano':range(0,8),'organ':range(16,24),'guitar':range(24,32),
          'bass':range(32,40),'strings':range(40,56),'brass':range(56,64),
          'reed_pipe':range(64,80),'synth':range(80,100),'vocals':range(100,102)}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--runs',nargs='+',required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();root=args.root
    reference=parse_osu((root/'reference.osu').read_text(encoding='utf-8-sig'))['notes']
    baseline=json.loads((root/'provided-pipeline/analysis.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if sha(root/'reference.osu')!=manifest['charts']['reference']['chart_sha256']:
        raise ValueError('Reference identity changed')
    rows=[]
    for name in args.runs:
        folder=root/name;report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        raw=json.loads((folder/'notes.json').read_text(encoding='utf-8'))
        identity=report['identity'];voices={Path(p).parent.name for p in identity['sources']}
        if voices=={'drums'}:role,lanes='drums',range(4)
        elif voices=={'piano'}:role,lanes='piano',range(4,6)
        elif voices=={'other','bass','guitar'}:role,lanes='other',range(6,10)
        else:raise ValueError('No declared author columns for these inputs')
        for path,digest in identity['sources'].items():
            if sha(path)!=digest:raise ValueError('Probe source changed')
        a=identity['start_ms'];b=a+identity['seconds']*1000
        human=[n for n in reference if n.lane in lanes and a<=n.start<b]
        result=dict(run=name,role=role,identity=identity,decoding_errors=report['decoding_errors'],
            predicted_program_counts=report['by_program'],elapsed_seconds=report['elapsed_seconds'],comparisons={})
        for shift in (0.,22.5):
            selected=[n for n in raw if a<=n['onset']*1000-shift<b]
            convert=lambda ns:[Note(round(n['onset']*1000),max(round(n['onset']*1000),round(n['offset']*1000)),
                pitch=n['pitch']) for n in ns]
            basic=[note_from(n) for n in baseline['candidates'] if n['voice'] in voices and a<=n['start']-shift<b]
            modes={'all_model_events':selected,
                'source_compatible_events':[n for n in selected if n['is_drum']] if role=='drums' else
                    [n for n in selected if not n['is_drum'] and n['program'] not in (100,101)]}
            comparison={mode:match_source(human,convert(notes),shift) for mode,notes in modes.items()}
            comparison['application_detector']=match_source(human,basic,shift)
            if role=='drums':
                comparison['drum_classes']={kind:event_similarity(
                    [(t,) for t in sorted({n.start for n in human if n.lane in
                        ({0} if kind=='kick' else {1,2} if kind=='snare' else {3})})],
                    [(t,) for t in sorted({round(n['onset']*1000)-shift for n in selected
                        if n['is_drum'] and n['pitch'] in pitches})],[40])
                    for kind,pitches in DRUMS.items()}
                comparison['diagnostic_drum_mappings']={name:event_similarity(
                    [(t,) for t in sorted({n.start for n in human if n.lane in columns})],
                    [(t,) for t in sorted({round(n['onset']*1000)-shift for n in selected
                        if n['is_drum'] and n['pitch'] in pitches})],[40])
                    for name,(pitches,columns) in DRUM_DIAGNOSTICS.items()}
            else:
                comparison['diagnostic_families']={family:match_source(human,
                    convert([n for n in selected if not n['is_drum'] and n['program'] in programs]),shift)['onset_groups']
                    for family,programs in FAMILIES.items()
                    if any(not n['is_drum'] and n['program'] in programs for n in selected)}
            result['comparisons'][str(shift)]=comparison
        result['segments']=[]
        for section in manifest['segments']:
            left,right=max(a,section['start_ms']),min(b,section['end_ms'])
            if right<=left:continue
            shift=22.5
            selected=[n for n in raw if left<=n['onset']*1000-shift<right]
            compatible=[n for n in selected if n['is_drum']] if role=='drums' else [
                n for n in selected if not n['is_drum'] and n['program'] not in (100,101)]
            target=[n for n in human if left<=n.start<right]
            basic=[note_from(n) for n in baseline['candidates'] if n['voice'] in voices
                and left<=n['start']-shift<right]
            result['segments'].append(dict(start_ms=left,end_ms=right,
                source_compatible=match_source(target,convert(compatible),shift)['onset_groups'],
                application=match_source(target,basic,shift)['onset_groups']))
        rows.append(result)
    output=dict(reference_sha256=sha(root/'reference.osu'),rows=rows,enabled_in_application=False,
        limits=['Reference labels are authored attack/layer choices, not exhaustive MIDI truth.',
            'Source-compatible excludes drums/vocals from accompaniment and non-drums from drums based on supplied source roles, not scored thresholds.',
            'Family breakdowns are diagnostic; the highest-scoring family is not selected as output.',
            'Column 4 also contains tom/other exceptions; hi-hat label comparison is incomplete.',
            'Snare/side-stick and hi-hat/ride/tom diagnostics are all reported; no mapping is selected by score.',
            'Author section boundaries are used only to summarize frozen predictions, never as model inputs.',
            'Fixed 22.5ms chart lead is predeclared, not fit to this song. No grid snapping.',
            'Single development song; instrument names and acoustic releases have no complete ground truth.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps([dict(run=r['run'],source_compatible=r['comparisons']['22.5']['source_compatible_events']['onset_groups']) for r in rows]),flush=True)


if __name__=='__main__':main()
