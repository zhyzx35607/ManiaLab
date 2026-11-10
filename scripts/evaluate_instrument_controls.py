"""Reuse frozen transcription controls; never tune on Sakura or its Timing."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import Note,note_from,parse_osu
from manialab.timing import chart_clock_notes
from scripts.evaluate_audio import match_attacks
from scripts.evaluate_synthetic_mix import pitched_match
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/yourmt3/controls'))
    cli.add_argument('--output',type=Path,default=Path('test-data/yourmt3/control-comparison.json'))
    args=cli.parse_args()
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
    cases=read('test-data/piano-release-probe/cases.json')
    previous=read('docs/piano-offset-research.json')
    rows={}
    for case in cases:
        name=case['name'];folder=args.root/name
        report=read(folder/'report.json');identity=report['identity']
        if (identity['start_ms']!=0 or identity['seconds']!=case['duration_seconds'] or
            identity['sources']!={case['audio']:sha(case['audio'])}):
            raise ValueError('Control audio/interval changed')
        raw=read(folder/'notes.json')
        notes=[Note(round(n['onset']*1000),round(n['offset']*1000),pitch=n['pitch'],source='pitch')
            for n in raw if not n['is_drum'] and 0<=n['onset']<case['duration_seconds']]
        row=dict(identity=identity,decoding_errors=report['decoding_errors'],programs=report['by_program'])
        if name=='controlled-mix':
            reference=[note_from(n) for n in read('test-data/synthetic-mix/ground-truth.json')]
            row['previous']=previous['synthetic_mixture']
            row['yourmt3']=dict(pitched_onsets=pitched_match(notes,reference),
                pitched_intervals=pitched_match(notes,reference,True),
                matched_over_2s=pitched_match(notes,[n for n in reference if n.end-n.start>=2000],True)['matched'])
        elif name=='repeated-chords':
            reference=[Note(round((.115+group+i*.5)*1000),round((.585+group+i*.5)*1000),pitch=pitch)
                for group in (0,5) for i in range(8) for pitch in (60,64,67)]
            row['previous']=previous['repeated_synthetic_chords']
            row['yourmt3']=dict(pitched_onsets=pitched_match(notes,reference),
                pitched_intervals=pitched_match(notes,reference,True),
                false_long_on_true_pitches=sum(n.pitch in (60,64,67) and n.end-n.start>=800 for n in notes))
        else:
            reference=parse_osu(Path('test-data/audio-extension2-10k/5317787/reference.osu').read_text(encoding='utf-8-sig'))['notes']
            reference=[n for n in reference if 0<=n.start<=n.end<40000]
            notes=[n for n in chart_clock_notes(notes,22.5) if 0<=n.start<=n.end<40000]
            row['previous']=previous['natural_piano_excerpt']['modes']
            row['yourmt3']=dict(onset_rows=match_attacks([n.start for n in notes],[n.start for n in reference],40),
                candidates=len(notes),notice='Tap-only chart: acoustic offsets cannot be evaluated.')
        rows[name]=row
    result=dict(rows=rows,enabled_in_application=False,
        limits=['Frozen controls previously used for piano research, no YourMT3 threshold fitting.',
            'Synthetic harmonic tones are not natural-instrument recordings.',
            'All non-drum programs retained; no best-scoring family selection.',
            'Natural-piano chart labels authored attack choices, not complete MIDI truth.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:r['yourmt3'] for k,r in rows.items()}),flush=True)


if __name__=='__main__':main()
