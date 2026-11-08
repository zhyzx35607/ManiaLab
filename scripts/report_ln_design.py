"""Audit cached fixed-rhythm outputs and publish only measurements and map hashes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manialab.domain import invariant,note_from,parse_osu,validate
from manialab.ln_patterns import analyze_ln_patterns


def summarize(path,corpus,expected,modes=None):
    value=json.loads(path.read_text(encoding='utf-8'))
    if len(value['rows'])!=expected:raise ValueError(f'{path.name}: {len(value["rows"])} / {expected}, still incomplete')
    for row in value['rows']:
        if modes:row['results']={k:v for k,v in row['results'].items() if k in modes}
        raw=(corpus/'maps'/f"{row['beatmap_id']}.osu").read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        original=parse_osu(raw.decode('utf-8-sig'))
        for name,result in row['results'].items():
            if not result['passed']:continue
            cached=ROOT/'test-data/ln-design/results'/f"{row['beatmap_id']}-{name}.json"
            raw_notes=cached.read_bytes();notes=[note_from(n) for n in json.loads(raw_notes)]
            assert invariant(notes)==invariant(original['notes'])
            assert validate(notes,row['keys'])['valid']
            result['output_sha256']=hashlib.sha256(raw_notes).hexdigest()
            result['pattern']=analyze_ln_patterns(notes,row['keys'],timing_points=original['timing'])
            result['pattern'].pop('locations');result['pattern'].pop('sections')
    totals={}
    for mode in value['rows'][0]['results']:
        good=[r['results'][mode] for r in value['rows'] if r['results'][mode]['passed']]
        counts=Counter()
        for r in good:counts.update(r['pattern']['counts'])
        totals[mode]=dict(passed=len(good),counts=dict(counts),
                         short_jacks=sum(r['layout']['short_jacks'] for r in good),
                         release_repeats=sum(r['layout']['release_repeats'] for r in good),
                         seconds=sum(r['seconds'] for r in good))
    gates={'structural_and_identity':all(v['passed']==expected for v in totals.values())}
    if 'baseline' in totals and 'current' in totals:
        for metric in ['short_jacks','release_repeats']:
            gates[metric]=totals['current'][metric]<=totals['baseline'][metric]+max(5,totals['baseline'][metric]*.05)
        gates['hybrid_control']=totals['hybrid']['counts'].get('same_hand_hybrid_taps',0)>totals['separate']['counts'].get('same_hand_hybrid_taps',0)
        gates['inverse_control']=totals['inverse']['counts'].get('inverse_heads',0)>totals['current']['counts'].get('inverse_heads',0)
    value.update(totals=totals,gates=gates,
        measurement_source_sha256=hashlib.sha256((ROOT/'manialab/ln_patterns.py').read_bytes()).hexdigest(),
        measurement_notice='Patterns recomputed from exact saved generated notes, with new release-only and asynchronous-action descriptors. Generation source hashes remain those captured before each run. Runtime includes other local workloads.')
    return value


if __name__=='__main__':
    configs=[('existing',ROOT/'test-data/ln-design/evaluation.json',ROOT/'test-data/corpus',14,None),
             ('ln_reference',ROOT/'test-data/ln-design/new-reference-evaluation.json',ROOT/'test-data/ln-reference',15,
              ('baseline','current','separate','hybrid','inverse')),
             ('release_control',ROOT/'test-data/ln-design/new-reference-evaluation.json',ROOT/'test-data/ln-reference',15,
              ('release_free','release_clear'))]
    report={name:summarize(path,corpus,n,modes) for name,path,corpus,n,modes in configs}
    output=ROOT/'docs/ln-design-validation.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({name:{'gates':r['gates'],'totals':r['totals']} for name,r in report.items()},ensure_ascii=False))
    if not all(all(r['gates'].values()) for r in report.values()):raise SystemExit('A fixed acceptance gate did not pass; inspect per-chart changes before release.')
