"""Fixed-rhythm LN layout comparison on the existing 14-chart development set.

All source object identities, timestamps, types and raw sound fields must stay
unchanged. These are known development charts, not an unseen player evaluation.
Run before tuning; keep every row, including failures and regressions.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.arrange import assign_lanes, layout_metrics, parameters
from manialab.domain import invariant, parse_osu, validate
from manialab.layout import search_layout
from manialab.ln_patterns import analyze_ln_patterns


def run(corpus, output, baseline_ref, selection_path=None, modes=None):
    raw = subprocess.run(['git','show',f'{baseline_ref}:manialab/layout.py'], cwd=ROOT,
                         check=True, capture_output=True).stdout
    baseline_path = ROOT/'test-data/ln-design/baseline_layout.py'
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_bytes(raw)
    spec = importlib.util.spec_from_file_location('manialab.baseline_layout', baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    selection = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path else json.loads((ROOT/'docs/full-remap-validation.json').read_text(encoding='utf-8'))['rows']
    report = dict(protocol=__doc__, baseline_ref=baseline_ref,
        baseline_layout_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ('manialab/layout.py','manialab/ln_patterns.py','manialab/arrange.py')},
        acceptance='Zero non-lane changes and structural errors. Default tight-repress and fast-repeat aggregate must not increase by >max(5,5%). Hybrid/inverse controls must change their intended metric on eligible material. Report all split-release changes and runtime. This does not measure musical quality.',
        rows=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    for entry in selection:
        raw = (corpus/'maps'/f"{entry['beatmap_id']}.osu").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        parsed = parse_osu(raw.decode('utf-8-sig'))
        notes, timing, keys = parsed['notes'], parsed['timing'], parsed['keys']
        row = {k: entry[k] for k in ('beatmap_id','keys','sha256','layout_membership')}
        row['results'] = {}
        cases=[('baseline',baseline.search_layout,{}),('current',search_layout,{}),
                                   ('separate',search_layout,{'ln_coordination':0}),
                                   ('hybrid',search_layout,{'ln_coordination':1}),
                                   ('inverse',search_layout,{'ln_inverse':1}),
                                   ('release_free',search_layout,{'ln_release':0}),
                                   ('release_clear',search_layout,{'ln_release':1})]
        for name, fn, overrides in cases:
            if name not in (modes or ('baseline','current','separate','hybrid','inverse')):continue
            started = time.perf_counter()
            p = parameters({'keys':keys}|overrides)
            try:
                fallback = False
                try:
                    result = fn(notes,p,timing_points=timing)
                except ValueError:
                    fallback = True
                    result = assign_lanes(notes,p|{'layout_engine':'classic'},timing_points=timing)
                assert invariant(result)==invariant(notes)
                structural = validate(result,keys)
                assert structural['valid'], structural
                saved=ROOT/'test-data/ln-design/results'/f"{entry['beatmap_id']}-{name}.json"
                saved.parent.mkdir(parents=True,exist_ok=True)
                saved.write_text(json.dumps([n.data() for n in result]),encoding='utf-8')
                pattern = analyze_ln_patterns(result,keys,timing_points=timing)
                pattern.pop('locations'); pattern.pop('sections')
                row['results'][name] = dict(passed=True, fallback=fallback,
                    pattern=pattern, layout=layout_metrics(result,keys,timing_points=timing),
                    seconds=round(time.perf_counter()-started,3))
            except Exception as error:
                row['results'][name] = dict(passed=False, error=str(error),seconds=round(time.perf_counter()-started,3))
            print(json.dumps(dict(beatmap_id=entry['beatmap_id'],mode=name,**row['results'][name])),flush=True)
        report['rows'].append(row)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(r['passed'] for row in report['rows'] for r in row['results'].values())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=ROOT/'test-data/corpus')
    parser.add_argument('--output',type=Path,default=ROOT/'test-data/ln-design/evaluation.json')
    parser.add_argument('--baseline-ref',default='916cc67')
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--modes',nargs='+',choices=['baseline','current','separate','hybrid','inverse','release_free','release_clear'])
    args=parser.parse_args()
    run(args.corpus,args.output,args.baseline_ref,args.selection,args.modes)
