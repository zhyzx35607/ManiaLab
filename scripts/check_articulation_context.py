"""Compare articulation decisions before/after a section-context fix."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.arrange import parameters
from manialab.expression import expression_evidence, propose_expression
from scripts.build_provenance import sha256


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--baseline', type=Path)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous comparisons')
    path = ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json'
    frozen = json.loads((ROOT/'test-data/piano-svsep/probe-v3/report.json').read_text(encoding='utf-8'))
    if sha256(path) != frozen['identity']['analysis_sha256']:
        raise ValueError('Frozen audio candidates changed')
    analysis = json.loads(path.read_text(encoding='utf-8'))
    settings = parameters({'keys':10,'ln_ratio':.55,'vocals_weight':0})
    plan = propose_expression(analysis, settings)
    for section in plan['sections']:
        section['voices']['piano']['mode'] = 'articulation'
    before = copy.deepcopy(analysis)
    result, _, trace = expression_evidence(analysis, plan, settings)
    assert before == analysis
    args.output.mkdir(parents=True)
    save = lambda name,value:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    identities = {'analysis_sha256':sha256(path), 'settings':settings, 'plan':plan}
    save('input.json',identities)
    files = [ROOT/'manialab/expression.py',ROOT/'manialab/expression_articulation.py',Path(__file__)]
    code = {p.name:sha256(p) for p in files}
    for p in files:
        shutil.copyfile(p,args.output/(p.stem+'-snapshot.py'))
    piano = [n.data() for n in result if n.voice=='piano']
    save('piano-notes.json',piano)
    save('piano-decisions.json',{key:row for key,row in trace.items() if row['voice']=='piano'})
    report = dict(input_sha256=sha256(args.output/'input.json'),code_sha256=code,
        piano_events=len(piano),piano_holds=sum(n['end']>n['start'] for n in piano),
        reasons=dict(Counter(row['reason'] for row in trace.values() if row['voice']=='piano')),
        physical_candidates_unchanged=True, model_rerun=False, musical_acceptance=False)
    if args.baseline:
        old_input=json.loads((args.baseline/'input.json').read_text(encoding='utf-8'))
        # The new evidence signature includes audio duration. The complete
        # analysis hash above still requires identical bytes and duration;
        # permit only this signature-schema change, not a policy change.
        old_plan=dict(old_input['plan']);now_plan=dict(identities['plan'])
        old_plan.pop('evidence_signature');now_plan.pop('evidence_signature')
        if (old_input['analysis_sha256']!=identities['analysis_sha256']
                or old_input['settings']!=settings or old_plan!=now_plan):
            raise ValueError('Different evidence or expression plan')
        previous=json.loads((args.baseline/'piano-notes.json').read_text(encoding='utf-8'))
        old={row['id']:row for row in previous};now={row['id']:row for row in piano}
        assert set(old)==set(now)
        changes=[]
        for key,row in now.items():
            prior=old[key]
            assert all(row[field]==prior[field] for field in row if field not in ('end','phrase'))
            if row['end']!=prior['end']:
                changes.append(dict(id=key,start=row['start'],old_end=prior['end'],new_end=row['end']))
        save('changed-decisions.json',changes)
        report.update(baseline_report_sha256=sha256(args.baseline/'report.json'),
            baseline_holds=sum(n['end']>n['start'] for n in previous),
            changed=len(changes),hold_to_tap=sum(n['new_end']==n['start'] for n in changes),
            tap_to_hold=sum(n['old_end']==n['start'] for n in changes),
            selected_attacks_preserved=True)
    report['artifacts_sha256']={p.name:sha256(p) for p in args.output.iterdir()}
    save('report.json',report)
    print(json.dumps({key:value for key,value in report.items() if key!='artifacts_sha256'}),flush=True)


if __name__=='__main__':
    main()
