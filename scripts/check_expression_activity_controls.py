"""Apply section policies to an additional real five-track ensemble.

Fixed 10-second cuts exercise section boundaries; they are not musical labels.
The original separate recordings define source identities. No annotations,
new neural inference or guessed instrument identities enter the proposal.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.arrange import parameters, assign_lanes
from manialab.domain import UNOBSERVED_RELEASES, note_from, validate, export_osu, parse_osu
from manialab.expression import propose_expression, expression_evidence, select_expression
from scripts.tsumugi_candidate_adapter import adapt_case


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)
    def save(name, value):
        with (args.output/name).open('x', encoding='utf-8') as file:
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
    root = ROOT/'test-data/tsumugi-urmp-comparison-v1'
    source = ROOT/'test-data/other-instrument-model-survey-2026-11-10/tsumugi'
    names = ['39-1-vn', '39-2-vn', '39-3-va', '39-4-sax', '39-5-db']
    files = ['manialab/expression.py', 'manialab/expression_activity.py', 'manialab/expression_articulation.py',
             'manialab/arrange.py', 'scripts/check_expression_activity_controls.py']
    identities = {path:sha(ROOT/path) for path in files}
    save('execution-plan.json', dict(cases=names, source_files=identities, cuts_ms=[0, 10000, 20000, 30000, 40000],
        neural_inference=False, musical_section_labels=False, reference_annotations_read=False,
        source_semantics='separate_recordings_not_model_class_predictions', timing_quantization=False))
    all_notes = []
    for i, name in enumerate(names):
        bridge = adapt_case(root/name, source, voice='other', allow_predicted_releases=True)
        save(name+'-bridge.json', bridge)
        for row in bridge['candidates']:
            # Do not fold real independent recordings together just because
            # two instruments share one predicted broad model class.
            note = copy.deepcopy(row)
            note['voice'] = f'source-{i+1}'
            note['id'] = f'{i+1}:{note["id"]}'
            all_notes.append(note)
    a = dict(duration=40, candidates=sorted(all_notes, key=lambda n:(n['start'], n['id'])),
             structure=dict(segments=[dict(start_ms=t, end_ms=t+10000) for t in range(0, 40000, 10000)]))
    saved = copy.deepcopy(a)
    settings = parameters({'keys':10, 'density':7})
    plan = propose_expression(a, settings)
    evidence, metadata, trace = expression_evidence(a, plan, settings)
    assert len(evidence) == len(all_notes)
    raw = {n['id']:note_from(n) for n in all_notes}
    for n in evidence:
        original = raw[n.id]
        assert (n.start, n.pitch, n.instrument_id) == (original.start, original.pitch, original.instrument_id)
        assert n.end in (n.start, original.end)
        assert not n.hold or original.release_reason not in UNOBSERVED_RELEASES
    selected, selection = select_expression(evidence, metadata, trace, settings, 500)
    placed = assign_lanes(selected, settings, beat_ms=500)
    structural = validate(placed, 10)
    assert structural['valid'] and a == saved
    expected = {n.id:(n.start, n.end) for n in selected}
    assert {n.id:(n.start, n.end) for n in placed} == expected
    # Arbitrary test clock is metadata only. No beat alignment is claimed.
    text = export_osu(placed, 10, ROOT/'vendor/mapperatorinator/template.osu',
        title='URMP source policy control', version='Unquantized diagnostic', audio='39-mix.wav')
    (args.output/'ensemble.osu').write_text(text, encoding='utf-8')
    parsed = parse_osu(text)
    assert sorted((n.start, n.end, n.lane) for n in parsed['notes']) == sorted((n.start, n.end, n.lane) for n in placed)
    save('analysis.json', a);save('plan.json', plan);save('selection.json', selection)
    save('notes.json', [n.data() for n in placed])
    assert identities == {path:sha(ROOT/path) for path in files}
    report = dict(source_events=len(all_notes), expressed_events=len(evidence), selected_events=len(placed),
        source_counts=dict(Counter(n['voice'] for n in all_notes)), selected_counts=dict(Counter(n.voice for n in placed)),
        main_voices=[s['main_voice'] for s in plan['sections']],
        carried_by_section=[{v:p['activity']['carried_candidate_count'] for v,p in s['voices'].items()} for s in plan['sections']],
        solver=selection['selection_solver'], structural=structural,
        source_inputs=[dict(name=name, prediction_report_sha256=sha(root/name/'report.json'),
                            bridge_sha256=sha(args.output/(name+'-bridge.json'))) for name in names],
        timing_quantization=False, musical_acceptance=False, reference_annotations_read=False,
        execution_plan_sha256=sha(args.output/'execution-plan.json'))
    save('report.json', report)
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':main()
