"""Export frozen piano stream expressions through real selection and layout."""
import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.arrange import parameters, assign_lanes, layout_metrics
from manialab.domain import UNOBSERVED_RELEASES, note_from, validate, export_osu, parse_osu
from manialab.expression import expression_evidence, select_expression, validate_expression
from manialab.expression_articulation import attack_runs
from manialab.timing import chart_clock_notes, align_candidates, grid_report
from scripts.build_provenance import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def field_key(note):
    return tuple(note[k] for k in ('id', 'start', 'end', 'pitch', 'voice', 'lane'))


def source_counts(notes):
    return dict(objects=dict(Counter(n.voice for n in notes)),
                attacks=dict(Counter(v for v, t in {(n.voice, n.start) for n in notes})),
                holds=dict(Counter(n.voice for n in notes if n.hold)))


def replace_piano_expression(evidence, metadata, trace, raw, choices, model, grouping, settings):
    """Only change eligible piano ends; retain input provenance and all roles."""
    result = []
    original_metadata = copy.deepcopy(metadata)
    omitted_before = {k: copy.deepcopy(v) for k, v in trace.items() if v['decision'] == 'omitted'}
    replaced = []
    for note in evidence:
        if note.voice != 'piano':
            result.append(note)
            continue
        original = raw[note.id]
        choice = choices[note.id]
        generated = note_from(choice['note'])
        expected = replace(original, end=generated.end)
        if canonical(expected.data()) != canonical(generated.data()):
            raise ValueError(f'Frozen stream expression changed raw evidence for {note.id}')
        if generated.end not in (original.start, original.end):
            raise ValueError(f'Invented piano endpoint for {note.id}')
        if choice['section'] != metadata[note.id]['section']:
            raise ValueError('Stream section differs from the fixed plan')
        if choice['hold'] != generated.hold:
            raise ValueError('Frozen hold flag and interval disagree')
        if generated.hold and (original.source != 'pitch' or
                               original.end-original.start < settings['min_ln'] or
                               original.release_reason in UNOBSERVED_RELEASES):
            raise ValueError('Unsupported stream hold')
        if original.strength < .35 or trace[note.id]['mode'] != 'articulation':
            raise ValueError('Stream override is outside the fixed articulation policy')
        updated = replace(note, end=generated.end, phrase=note.phrase and generated.hold)
        reason = ('supported_sustain' if generated.hold else
                  'stream_articulated_attack_run' if choice['choice']['tap'] else
                  'unobserved_release' if original.release_reason in UNOBSERVED_RELEASES else 'short_duration')
        trace[note.id].update(expression_end=updated.end, articulation=copy.deepcopy(choice['choice']),
                             stream_model=model, stream_grouping=grouping, stream_group=choice['group'], reason=reason)
        if updated.end != note.end:
            replaced.append(dict(id=note.id, previous_end=note.end, expression_end=updated.end))
        result.append(updated)
    assert metadata == original_metadata
    assert {k: v for k, v in trace.items() if v['decision'] == 'omitted'} == omitted_before
    assert [(n.id, n.start, n.pitch) for n in result] == [(n.id, n.start, n.pitch) for n in evidence]
    assert [n.data() for n in result if n.voice != 'piano'] == [n.data() for n in evidence if n.voice != 'piano']
    return result, replaced


def drum_groups(notes, metadata):
    result = defaultdict(list)
    for note in notes:
        if note.voice == 'drums':
            result[(metadata[note.id]['section'], note.start)].append(note.id)
    return {k: sorted(v) for k, v in result.items()}


def drum_delta(aligned, placed, baseline, metadata):
    available = drum_groups(aligned, metadata)
    current = drum_groups(placed, metadata)
    previous = drum_groups(baseline, metadata)
    assert set(current) <= set(available) and set(previous) <= set(available)
    def rows(keys):
        return [dict(section=section, chart_start=start, candidate_ids=available[section, start],
                     baseline_selected_ids=previous.get((section, start), []),
                     selected_ids=current.get((section, start), [])) for section, start in sorted(keys)]
    details = dict(lost=rows(set(previous)-set(current)), gained=rows(set(current)-set(previous)),
                   omitted=rows(set(available)-set(current)))
    summary = dict(expressed_groups=len(available), selected_groups=len(current),
                   baseline_selected_groups=len(previous), lost_groups=len(details['lost']),
                   gained_groups=len(details['gained']), omitted_groups=len(details['omitted']),
                   selected_objects=sum(map(len, current.values())),
                   baseline_selected_objects=sum(map(len, previous.values())))
    return details, summary


def hold_diagnostics(before, after, selected, metadata, raw, choices, assignments, grouping):
    """Explain new holds without declaring any group musically correct."""
    previous = {n.id: n for n in before}
    effective = defaultdict(list)
    for note in after:
        if note.voice == 'piano':
            effective[(metadata[note.id]['section'], choices[note.id]['group'])].append(raw[note.id])
    raw_sizes = Counter(str(row[grouping]) for row in assignments.values())
    effective_sizes = {}
    for key, notes in effective.items():
        runs, _ = attack_runs(notes)
        effective_sizes[key] = dict(notes=len(notes), attacks=sum(map(len, runs)))
    selected_ids = {n.id for n in selected if n.hold}
    rows = []
    for note in after:
        if note.voice != 'piano' or not note.hold or previous[note.id].hold:
            continue
        choice = choices[note.id]
        sizes = effective_sizes[(metadata[note.id]['section'], choice['group'])]
        rows.append(dict(id=note.id, start=note.start, end=note.end, section=metadata[note.id]['section'],
                         group=choice['group'], selected=note.id in selected_ids,
                         declared_articulation=choice['choice']['articulation'],
                         run_attack_groups=choice['choice']['attack_groups'],
                         raw_group_candidates=raw_sizes[choice['group']],
                         effective_group_candidates=sizes['notes'], effective_group_attacks=sizes['attacks']))
    def summarize(items):
        data = dict(count=len(items), declared_articulation=dict(Counter(r['declared_articulation'] for r in items)),
                    run_attack_group_histogram=dict(Counter(r['run_attack_groups'] for r in items)),
                    fewer_than_three_run_attack_groups=sum(r['run_attack_groups'] < 3 for r in items),
                    independent_with_fewer_than_three_run_attack_groups=sum(
                        r['declared_articulation'] == 'independent_sustain' and r['run_attack_groups'] < 3 for r in items))
        if grouping == 'components':
            data.update(raw_singleton_component_holds=sum(r['raw_group_candidates'] == 1 for r in items),
                        effective_single_note_component_holds=sum(r['effective_group_candidates'] == 1 for r in items),
                        effective_one_attack_component_holds=sum(r['effective_group_attacks'] == 1 for r in items),
                        effective_two_attack_component_holds=sum(r['effective_group_attacks'] == 2 for r in items),
                        effective_fewer_than_three_attack_component_holds=sum(r['effective_group_attacks'] < 3 for r in items))
        return data
    return dict(grouping=grouping, expressed=summarize(rows), selected=summarize([r for r in rows if r['selected']]),
                note='Declared articulation is a rule label, not verified acoustic sustain or musical intent.', events=rows)


def diagnose_saved(args):
    """Cheap replay of saved expression/selection only; never rerun a model."""
    folder = args.diagnose_run
    report = read(folder/'report.json')
    if sha256(folder/'runner-snapshot.py') != report['identity']['runner_sha256']:
        raise ValueError('Saved comparison runner changed')
    analysis_path = ROOT/'test-data/sakura-benchmark/instrument-expression-v1/instrument-analysis.json'
    if sha256(analysis_path) != report['identity']['analysis_sha256']:
        raise ValueError('Saved comparison analysis changed')
    raw = {n['id']: note_from(n) for n in read(analysis_path)['candidates']}
    for case in report['cases']:
        for name, digest in case['artifacts_sha256'].items():
            if sha256(folder/name) != digest:
                raise ValueError('Saved comparison artifact changed')
    baseline = read(folder/'baseline-all-10k-expression.json')
    before = [note_from(n) for n in baseline['notes']]
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    summaries = []
    for case in report['cases']:
        if case['model'] == 'baseline':
            continue
        name, model, grouping = case['name'], case['model'], case['grouping']
        probe = report['identity']['probe_only']
        source = args.frozen if probe else args.evaluation
        prefix = '' if probe else model+'-'
        choices_path = source/(prefix+f'sakura-expression-{grouping}.json')
        assignments_path = source/(prefix+'sakura-assignments.json')
        if sha256(choices_path) != case['choices_sha256']:
            raise ValueError('Saved stream choices changed')
        assignments = read(assignments_path)
        if set(assignments) != {key for key, note in raw.items() if note.voice == 'piano'}:
            raise ValueError('Saved assignments do not cover exactly the raw piano candidates')
        expressed = read(folder/(name+'-expression.json'))
        metadata = {key: dict(section=value['section']) for key, value in expressed['trace'].items()}
        diagnostic = hold_diagnostics(before, [note_from(n) for n in expressed['notes']],
                                      [note_from(n) for n in read(folder/(name+'-notes.json'))], metadata,
                                      raw, read(choices_path), assignments, grouping)
        path = args.output/(name+'-hold-diagnostics.json')
        path.write_text(json.dumps(diagnostic, indent=2), encoding='utf-8')
        summaries.append(dict(name=name, expressed=diagnostic['expressed'], selected=diagnostic['selected'],
                              choices_sha256=sha256(choices_path), assignments_sha256=sha256(assignments_path),
                              diagnostics_sha256=sha256(path)))
    result = dict(source_report_sha256=sha256(folder/'report.json'), runner_sha256=sha256(__file__),
                  diagnostic_revision=1, cases=summaries, model_rerun=False, selection_rerun=False, musical_acceptance=False)
    (args.output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--evaluation', type=Path, default=ROOT/'test-data/piano-svsep/adaptation-evaluation-v1')
    cli.add_argument('--frozen', type=Path, default=ROOT/'test-data/piano-svsep/probe-v3')
    cli.add_argument('--previous', type=Path, default=ROOT/'test-data/sakura-benchmark/articulated-expression-v2')
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--probe', action='store_true', help='Only baseline and two original-model 10K checks')
    cli.add_argument('--diagnose-run', type=Path, help='Only explain newly held notes in a saved comparison')
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous expression comparisons')
    if args.diagnose_run is not None:
        diagnose_saved(args)
        return
    analysis_path = ROOT/'test-data/sakura-benchmark/instrument-expression-v1/instrument-analysis.json'
    basic_path = ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json'
    basic_report_path = basic_path.parent/'report.json'
    previous_report = read(args.previous/'report.json')
    frozen_report = read(args.frozen/'report.json')
    if sha256(analysis_path) != previous_report['identity']['analyses_sha256']['instrument-other']:
        raise ValueError('Frozen accompaniment analysis changed')
    if sha256(basic_path) != frozen_report['identity']['analysis_sha256']:
        raise ValueError('Frozen piano analysis changed')
    analysis, basic = read(analysis_path), read(basic_path)
    piano = lambda value: {n['id']: n for n in value['candidates'] if n['voice'] == 'piano'}
    if piano(analysis) != piano(basic):
        raise ValueError('The two analyses do not contain identical piano candidates')
    if analysis['structure']['segments'] != basic['structure']['segments']:
        raise ValueError('The two analyses do not share fixed section boundaries')
    original_analysis = copy.deepcopy(analysis)
    raw = {n['id']: note_from(n) for n in analysis['candidates']}
    if len(raw) != len(analysis['candidates']):
        raise ValueError('Candidate IDs must be unique')
    clock = previous_report['identity']['generation_clock']
    if clock != dict(bpm=74.0, offset=178.0, division=4, audio_shift_ms=22.5):
        raise ValueError('Generation clock changed from the fixed protocol')
    settings_template = read(basic_report_path)['identity']['settings']
    if settings_template['min_ln'] != 240:
        raise ValueError('Expression threshold differs from frozen choices')
    plan_path = args.previous/'instrument-other-independent-10k-plan.json'
    plan = read(plan_path)
    validate_expression(plan, analysis)
    evaluation = None
    if not args.probe:
        evaluation = read(args.evaluation/'report.json')
        if not evaluation['frozen_baseline_reproduced'] or not evaluation['raw_candidates_unchanged']:
            raise ValueError('Stream evaluation did not preserve its baseline')
        if evaluation['identity']['analysis_sha256'] != sha256(basic_path):
            raise ValueError('Stream evaluation used different audio candidates')
        if sha256(args.evaluation/'runner-snapshot.py') != evaluation['identity']['runner_sha256']:
            raise ValueError('Stream evaluation snapshot changed')
    specs = [('baseline', 'all', 10), ('original', 'staff', 10), ('original', 'components', 10)]
    if not args.probe:
        specs += [('adapted', grouping, 10) for grouping in ('staff', 'components')]
        specs += [('adapted', grouping, keys) for keys in range(6, 10) for grouping in ('staff', 'components')]
    args.output.mkdir(parents=True)
    save = lambda name, value: (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    snapshots = {}
    for name in ('manialab/expression.py', 'manialab/expression_articulation.py', 'manialab/arrange.py',
                 'manialab/domain.py', 'manialab/timing.py'):
        target = args.output/(Path(name).stem+'-snapshot.py')
        shutil.copyfile(ROOT/name, target)
        snapshots[name] = sha256(target)
    identity = dict(runner_sha256=sha256(__file__), snapshots=snapshots,
                    protocol_sha256=sha256(ROOT/'docs/piano-stream-export-protocol.md'),
                    analysis_sha256=sha256(analysis_path), piano_analysis_sha256=sha256(basic_path),
                    previous_report_sha256=sha256(args.previous/'report.json'), plan_sha256=sha256(plan_path),
                    settings_report_sha256=sha256(basic_report_path), generation_clock=clock,
                    stream_report_sha256=sha256((args.frozen if args.probe else args.evaluation)/'report.json'),
                    selected_model_sha256=None if args.probe else evaluation['identity']['selected_sha256'],
                    piano_candidates_identical=True, reference_chart_read=False, network_rerun=False,
                    enabled_in_application=False, probe_only=args.probe, diagnostic_revision=1)
    save('identity.json', identity)
    cases = []
    for model, grouping, keys in specs:
        started = time.monotonic()
        name = f'{model}-{grouping}-{keys}k'
        settings = parameters(settings_template | {'keys': keys})
        evidence, metadata, trace = expression_evidence(analysis, plan, settings)
        original_evidence = [n.data() for n in evidence]
        before_notes = list(evidence)
        replaced = []
        choice_path = None
        assignments_path = None
        if model != 'baseline':
            choice_path = (args.frozen/f'sakura-expression-{grouping}.json' if args.probe else
                           args.evaluation/f'{model}-sakura-expression-{grouping}.json')
            choices = read(choice_path)
            assignments_path = (args.frozen/'sakura-assignments.json' if args.probe else
                                args.evaluation/f'{model}-sakura-assignments.json')
            assignments = read(assignments_path)
            if set(assignments) != set(piano(analysis)):
                raise ValueError('Stream assignments do not cover exactly the raw piano candidates')
            evidence, replaced = replace_piano_expression(evidence, metadata, trace, raw, choices,
                                                         model, grouping, settings)
        assert {n.id for n in evidence} == {n['id'] for n in original_evidence}
        assert all(not n.hold or n.release_reason not in UNOBSERVED_RELEASES for n in evidence)
        evidence_trace = copy.deepcopy(trace)
        aligned, alignment = align_candidates(chart_clock_notes(evidence, clock['audio_shift_ms']), clock, confirmed=True)
        selected, selection = select_expression(aligned, metadata, trace, settings, 60000/clock['bpm'])
        placed = assign_lanes(selected, settings, beat_ms=60000/clock['bpm'])
        structural, grid = validate(placed, keys), grid_report(placed, clock)
        assert structural['valid'] and grid['off_grid'] == 0 and len(placed) == len(selected)
        aligned_by_id = {n.id: n for n in aligned}
        assert all((n.start, n.end, n.voice, n.pitch) == (aligned_by_id[n.id].start, aligned_by_id[n.id].end,
                    aligned_by_id[n.id].voice, aligned_by_id[n.id].pitch) for n in placed)
        assert all(not n.hold or n.release_reason not in UNOBSERVED_RELEASES for n in placed)
        assert all(n.voice != 'vocals' for n in placed)
        assert analysis == original_analysis
        baseline_path = args.previous/f'instrument-other-independent-{keys}k-notes.json'
        baseline_data = read(baseline_path)
        baseline = [note_from(n) for n in baseline_data]
        if model == 'baseline':
            assert sorted(map(field_key, baseline_data)) == sorted(field_key(n.data()) for n in placed)
        delta, drums = drum_delta(aligned, placed, baseline, metadata)
        diagnostic = (None if model == 'baseline' else
                      hold_diagnostics(before_notes, evidence, placed, metadata, raw, choices, assignments, grouping))
        content = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu',
                             title='Sakura piano stream expression comparison', audio='audio.mp3',
                             bpm=clock['bpm'], offset=clock['offset'], version=name+' research',
                             grid_division=clock['division'])
        parsed = parse_osu(content)
        endpoints = lambda notes: sorted((n.start, n.end, n.lane) for n in notes)
        assert parsed['keys'] == keys and endpoints(parsed['notes']) == endpoints(placed)
        assert validate(parsed['notes'], keys)['valid'] and grid_report(parsed['notes'], clock)['off_grid'] == 0
        (args.output/(name+'.osu')).write_text(content, encoding='utf-8')
        save(name+'-plan.json', plan)
        save(name+'-expression.json', dict(notes=[n.data() for n in evidence], trace=evidence_trace, changes=replaced))
        save(name+'-aligned.json', [n.data() for n in aligned])
        save(name+'-selection.json', selection)
        save(name+'-notes.json', [n.data() for n in placed])
        save(name+'-drum-delta.json', delta)
        save(name+'-hold-diagnostics.json', diagnostic)
        output_paths = [args.output/(name+suffix) for suffix in
                        ('.osu', '-plan.json', '-expression.json', '-aligned.json', '-selection.json', '-notes.json', '-drum-delta.json', '-hold-diagnostics.json')]
        row = dict(name=name, model=model, grouping=grouping, keys=keys,
                   elapsed_seconds=time.monotonic()-started, structural=structural, grid=grid, alignment=alignment,
                   selection_optimal=selection['attack_optimal'], source_counts=source_counts(placed),
                   expressed_source_counts=source_counts(evidence), expression_changes=len(replaced), drums=drums,
                   large_attack_shifts=dict(Counter(n.voice for n in placed if n.evidence_start is not None and abs(n.start-n.evidence_start)>30)),
                   omission_reasons=selection['omission_reasons'], layout=layout_metrics(placed, keys, 60000/clock['bpm']),
                   choices_sha256=None if choice_path is None else sha256(choice_path),
                   assignments_sha256=None if assignments_path is None else sha256(assignments_path),
                   newly_held_diagnostics=None if diagnostic is None else {k: diagnostic[k] for k in ('expressed', 'selected')},
                   baseline_notes_sha256=sha256(baseline_path), artifacts_sha256={p.name: sha256(p) for p in output_paths},
                   original_candidates_unchanged=True, non_piano_expression_unchanged=True,
                   weak_and_omitted_candidates_unchanged=True, selected_endpoints_preserved=True,
                   roundtrip_endpoints_equal=True, baseline_exactly_reproduced=model=='baseline')
        cases.append(row)
        save('progress.json', cases)
        print(json.dumps({k: row[k] for k in ('name', 'source_counts', 'expression_changes', 'drums', 'selection_optimal')}), flush=True)
    audio_path = args.previous/'audio.mp3'
    shutil.copyfile(audio_path, args.output/'audio.mp3')
    save('report.json', dict(identity=identity, cases=cases, audio_sha256=sha256(audio_path),
                             musical_acceptance=False, visual_check=False, enabled_in_application=False,
                             notice='Frozen expression consequences, with manual timing. Counts and valid exports do not establish musical quality.'))


if __name__ == '__main__':
    main()
