"""Isolate nonforward rest-edge repair with frozen inputs and model weights."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.adapt_piano_streams import augment, f1, load_model, make_graph, pairs, verify_source
from scripts.build_provenance import sha256
from scripts.check_piano_stream_context import contexts, differences, full_items, read, windows
from scripts.evaluate_piano_stream_adaptation import component_pair_counts, graph_for
from scripts.research_piano_streams import controls, expression_choices
from manialab.domain import note_from


REST = ('note', 'rest', 'note')


def repair_rest(graph):
    """Drop only impossible forward rest edges before official reverse edges."""
    repaired = graph.clone()
    assert set(graph[REST].keys()) == {'edge_index'}
    onset = graph['note'].onset_div
    end = onset + graph['note'].duration_div
    edges = graph[REST].edge_index
    keep = onset[edges[1]] > end[edges[0]]
    repaired[REST].edge_index = edges[:, keep].clone()
    return repaired


def verify_repair(graph, repaired):
    """Check graph identity plus exact next-onset rest semantics, not just counts."""
    before, after = graph.to_dict(), repaired.to_dict()
    assert before.keys() == after.keys()
    for store, fields in before.items():
        assert fields.keys() == after[store].keys()
        for field, value in fields.items():
            if (store, field) == (REST, 'edge_index'):
                continue
            other = after[store][field]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), (store, field)
            else:
                assert value == other, (store, field)
    onset = graph['note'].onset_div.numpy()
    duration = graph['note'].duration_div.numpy()
    assert np.all(duration > 0)
    assert np.all(np.diff(onset) >= 0)
    end = onset + duration
    expected = set()
    for source, tail in enumerate(end):
        if np.any(onset == tail):
            continue
        future = onset[onset > tail]
        if future.size:
            expected.update((source, int(target)) for target in np.flatnonzero(onset == future.min()))
    original = pairs(graph[REST].edge_index.numpy())
    fixed = pairs(repaired[REST].edge_index.numpy())
    assert fixed == expected, 'Rest graph must exactly match earliest strictly future onset'
    assert fixed <= original, 'Repair must not add edges'
    assert original - fixed == {(a, b) for a, b in original if onset[b] <= end[a]}
    return dict(before=len(original), after=len(fixed), removed=len(original - fixed))


def prediction_delta(before, after, measures):
    # This function compares predictions; relation audits are reported separately.
    result = differences(before, dict(after, observed_edge_audit_json=np.asarray('{}')), measures)
    del result['observed_edges']
    return result


def reproduce(prediction, path):
    with np.load(path, allow_pickle=False) as saved:
        assert set(prediction) == set(saved.files)
        for key in saved.files:
            np.testing.assert_array_equal(prediction[key], saved[key], err_msg=f'{path.name}: {key}')


def run_contexts(model, count, measures, build, baseline, prefix, output):
    chunks = windows(measures)
    all_onsets = build(np.arange(count))['note'].onset_div.numpy()
    for indices in chunks:
        assert np.all(np.diff(indices) == 1), 'Induced-graph audit requires contiguous windows'
        assert indices[0] == 0 or all_onsets[indices[0]-1] < all_onsets[indices[0]]
        assert indices[-1] == count-1 or all_onsets[indices[-1]] < all_onsets[indices[-1]+1]
    raw_full, raw_windows = contexts(model, count, chunks, build)
    for name, prediction in (('full', raw_full), ('four_measures', raw_windows)):
        reproduce(prediction, baseline / f'{prefix}-{name}.npz')
    audits = []

    def corrected(indices):
        graph = build(indices)
        snapshot = graph.clone()
        repaired = repair_rest(graph)
        audit = verify_repair(graph, repaired)
        # Applying the filter must not mutate the source graph.
        unchanged = graph.to_dict()
        for key, fields in snapshot.to_dict().items():
            for field, value in fields.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, unchanged[key][field])
                else:
                    assert value == unchanged[key][field]
        audits.append(dict(context='full' if len(indices) == count else 'window',
                           notes=len(indices), **audit))
        return repaired

    fixed_full, fixed_windows = contexts(model, count, chunks, corrected)
    observed = json.loads(str(fixed_windows['observed_edge_audit_json']))
    for kind, audit in observed.items():
        assert audit['window_only'] == audit['full_internal_missing_in_windows'] == 0, kind
    assert observed['rest']['full_nonforward'] == observed['rest']['window_nonforward'] == 0
    raw = dict(full=raw_full, four_measures=raw_windows)
    fixed = dict(full=fixed_full, four_measures=fixed_windows)
    if audits[0]['removed'] == 0:
        for key in raw_full:
            np.testing.assert_array_equal(raw_full[key], fixed_full[key])
    if all(a['removed'] == 0 for a in audits[1:]):
        for key in raw_windows:
            # This audit also describes the full graph, which may have changed.
            if key != 'observed_edge_audit_json':
                np.testing.assert_array_equal(raw_windows[key], fixed_windows[key])
    for name, prediction in fixed.items():
        np.savez_compressed(output / f'{prefix}-{name}.npz', **prediction)
    return raw, fixed, dict(graphs=audits, observed_edges=observed,
        raw_predictions_reproduced=True,
        changes={name: prediction_delta(raw[name], fixed[name], measures) for name in raw},
        fixed_context_difference=differences(fixed_full, fixed_windows, measures))


def totals(cases):
    result = {}
    for model in ('original', 'adapted'):
        result[model] = {}
        for kind in ('clean', 'whole'):
            subset = [c for c in cases if c['model'] == model and c['input'] == kind]
            result[model][kind] = {}
            for context in ('full', 'four_measures'):
                row = {}
                for variant in ('raw', 'fixed'):
                    counts = Counter(tp=0, fp=0, fn=0)
                    for case in subset:
                        counts.update(case['scores'][variant][context]['counts'])
                    row[variant] = dict(counts=dict(counts), f1=f1(counts),
                        piece_macro_f1=float(np.mean([c['scores'][variant][context]['f1'] for c in subset])))
                delta = [c['scores']['fixed'][context]['f1'] - c['scores']['raw'][context]['f1'] for c in subset]
                row['piece_changes'] = dict(better=sum(d > 1e-9 for d in delta),
                    equal=sum(abs(d) <= 1e-9 for d in delta), worse=sum(d < -1e-9 for d in delta))
                result[model][kind][context] = row
    return result


def expression_delta(before, after):
    assert before.keys() == after.keys()
    return dict(before_holds=sum(c['hold'] for c in before.values()),
        after_holds=sum(c['hold'] for c in after.values()),
        hold_tap_flips=sum(c['hold'] != after[key]['hold'] for key, c in before.items()))


def expression_ids(before, after):
    return {direction: [key for key, value in before.items()
            if value['hold'] == old and after[key]['hold'] != old]
            for direction, old in (('hold_to_tap', True), ('tap_to_hold', False))}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--dataset', type=Path, default=ROOT/'test-data/piano-svsep/dataset-v2')
    cli.add_argument('--run', type=Path, default=ROOT/'test-data/piano-svsep/adaptation-v1')
    cli.add_argument('--baseline', type=Path, default=ROOT/'test-data/piano-svsep/context-v1')
    cli.add_argument('--frozen', type=Path, default=ROOT/'test-data/piano-svsep/probe-v3')
    cli.add_argument('--analysis', type=Path, default=ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous rest-edge experiments')
    revision = verify_source(args.source)
    sys.path.insert(0, str(args.source.resolve()))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    baseline = read(args.baseline/'report.json')
    assert sha256(args.baseline/'runner-snapshot.py') == baseline['identity']['runner_sha256']
    # Allow Git's line-ending normalization and removal of a trailing empty line;
    # every non-newline character must still match the executed baseline helper.
    current_context = (ROOT/'scripts/check_piano_stream_context.py').read_text(encoding='utf-8')
    frozen_context = (args.baseline/'runner-snapshot.py').read_text(encoding='utf-8')
    assert current_context.rstrip('\n') == frozen_context.rstrip('\n')
    selected_sha = sha256(args.run/'selected.pt')
    assert selected_sha == baseline['identity']['selected_sha256']
    assert sha256(args.dataset/'manifest.json') == baseline['identity']['dataset_sha256']
    assert sha256(args.analysis) == baseline['identity']['analysis_sha256']
    assert sha256(args.frozen/'sakura-graph-input.json') == baseline['identity']['graph_rows_sha256']
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    save = lambda name, value: (args.output/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    helpers = ('adapt_piano_streams', 'check_piano_stream_context', 'evaluate_piano_stream_adaptation',
               'research_piano_streams')
    snapshots = {}
    for name in helpers:
        path = ROOT/'scripts'/f'{name}.py'
        snapshots[name] = sha256(path)
        shutil.copyfile(path, args.output/f'{name}-snapshot.py')
    identity = dict(source_revision=revision, selected_sha256=selected_sha,
        dataset_sha256=sha256(args.dataset/'manifest.json'), analysis_sha256=sha256(args.analysis),
        graph_rows_sha256=sha256(args.frozen/'sakura-graph-input.json'),
        baseline_sha256=sha256(args.baseline/'report.json'), runner_sha256=sha256(__file__),
        protocol_sha256=sha256(ROOT/'docs/piano-stream-rest-protocol.md'), snapshots=snapshots)
    save('identity.json', identity)
    control_results = []
    boundary = [dict(tick=t, duration=d, pitch=p) for t, d, p in
                ((0, 120, 48), (0, 240, 60), (360, 120, 52), (360, 120, 64),
                 (480, 600, 55), (600, 240, 67))]
    for name, rows in dict(controls(), rest_boundary=boundary).items():
        for expanded in (False, True):
            graph = graph_for(rows, expanded)
            control_results.append(dict(name=name, expanded=expanded,
                **verify_repair(graph, repair_rest(graph))))
    # Minimal graphs avoid unrelated feature/candidate construction limitations.
    from torch_geometric.data import HeteroData
    for name, onset, duration in (('single_note', [0], [120]),
            ('equal_tails', [0, 120, 240], [480, 360, 240])):
        graph = HeteroData()
        graph['note'].onset_div = torch.tensor(onset)
        graph['note'].duration_div = torch.tensor(duration)
        graph[REST].edge_index = torch.empty((2, 0), dtype=torch.long)
        control_results.append(dict(name=name, **verify_repair(graph, repair_rest(graph))))
    save('controls.json', control_results)
    items = full_items(args.dataset, read(args.dataset/'manifest.json'))
    model = load_model(args.source/'pretrained_models/model.ckpt').eval()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    adapted = torch.load(args.run/'selected.pt', map_location='cpu', weights_only=True)['state_dict']
    analysis = read(args.analysis)
    notes = [note_from(n) for n in analysis['candidates'] if n['voice'] == 'piano']
    rows = read(args.frozen/'sakura-graph-input.json')
    rebuilt = []
    beat = 60000 / 74
    for note in notes:
        start = max(0, round((note.start-22.5-178)/beat*4)*120)
        end = round((note.end-22.5-178)/beat*4)*120
        rebuilt.append(dict(id=note.id, pitch=note.pitch, tick=start, duration=max(120, end-start)))
    rebuilt.sort(key=lambda row: (row['tick'], row['pitch'], row['id']))
    assert rows == rebuilt
    measures = np.asarray([row['tick']//1920 for row in rows])
    assert {row['id'] for row in rows} == {note.id for note in notes}
    before_notes = [n.data() for n in notes]
    cases, sakura = [], []
    started = time.monotonic()
    for name, weights, expanded in (('original', original, False), ('adapted', adapted, True)):
        model.load_state_dict(weights, strict=True)
        model.eval()
        for item in items:
            original_notes = item['notes'].copy()
            for kind in ('clean', 'whole'):
                frozen = augment(original_notes, item['bounds'], kind)

                def build(indices):
                    local = dict(name=item['name'], notes=frozen[indices],
                        measures=item['measures'][indices], bounds=item['bounds'][indices])
                    return make_graph(local, 'clean', expanded)[0]

                raw, fixed, audit = run_contexts(model, len(frozen), item['measures'], build,
                    args.baseline, f'{name}-{item["name"]}-{kind}', args.output)
                scores = {}
                for variant, predictions in (('raw', raw), ('fixed', fixed)):
                    scores[variant] = {}
                    for context, prediction in predictions.items():
                        counts = component_pair_counts(item, prediction['components'])
                        scores[variant][context] = dict(counts=dict(counts), f1=f1(counts),
                            staff_accuracy=float(np.mean(prediction['staff'] == original_notes['staff']-1)))
                cases.append(dict(model=name, name=item['name'], input=kind, scores=scores, audit=audit))
                save('validation-progress.json', cases)
            assert np.array_equal(original_notes, item['notes'])
            print(json.dumps(dict(model=name, completed=item['name'], cases=len(cases))), flush=True)
        raw, fixed, audit = run_contexts(model, len(rows), measures,
            lambda indices: graph_for([rows[i] for i in indices], expanded),
            args.baseline, f'{name}-sakura', args.output)
        policies = {}
        for variant, predictions in (('raw', raw), ('fixed', fixed)):
            policies[variant] = {}
            for context, prediction in predictions.items():
                assignments = {row['id']: dict(staff=int(prediction['staff'][i]),
                    components=int(prediction['components'][i])) for i, row in enumerate(rows)}
                policies[variant][context] = {mode: expression_choices(notes,
                    analysis['structure']['segments'], assignments, mode) for mode in ('staff', 'components')}
                for mode, choices in policies[variant][context].items():
                    save(f'{name}-sakura-{variant}-{context}-{mode}.json', choices)
        sakura.append(dict(model=name, audit=audit,
            repair_expression_changes={context: {mode: expression_delta(policies['raw'][context][mode],
                policies['fixed'][context][mode]) for mode in ('staff', 'components')} for context in raw},
            context_expression_changes={variant: {mode: expression_delta(policies[variant]['full'][mode],
                policies[variant]['four_measures'][mode]) for mode in ('staff', 'components')}
                for variant in ('raw', 'fixed')}))
        save(f'{name}-sakura-expression-change-ids.json', {
            context: {mode: expression_ids(policies['raw'][context][mode], policies['fixed'][context][mode])
            for mode in ('staff', 'components')} for context in raw})
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    assert [n.data() for n in notes] == before_notes
    report = dict(identity=identity, controls=control_results, validation=totals(cases),
        cases=cases, sakura=sakura, elapsed_seconds=time.monotonic()-started,
        test_data_used=False, model_selection_changed=False, weights_unchanged=True,
        raw_candidates_unchanged=True, enabled_in_application=False, musical_acceptance=False)
    report['artifacts_sha256'] = {path.name: sha256(path) for path in args.output.iterdir()
        if path.suffix in ('.json', '.npz') and path.name not in ('validation-progress.json', 'report.json')}
    save('report.json', report)
    print(json.dumps(report['validation']), flush=True)


if __name__ == '__main__':
    main()
