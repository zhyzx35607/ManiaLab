"""Replay fixed stream predictions with explicit global/fragment context."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.arrange import parameters, assign_lanes
from manialab.domain import Note, UNOBSERVED_RELEASES, note_from, validate, export_osu, parse_osu
from manialab.expression import propose_expression, expression_evidence, select_expression
from manialab.expression_articulation import articulation_policy
from manialab.expression_streams import candidate_signature
from manialab.timing import chart_clock_notes, align_candidates, grid_report
from scripts.build_provenance import sha256
from scripts.check_piano_stream_expression import drum_delta

FROZEN = ROOT/'test-data/piano-svsep/probe-v3'
ADAPTED = ROOT/'test-data/piano-svsep/adaptation-evaluation-v1'
CONTROLS = ('lower_hold', 'upper_hold', 'chord_layers', 'crossing', 'wide_arpeggio', 'pedal_overlap')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))


def prediction(path, rows):
    with np.load(path, allow_pickle=False) as archive:
        groups = {key:archive[key].copy() for key in ('staff', 'components')}
    if any(v.shape != (len(rows),) or v.dtype.kind not in 'iu' or np.any(v<0) for v in groups.values()):
        raise ValueError('Invalid saved stream assignments')
    return {row['id']:{key:str(int(v[i])) for key,v in groups.items()} for i,row in enumerate(rows)}


def sidecar(notes, assignments, path, method, duration):
    hypotheses = [dict(method='staff', identity_scope='source', prediction_sha256=sha256(path),
                       groups={n.id:assignments[n.id]['staff'] for n in notes})]
    if method == 'agreement':
        hypotheses.append(dict(method='components', identity_scope='fragment', prediction_sha256=sha256(path),
                               groups={n.id:assignments[n.id]['components'] for n in notes}))
    return dict(revision=1, voice='piano', observed_until_ms=round(duration*1000),
                candidates_sha256=candidate_signature(notes), hypotheses=hypotheses)


def diagnostic_components(notes, assignments, until):
    """Old assumption for comparison only, not a valid global sidecar."""
    grouped = defaultdict(list)
    for n in notes:grouped[assignments[n.id]['components']].append(n)
    choices = {}
    for group in grouped.values():choices.update(articulation_policy(group, observed_until_ms=until))
    return choices


def control_cases(output, save):
    cases = []
    for name in CONTROLS:
        rows = read(FROZEN/(name+'-input.json'))
        # Part labels never enter sidecar construction or expression rules.
        notes = [Note(1000+round(r['tick']*1000/480), 1000+round((r['tick']+r['duration'])*1000/480),
                      pitch=r['pitch'], id=r['id'], voice='piano', source='pitch', release_reason='symbolic_control') for r in rows]
        analysis = dict(duration=18, candidates=[n.data() for n in notes],
                        structure=dict(segments=[dict(start_ms=0, end_ms=18000)]))
        p = parameters();plan = propose_expression(analysis, p)
        baseline, _, baseline_trace = expression_evidence(analysis, plan, p)
        by_id = {n.id:n for n in baseline}
        for model in ('original', 'adapted'):
            path = FROZEN/(name+'.npz') if model == 'original' else ADAPTED/('adapted-'+name+'.npz')
            assignments = prediction(path, rows)
            legacy = diagnostic_components(notes, assignments, 18000)
            legacy_holds = [n.id for n in notes if n.end-n.start>=p['min_ln'] and not legacy[n.id]['tap']]
            for method in ('staff', 'agreement'):
                evidence = sidecar(notes, assignments, path, method, 18)
                current, _, trace = expression_evidence(analysis, plan, p, stream_evidence=evidence)
                original = {n.id:n for n in notes}
                assert len(current) == len(notes) and all(n.end in (n.start, original[n.id].end) for n in current)
                changed = [dict(id=n.id, pitch=n.pitch, start=n.start, original_end=original[n.id].end,
                    baseline_end=by_id[n.id].end, expression_end=n.end) for n in current if n.end != by_id[n.id].end]
                prefix = f'{model}-{name}-{method}'
                save(prefix+'-sidecar.json', evidence);save(prefix+'-trace.json', trace)
                # Labels are used only to describe which symbolic reference
                # parts changed after decisions have been frozen.
                parts = {r['id']:str(r['part']) for r in rows}
                row = dict(name=name, model=model, method=method, candidates=len(notes),
                    baseline_holds=sum(n.hold for n in baseline), holds=sum(n.hold for n in current),
                    naive_component_holds=len(legacy_holds), holds_by_control_part=dict(Counter(parts[n.id] for n in current if n.hold)),
                    statuses=dict(Counter(r['articulation']['stream']['status'] for r in trace.values())),
                    changed=changed, input_sha256=sha256(FROZEN/(name+'-input.json')), prediction_sha256=sha256(path))
                cases.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('changed', 'input_sha256', 'prediction_sha256')}, ensure_ascii=True), flush=True)
    save('controls.json', cases)
    return cases


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--controls-only', action='store_true')
    args = cli.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier runs')
    frozen_report = read(FROZEN/'report.json');adapted_report = read(ADAPTED/'report.json')
    assert sha256(FROZEN/'runner-snapshot.py') == frozen_report['identity']['runner_sha256']
    assert sha256(ADAPTED/'runner-snapshot.py') == adapted_report['identity']['runner_sha256']
    assert sha256(FROZEN/'report.json') == adapted_report['identity']['frozen_report_sha256']
    args.output.mkdir(parents=True)
    def save(name, data):
        with (args.output/name).open('x', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
    files = ['manialab/expression.py', 'manialab/expression_streams.py', 'manialab/expression_articulation.py',
             'manialab/expression_activity.py', 'manialab/arrange.py', 'manialab/domain.py', 'manialab/timing.py',
             'scripts/check_stream_expression_context.py', 'scripts/check_piano_stream_expression.py']
    identities = {path:sha256(ROOT/path) for path in files}
    inputs = [FROZEN/'report.json', ADAPTED/'report.json']
    for name in CONTROLS:inputs.extend([FROZEN/(name+'-input.json'), FROZEN/(name+'.npz'), ADAPTED/('adapted-'+name+'.npz')])
    frozen = {str(p):sha256(p) for p in inputs}
    save('execution-plan.json', dict(source_files=identities, inputs=frozen, controls=list(CONTROLS),
        controls_only=args.controls_only, protocol_sha256=sha256(ROOT/'docs/stream-expression-context-protocol.md'),
        model_inference=False, reference_chart_read=False, musical_acceptance=False))
    snapshots = args.output/'source';snapshots.mkdir()
    for path in files:shutil.copyfile(ROOT/path, snapshots/Path(path).name)
    controls = control_cases(args.output, save)
    if args.controls_only:
        assert frozen == {p:sha256(p) for p in frozen} and identities == {p:sha256(ROOT/p) for p in files}
        save('completed.json', dict(controls_sha256=sha256(args.output/'controls.json'), musical_acceptance=False))
        return
    analysis_path = ROOT/'test-data/sakura-tsumugi-expression-onset-v3/analysis.json'
    piano_path = ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json'
    config_path = ROOT/'test-data/sakura-tsumugi-fullsong-v1/execution-plan.json'
    analysis, old = read(analysis_path), read(piano_path)
    assert sha256(piano_path) == frozen_report['identity']['analysis_sha256']
    piano_data = lambda a: {n['id']:n for n in a['candidates'] if n['voice']=='piano'}
    assert piano_data(analysis) == piano_data(old)
    raw = {n['id']:note_from(n) for n in analysis['candidates']}
    piano = [n for n in raw.values() if n.voice=='piano']
    original_analysis = copy.deepcopy(analysis)
    rows = read(FROZEN/'sakura-graph-input.json')
    assert {r['id'] for r in rows} == {n.id for n in piano}
    prediction_paths = dict(original=FROZEN/'sakura-predictions.npz', adapted=ADAPTED/'adapted-sakura-predictions.npz')
    assignments = {model:prediction(path, rows) for model,path in prediction_paths.items()}
    for model in assignments:
        path = FROZEN/'sakura-assignments.json' if model=='original' else ADAPTED/'adapted-sakura-assignments.json'
        saved = read(path)
        assert assignments[model] == {key:{kind:str(value[kind]) for kind in ('staff', 'components')} for key,value in saved.items()}
        frozen[str(path)] = sha256(path)
    config = read(config_path)['frozen_generation'];clock = config['generation_clock']
    baseline_dir = ROOT/'test-data/balanced-expression-default-v1'
    assert sha256(baseline_dir/'report.json') == '6425c5e5da7706e819dc603276d202d832ebf37c210e48fc4a8d65a9b88e1079'
    audio = piano_path.parent/'audio.mp3'
    assert sha256(audio) == config['mix_sha256']
    for path in [analysis_path, piano_path, config_path, *prediction_paths.values(), FROZEN/'sakura-graph-input.json', audio]:
        frozen[str(path)] = sha256(path)
    suite = [('baseline', 'all', 10)] + [(model, method, 10) for model in assignments for method in ('staff', 'agreement')]
    suite += [('adapted', 'agreement', keys) for keys in range(6, 10)]
    save('song-plan.json', dict(cases=suite, inputs=frozen, generation_clock=clock, settings=config['settings'],
        model_inference=False, automatic_timing_acceptance=False))
    cases, charts = [], []
    for model, method, keys in suite:
        started = time.monotonic();name=f'{model}-{method}-{keys}k'
        p = parameters(config['settings'] | {'keys':keys});plan=propose_expression(analysis, p)
        hint = None if model=='baseline' else sidecar(piano, assignments[model], prediction_paths[model], method, analysis['duration'])
        if hint is not None:save(name+'-sidecar.json', hint)
        evidence, metadata, trace = expression_evidence(analysis, plan, p, stream_evidence=hint)
        before, _, _ = expression_evidence(analysis, plan, p)
        assert [(n.id,n.start,n.pitch,n.voice,n.instrument_id) for n in evidence] == [(n.id,n.start,n.pitch,n.voice,n.instrument_id) for n in before]
        assert [n.data() for n in evidence if n.voice!='piano'] == [n.data() for n in before if n.voice!='piano']
        assert all(n.end in (n.start,raw[n.id].end) and (not n.hold or n.release_reason not in UNOBSERVED_RELEASES) for n in evidence)
        expression_trace = copy.deepcopy(trace)
        save(name+'-expression.json', dict(notes=[n.data() for n in evidence], trace=expression_trace))
        aligned, alignment = align_candidates(chart_clock_notes(evidence, clock['audio_shift_ms']), clock, confirmed=True)
        selected, selection = select_expression(aligned, metadata, trace, p, 60000/clock['bpm'])
        placed = assign_lanes(selected, p, beat_ms=60000/clock['bpm'])
        assert {(n.id,n.start,n.end) for n in placed} == {(n.id,n.start,n.end) for n in selected}
        structural, grid = validate(placed, keys), grid_report(placed, clock)
        assert structural['valid'] and grid['off_grid']==0 and analysis == original_analysis
        baseline = [note_from(n) for n in read(baseline_dir/f'independent-articulation-{keys}k-notes.json')]
        if model=='baseline':
            assert sorted((n.id,n.start,n.end,n.lane) for n in placed) == sorted((n.id,n.start,n.end,n.lane) for n in baseline)
        drum_details, drum_summary = drum_delta(aligned, placed, baseline, metadata)
        content = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu', title='Sakura stream context comparison',
            version=name+' research', audio='audio.mp3', bpm=clock['bpm'], offset=clock['offset'], grid_division=clock['division'],
            bookmarks=analysis['structure']['boundaries'][1:-1])
        parsed = parse_osu(content)
        assert sorted((n.start,n.end,n.lane) for n in parsed['notes']) == sorted((n.start,n.end,n.lane) for n in placed)
        path=args.output/(name+'.osu');path.write_text(content, encoding='utf-8');charts.append(path)
        save(name+'-notes.json', [n.data() for n in placed]);save(name+'-selection.json', selection)
        save(name+'-drum-deltas.json', drum_details)
        base = {n.id:n for n in before}
        decisions = [r['articulation']['stream'] for r in expression_trace.values() if r.get('articulation',{}).get('stream')]
        row = dict(name=name, notes=len(placed), ln=sum(n.hold for n in placed), seconds=time.monotonic()-started,
            source_counts=dict(Counter(n.voice for n in placed)), source_holds=dict(Counter(n.voice for n in placed if n.hold)),
            piano_expression_holds=sum(n.hold and n.voice=='piano' for n in evidence),
            expression_new_holds=sum(n.hold and not base[n.id].hold for n in evidence),
            expression_new_taps=sum(not n.hold and base[n.id].hold for n in evidence),
            statuses=dict(Counter(r['status'] for r in decisions)), drums=drum_summary,
            large_attack_shifts=dict(Counter(n.voice for n in placed if abs(n.start-n.evidence_start)>30)),
            structural=structural, grid=grid, alignment=alignment, solver=selection['selection_solver'])
        cases.append(row);save(name+'-metrics.json', row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('structural','grid','alignment')}, ensure_ascii=True), flush=True)
    archive_path=args.output/'sakura-stream-context.osz'
    with zipfile.ZipFile(archive_path,'x',zipfile.ZIP_DEFLATED) as archive:
        archive.write(audio,'audio.mp3')
        for path in charts:archive.write(path,path.name)
    with zipfile.ZipFile(archive_path) as archive:
        import hashlib
        assert hashlib.sha256(archive.read('audio.mp3')).hexdigest()==config['mix_sha256']
    assert frozen == {p:sha256(p) for p in frozen} and identities == {p:sha256(ROOT/p) for p in files}
    save('report.json', dict(cases=cases, controls_sha256=sha256(args.output/'controls.json'),
        execution_plan_sha256=sha256(args.output/'execution-plan.json'), song_plan_sha256=sha256(args.output/'song-plan.json'),
        source_files=identities, osz_sha256=sha256(archive_path), musical_acceptance=False, enabled_in_application=False,
        neural_inference=False, automatic_timing_acceptance=False))


if __name__=='__main__':main()
