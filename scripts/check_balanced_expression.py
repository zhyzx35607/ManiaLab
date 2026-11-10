"""Compare section priority and expression using frozen full-song detections."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.arrange import parameters, assign_lanes
from manialab.domain import UNOBSERVED_RELEASES, export_osu, parse_osu, validate
from manialab.expression import propose_expression, expression_evidence, select_expression
from manialab.timing import chart_clock_notes, align_candidates, grid_report


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(notes):
    return dict(Counter(n.voice for n in notes))


def comparison_plan(analysis, settings, legacy, policy):
    if policy in ('legacy', 'neutral-priority'):
        plan = copy.deepcopy(legacy)
        if policy == 'neutral-priority':
            revised = propose_expression(analysis, settings)
            for section, neutral in zip(plan['sections'], revised['sections']):
                section['main_voice'] = neutral['main_voice']
                for voice, value in section['voices'].items():
                    value['role'] = neutral['voices'][voice]['role']
    else:
        plan = propose_expression(analysis, settings)
        for section in plan['sections']:
            for voice, value in section['voices'].items():
                if voice != 'drums' and value['mode'] != 'omit' and settings['ln_ratio'] > 0:
                    value['mode'] = 'articulation' if policy == 'independent-articulation' and voice in ('piano', 'guitar') else 'sustain'
    return plan


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, default=Path('test-data/sakura-tsumugi-expression-onset-v3'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--policy', choices=['all', 'independent-sustain', 'independent-articulation'], default='all')
    cli.add_argument('--keys', type=int, nargs='+', choices=range(6, 11), default=[10])
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Keep prior comparisons; choose a new output directory')
    source_report = args.input/'report.json'
    if sha(source_report) != '985e18d620a0a4724b74926dcde1486f5ed23c539ac3e834d7039c728577b322':
        raise ValueError('Expected the fixed onset-v3 completed comparison')
    config_path = ROOT/'test-data/sakura-tsumugi-fullsong-v1/execution-plan.json'
    config = read(config_path)['frozen_generation']
    clock = config['generation_clock']
    analysis_path = args.input/'analysis.json'
    analysis = read(analysis_path)
    original = copy.deepcopy(analysis)
    plan_path = args.input/'tsumugi-releases-10k-expression.json'
    legacy = read(plan_path)
    audio = ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/audio.mp3'
    if sha(audio) != config['mix_sha256']:
        raise ValueError('Source audio changed')
    policies = ['legacy', 'neutral-priority', 'independent-sustain', 'independent-articulation'] if args.policy == 'all' else [args.policy]
    args.output.mkdir(parents=True)
    def save(name, value):
        with (args.output/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    files = ['manialab/expression.py', 'manialab/expression_activity.py', 'manialab/expression_articulation.py',
             'manialab/arrange.py', 'manialab/domain.py', 'manialab/timing.py', 'scripts/check_balanced_expression.py']
    identities = {path: sha(ROOT/path) for path in files}
    inputs = {str(path):sha(path) for path in (source_report, config_path, analysis_path, plan_path, audio)}
    save('execution-plan.json', dict(source_files=identities, inputs=inputs, policies=policies, keys=args.keys,
        settings=config['settings'], generation_clock=clock, neural_inference=False,
        automatic_timing_acceptance=False, musical_acceptance=False))
    snapshots = args.output/'source';snapshots.mkdir()
    for path in files:
        shutil.copyfile(ROOT/path, snapshots/Path(path).name)
    chart_files, cases = [], []
    for keys in args.keys:
        for policy in policies:
            started = time.monotonic()
            settings = parameters(config['settings'] | {'keys':keys})
            plan = comparison_plan(analysis, settings, legacy, policy)
            evidence, metadata, trace = expression_evidence(analysis, plan, settings)
            aligned, alignment = align_candidates(chart_clock_notes(evidence, clock['audio_shift_ms']), clock, confirmed=True)
            selected, selection = select_expression(aligned, metadata, trace, settings, 60000/clock['bpm'])
            placed = assign_lanes(selected, settings, beat_ms=60000/clock['bpm'])
            by_id = {n.id:n for n in selected}
            assert len(placed) == len(selected)
            assert all((n.start, n.end, n.voice, n.pitch, n.instrument_id) ==
                       (by_id[n.id].start, by_id[n.id].end, by_id[n.id].voice, by_id[n.id].pitch, by_id[n.id].instrument_id) for n in placed)
            assert all(n.voice != 'vocals' and (not n.hold or n.release_reason not in UNOBSERVED_RELEASES) for n in placed)
            structural, grid = validate(placed, keys), grid_report(placed, clock)
            assert structural['valid'] and grid['off_grid'] == 0 and analysis == original
            name = f'{policy}-{keys}k'
            content = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu',
                title='Sakura source expression comparison', audio='audio.mp3', bpm=clock['bpm'], offset=clock['offset'],
                version=name+' research', grid_division=clock['division'],
                bookmarks=analysis['structure']['boundaries'][1:-1])
            parsed = parse_osu(content)
            field = lambda n:(n.start, n.end, n.lane)
            assert parsed['keys'] == keys and sorted(map(field, parsed['notes'])) == sorted(map(field, placed))
            path = args.output/(name+'.osu');path.write_text(content, encoding='utf-8');chart_files.append(path)
            save(name+'-plan.json', plan);save(name+'-selection.json', selection)
            save(name+'-notes.json', [n.data() for n in placed])
            source = {n['id']:n for n in analysis['candidates']}
            section_rows = []
            for section in plan['sections']:
                a, b = section['start_ms'], section['end_ms']
                candidates = [n for n in evidence if a <= n.start < b]
                final = [n for n in placed if a <= source[n.id]['start'] < b]
                section_rows.append(dict(start_ms=a, end_ms=b, main_voice=section['main_voice'],
                    source_counts=counts(final), source_holds=counts([n for n in final if n.hold]),
                    expression_counts=counts(candidates)))
            attacks = {(n.voice, n.start) for n in placed}
            case = dict(name=name, keys=keys, policy=policy, seconds=time.monotonic()-started,
                matches_application_default=plan == propose_expression(analysis, settings),
                notes=len(placed), ln=sum(n.hold for n in placed), source_counts=counts(placed),
                source_holds=counts([n for n in placed if n.hold]),
                source_attacks=dict(Counter(voice for voice, _ in attacks)),
                large_attack_shifts=counts([n for n in placed if abs(n.start-n.evidence_start)>30]),
                main_voices=[s['main_voice'] for s in plan['sections']], sections=section_rows,
                structural=structural, grid=grid, alignment=alignment, solver=selection['selection_solver'],
                attack_optimal=selection['attack_optimal'], notes_sha256=sha(args.output/(name+'-notes.json')))
            if policy == 'legacy' and keys == 10:
                previous = read(args.input/'tsumugi-releases-10k-notes.json')
                fields = lambda n:(n['id'], n['start'], n['end'], n['lane'])
                case['frozen_output_equal'] = sorted(map(fields, previous)) == sorted(fields(n.data()) for n in placed)
            cases.append(case);save(name+'-metrics.json', case)
            print(json.dumps({k:v for k,v in case.items() if k in ('name', 'notes', 'ln', 'source_counts', 'source_holds', 'large_attack_shifts', 'solver', 'frozen_output_equal')}, ensure_ascii=True), flush=True)
    archive_path = args.output/'sakura-expression-comparison.osz'
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.write(audio, 'audio.mp3')
        for path in chart_files:archive.write(path, path.name)
    with zipfile.ZipFile(archive_path) as archive:
        assert hashlib.sha256(archive.read('audio.mp3')).hexdigest() == config['mix_sha256']
    assert identities == {path:sha(ROOT/path) for path in files}
    assert inputs == {path:sha(path) for path in inputs}
    save('report.json', dict(cases=cases, execution_plan_sha256=sha(args.output/'execution-plan.json'),
        osz_sha256=sha(archive_path), neural_inference=False, reference_chart_read=False,
        automatic_timing_acceptance=False, musical_acceptance=False))


if __name__ == '__main__':
    main()
