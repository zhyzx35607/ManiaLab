"""Export a private 6–10K comparison pack through an already analyzed EXE project.

This intentionally contains the source audio. Keep the destination private.
The source project must be a test copy; its recent candidates will be updated.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.domain import clean_meta, parse_osu, validate
from manialab.timing import grid_report


def run(executable, data_dir, project_id, output, bpm, offset, title=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir.resolve()
    if not re.fullmatch(r'[a-f0-9]{32}', project_id):
        raise ValueError('Invalid project ID')
    state_path = data_dir / 'projects' / project_id / 'state.json'
    saved = json.loads(state_path.read_text(encoding='utf-8'))
    if not saved.get('analysis'):
        raise ValueError('Analyze the project before exporting comparisons')
    marker = data_dir / 'current-url.txt'
    marker.unlink(missing_ok=True)
    process = None
    base = '/api/projects/' + project_id
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (output / 'launch.log').open('w', encoding='utf-8') as log:
        try:
            process = subprocess.Popen([str(executable.resolve()), '--no-browser', '--data-dir', str(data_dir)],
                stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
            deadline = time.monotonic() + 60
            while not marker.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('EXE did not start')
                time.sleep(.1)
            url = marker.read_text(encoding='utf-8').strip()
            while True:
                try:
                    html = http.open(url, timeout=2).read().decode()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.1)
            token = re.search(r'name="manialab-token" content="([^"]+)"', html).group(1)

            def request(path, payload=None):
                body = json.dumps(payload).encode() if payload is not None else None
                headers = {'Content-Type': 'application/json', 'X-ManiaLab-Token': token} if body else {}
                with http.open(urllib.request.Request(url + path, data=body, headers=headers), timeout=90) as response:
                    raw = response.read()
                    return json.loads(raw) if 'application/json' in response.headers.get('Content-Type', '') else raw

            def wait(job):
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    row = request('/api/jobs/' + job['id'])
                    if row['status'] in ('done', 'error', 'cancelled'):
                        if row['status'] != 'done':
                            raise RuntimeError(str(row))
                        return
                    time.sleep(.2)
                raise TimeoutError('Comparison generation exceeded ten minutes')

            # Full settings avoid inheriting the preceding variant's overrides.
            default = request(base)['settings'].copy()
            default.update(keys=10, density=7., ln_ratio=.55, min_ln=240, max_held=5,
                release_gap=55, chord_size=4, pitch_follow=.55, balance=.8, jack_avoid=.8,
                seed=42, focus='balanced', phrase_ln=.65, layout_engine='beam', pattern_prior=.35,
                drums_weight=1., bass_weight=1., vocals_weight=1., other_weight=1.)
            cases = [
                ('A-default-10K', {}),
                ('B-LN-target20-10K', {'ln_ratio': .2}),
                ('C-LN-target80-10K', {'ln_ratio': .8}),
                ('D-vocal-focus-10K', {'ln_ratio': .7, 'phrase_ln': 1., 'vocals_weight': 1.8,
                                     'other_weight': .6, 'drums_weight': .7, 'bass_weight': .8}),
                *[(f'{keys}K-default', {'keys': keys}) for keys in range(6, 10)],
            ]
            charts = []
            reports = []
            original_payloads = None
            for name, updates in cases:
                settings = default | updates
                tempo = dict(bpm=bpm, offset=offset, division=4)
                wait(request(base + '/generate', dict(mode='audio', settings=settings,
                    audio_shift_ms=22.5, timing_confirmed=True, **tempo)))
                state = request(base)
                variant = state['variants'][state['active']]
                content = request(base + '/export/osu').decode('utf-8')
                # Only visible chart metadata changes after EXE export.
                content = re.sub(r'(?m)^Version:.*$', 'Version:ManiaLab alpha4 ' + name, content)
                if title is not None:
                    for field in ('Title', 'TitleUnicode'):
                        content = re.sub(r'(?m)^' + field + r':.*$', lambda _: field + ':' + clean_meta(title), content)
                parsed = parse_osu(content)
                structure = validate(parsed['notes'], settings['keys'])
                grid = grid_report(parsed['notes'], tempo)
                if not structure['valid'] or grid['off_grid']:
                    raise AssertionError({'structure': structure, 'grid': grid})
                charts.append((name + '.osu', content))
                reports.append(dict(name=name, settings=settings, structure=structure, grid=grid,
                    actual_ln_ratio=variant['selection']['actual_ln_ratio'],
                    feasible_ln_ratio=variant['selection']['feasible_ln_ratio'],
                    large_attack_shifts=variant['selection']['selected_large_attack_shifts'],
                    osu_sha256=hashlib.sha256(content.encode()).hexdigest()))
                if original_payloads is None:
                    with zipfile.ZipFile(io.BytesIO(request(base + '/export/osz'))) as source:
                        original_payloads = {name: source.read(name) for name in source.namelist() if not name.lower().endswith('.osu')}
                print(json.dumps({'name': name, **structure, 'off_grid': grid['off_grid']}, ensure_ascii=True), flush=True)
            archive_path = output / 'ManiaLab-alpha4-private-comparison-6-10K.osz'
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name, payload in original_payloads.items():
                    archive.writestr(name, payload)
                for name, content in charts:
                    archive.writestr(name, content.encode('utf-8'))
            with zipfile.ZipFile(archive_path) as archive:
                assert archive.testzip() is None
                assert len([n for n in archive.namelist() if n.endswith('.osu')]) == len(cases)
                for name, content in charts:
                    assert archive.read(name) == content.encode('utf-8')
            report = dict(version=request('/api/status')['version'], project_id=project_id,
                source_executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                timing=dict(bpm=bpm, offset=offset, division=4, audio_shift_ms=22.5),
                notice='Private source audio included; do not publish. Parameter comparisons, not ranked difficulty or player-quality certification.',
                charts=reports, archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
            (output / 'comparison-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            return report
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--project-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bpm', type=float, required=True)
    parser.add_argument('--offset', type=int, required=True)
    parser.add_argument('--title')
    args = parser.parse_args()
    run(args.executable, args.data_dir, args.project_id, args.output, args.bpm, args.offset, args.title)
