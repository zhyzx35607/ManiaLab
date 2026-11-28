"""Candidate edits and exports keep saved sources after analysis changes."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import time
import wave
import zipfile

import numpy as np
import pytest
from scipy.io import wavfile

from manialab.arrange import parameters
from manialab.domain import Note, parse_osu
from manialab.server import create_app
from manialab.stems import separation_identity
from manialab import export_routes


ROOT = Path(__file__).resolve().parents[1]


def _wave(at_sample=0):
    samples = bytearray(8000 * 5 * 2)
    samples[at_sample * 2:at_sample * 2 + 2] = b'\x00\x60'
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        audio.writeframes(samples)
    return output.getvalue()


def _edit(client, base, headers, candidate, **changes):
    response = client.post(base + '/edit', json={
        'base_candidate_id': candidate['candidate_id'], **changes,
    }, headers=headers)
    assert response.status_code == 200, response.json
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + response.json['id']).json
        if job['status'] not in ('queued', 'running'):
            assert job['status'] == 'done', job
            state = client.get(base).json
            return state, state['variants'][state['active']]
        time.sleep(.01)
    pytest.fail('Candidate edit did not finish')


def _generate(client, base, headers, payload):
    response = client.post(base + '/generate', json=payload, headers=headers)
    assert response.status_code == 200, response.json
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + response.json['id']).json
        if job['status'] not in ('queued', 'running'):
            assert job['status'] == 'done', job
            state = client.get(base).json
            return state, state['variants'][state['active']]
        time.sleep(.01)
    pytest.fail('Candidate generation did not finish')


def _assert_exports(client, base, candidate, analysis, music):
    evidence = client.get(base + '/export/json')
    assert evidence.status_code == 200
    assert evidence.json['variant'] == candidate
    assert evidence.json['analysis'] == analysis
    chart = client.get(base + '/export/osu')
    bundle = client.get(base + '/export/osz')
    assert chart.status_code == bundle.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        chart_name = next(name for name in archive.namelist() if name.endswith('.osu'))
        assert archive.read(chart_name) == chart.data
        assert archive.read('song.wav') == music
    parsed = parse_osu(chart.get_data(as_text=True))
    assert [(note.start, note.end, note.lane) for note in parsed['notes']] == [
        (note['start'], note['end'], note['lane']) for note in candidate['notes']
    ]
    assert parsed['general']['AudioFilename'] == 'song.wav'


def _export_snapshot(route, response):
    if route != '/export/osz' or response.status_code != 200:
        return response.status_code, response.data
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        payloads = [(info.filename, archive.read(info)) for info in archive.infolist()]
    return response.status_code, payloads


def _audio_export_project(tmp_path, *, candidate_mode='audio', source_sha256=None,
                          record_audio_identity=True, legacy_deep=False, audio_name='song.wav'):
    """Build a tiny project without invoking analysis or inference code."""
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    music = _wave()
    imported = client.post('/api/import', headers=headers,
        data={'files': (io.BytesIO(music), 'song.wav')})
    assert imported.status_code == 200, imported.json
    pid = imported.json['id']
    base = '/api/projects/' + pid
    projects = app.extensions['projects']
    state = projects.load(pid)
    if audio_name != 'song.wav':
        (projects.path(pid) / 'files' / 'song.wav').rename(
            projects.path(pid) / 'files' / audio_name)
        state['audio'] = audio_name
    candidate = dict(
        mode=candidate_mode, timing_revision=3, settings=parameters(),
        notes=[Note(1000, 1500, 1).data()], bpm=120, offset=0, division=4, meter=4,
    )
    if candidate_mode == 'audio' and record_audio_identity:
        candidate['source_audio_sha256'] = (
            source_sha256 if source_sha256 is not None else hashlib.sha256(music).hexdigest())
    if legacy_deep:
        candidate['source_separation'] = {
            'separation': {'audio_sha256': hashlib.sha256(music).hexdigest()}}
    if candidate_mode == 'remap':
        from manialab.domain import export_osu

        source = export_osu([Note(1000, 1500, 1)], 6, ROOT / 'vendor/mapperatorinator/template.osu',
                            audio=audio_name)
        source_path = projects.path(pid) / 'files' / 'source.osu'
        source_path.write_text(source, encoding='utf-8')
        state.update(original='source.osu', input_notes=[Note(1000, 1500, 1).data()], input_keys=6,
                     input_timing=parse_osu(source)['timing'], charts=[{'path': 'source.osu', 'label': '6K'}])
    state.update(variants=[candidate], active=0)
    projects.save(state)
    return app, client, headers, base, projects.path(pid) / 'files' / state['audio'], music, candidate


@pytest.mark.parametrize('kind', ['json', 'osu', 'osz'])
def test_export_rejects_an_old_tabs_candidate(tmp_path, kind):
    app, first, headers, base, _, _, _ = _audio_export_project(tmp_path)
    try:
        pid = base.rsplit('/', 1)[1]
        projects = app.extensions['projects']
        state = projects.load(pid)
        old_id = state['variants'][0]['candidate_id']
        new_candidate = deepcopy(state['variants'][0])
        new_candidate.pop('candidate_id')
        new_candidate['notes'] = [Note(2000, 2500, 1).data()]
        state['variants'].append(new_candidate)
        projects.save(state)
        new_id = state['variants'][1]['candidate_id']
        route = f'{base}/export/{kind}'
        old_link = f'{route}?candidate_id={old_id}'
        assert first.get(old_link).status_code == 200

        second = app.test_client()
        selected = second.post(base + '/variant', headers=headers,
                               json={'index': 1, 'candidate_id': new_id})
        assert selected.status_code == 200, selected.json
        stale = first.get(old_link)
        assert stale.status_code == 400
        assert '刷新' in stale.json['error']
        assert 'Content-Disposition' not in stale.headers

        fresh = second.get(route, query_string={'candidate_id': new_id})
        current = first.get(route)
        assert fresh.status_code == current.status_code == 200
        assert _export_snapshot('/export/' + kind, fresh) == _export_snapshot('/export/' + kind, current)
    finally:
        app.extensions['executor'].shutdown()


def test_analysis_export_can_require_that_no_candidate_is_selected(tmp_path):
    app, client, _, base, _, _, _ = _audio_export_project(tmp_path)
    try:
        pid = base.rsplit('/', 1)[1]
        projects = app.extensions['projects']
        state = projects.load(pid)
        state['analysis'] = {'duration': 5}
        state['active'] = -1
        projects.save(state)
        route = base + '/export/json'
        analysis_link = route + '?candidate_id='
        response = client.get(analysis_link)
        assert response.status_code == 200
        assert response.json['variant'] is None
        candidate_id = state['variants'][0]['candidate_id']
        assert client.get(route, query_string={'candidate_id': candidate_id}).status_code == 400

        state['active'] = 0
        projects.save(state)
        stale = client.get(analysis_link)
        assert stale.status_code == 400
        assert '刷新' in stale.json['error']
        assert client.get(route).json['variant']['candidate_id'] == candidate_id
    finally:
        app.extensions['executor'].shutdown()


def test_osz_rejects_same_name_audio_replacement_but_json_and_osu_remain_available(tmp_path):
    original = _wave()
    replacement = _wave(400)
    app, client, _, base, audio, _, candidate = _audio_export_project(
        tmp_path)
    try:
        audio.write_bytes(original)
        json_before = client.get(base + '/export/json')
        osu_before = client.get(base + '/export/osu')
        assert json_before.status_code == osu_before.status_code == 200
        variant_before = json_before.json['variant']
        audio.write_bytes(replacement)

        response = client.get(base + '/export/osz')

        assert response.status_code == 400
        assert '原始音频' in response.json['error']
        assert client.get(base + '/export/json').json['variant'] == variant_before
        assert client.get(base + '/export/osu').data == osu_before.data
    finally:
        app.extensions['executor'].shutdown()


def test_osz_audio_mismatch_closes_temporary_file(tmp_path, monkeypatch):
    original = _wave()
    replacement = _wave(400)
    app, client, _, base, audio, _, _ = _audio_export_project(
        tmp_path)
    real_temporary_file = export_routes.tempfile.TemporaryFile
    opened = []

    def track_temporary_file(*args, **kwargs):
        handle = real_temporary_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(export_routes.tempfile, 'TemporaryFile', track_temporary_file)
    try:
        audio.write_bytes(replacement)
        response = client.get(base + '/export/osz')
        assert response.status_code == 400
        assert len(opened) == 1 and opened[0].closed
    finally:
        app.extensions['executor'].shutdown()


def test_osz_checks_final_payload_when_audio_changes_during_zip_write(tmp_path, monkeypatch):
    original = _wave()
    replacement = _wave(400)
    app, client, _, base, audio, _, _ = _audio_export_project(
        tmp_path)
    real_write = export_routes.zipfile.ZipFile.write

    def replace_during_copy(archive, filename, *args, **kwargs):
        if Path(filename).resolve() != audio.resolve():
            return real_write(archive, filename, *args, **kwargs)
        assert audio.read_bytes() == original
        audio.write_bytes(replacement)
        try:
            return real_write(archive, filename, *args, **kwargs)
        finally:
            audio.write_bytes(original)

    monkeypatch.setattr(export_routes.zipfile.ZipFile, 'write', replace_during_copy)
    try:
        response = client.get(base + '/export/osz')
        assert response.status_code == 400
        assert '原始音频' in response.json['error']
        assert audio.read_bytes() == original
    finally:
        app.extensions['executor'].shutdown()


def test_osz_legacy_deep_audio_identity_fallback(tmp_path):
    original = _wave()
    app, client, _, base, audio, _, _ = _audio_export_project(
        tmp_path, record_audio_identity=False, legacy_deep=True)
    try:
        audio.write_bytes(original)
        response = client.get(base + '/export/osz')
        assert response.status_code == 200, response.json
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            assert archive.read('song.wav') == original
    finally:
        app.extensions['executor'].shutdown()


def test_osz_rejects_malformed_explicit_identity_instead_of_using_legacy_fallback(tmp_path):
    original = _wave()
    app, client, _, base, _, _, candidate = _audio_export_project(
        tmp_path, source_sha256='invalid', legacy_deep=True)
    try:
        assert client.get(base + '/export/json').status_code == 200
        assert client.get(base + '/export/osu').status_code == 200
        response = client.get(base + '/export/osz')
        assert response.status_code == 400
        assert '未保存有效的原始音频 SHA-256' in response.json['error']
        assert candidate['source_audio_sha256'] == 'invalid'
    finally:
        app.extensions['executor'].shutdown()


def test_osz_rejects_legacy_basic_candidate_without_audio_identity(tmp_path):
    app, client, _, base, _, _, candidate = _audio_export_project(
        tmp_path, record_audio_identity=False)
    try:
        assert 'source_audio_sha256' not in candidate
        assert client.get(base + '/export/json').status_code == 200
        assert client.get(base + '/export/osu').status_code == 200
        response = client.get(base + '/export/osz')
        assert response.status_code == 400
        assert '未保存有效的原始音频 SHA-256' in response.json['error']
    finally:
        app.extensions['executor'].shutdown()


def test_remap_osz_does_not_require_audio_identity(tmp_path):
    app, client, _, base, audio, _, candidate = _audio_export_project(
        tmp_path, candidate_mode='remap')
    try:
        assert 'source_audio_sha256' not in candidate
        response = client.get(base + '/export/osz')
        assert response.status_code == 200, response.json
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            assert archive.read('song.wav') == audio.read_bytes()
    finally:
        app.extensions['executor'].shutdown()


def test_audio_osz_hash_resolves_case_insensitive_reference_to_unique_member(tmp_path, monkeypatch):
    original = _wave()
    app, client, _, base, audio, _, _ = _audio_export_project(
        tmp_path, audio_name='SONG.WAV')
    real_export_osu = export_routes.export_osu

    def lowercase_audio_reference(*args, **kwargs):
        content = real_export_osu(*args, **kwargs)
        return content.replace('AudioFilename: SONG.WAV', 'AudioFilename: song.wav')

    monkeypatch.setattr(export_routes, 'export_osu', lowercase_audio_reference)
    try:
        assert audio.name == 'SONG.WAV' and audio.exists()
        response = client.get(base + '/export/osz')
        assert response.status_code == 200, response.json
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            chart_name = next(name for name in archive.namelist() if name.endswith('.osu'))
            chart = parse_osu(archive.read(chart_name).decode())
            assert chart['general']['AudioFilename'] == 'SONG.WAV'
            assert archive.read('SONG.WAV') == original
            assert [name for name in archive.namelist()
                    if name.casefold() == 'song.wav'] == ['SONG.WAV']
    finally:
        app.extensions['executor'].shutdown()


@pytest.mark.parametrize('duplicate', [False, True])
def test_audio_osz_rejects_missing_or_duplicate_referenced_payload(tmp_path, monkeypatch, duplicate):
    original = _wave()
    app, client, _, base, audio, _, _ = _audio_export_project(
        tmp_path)
    real_write = export_routes.zipfile.ZipFile.write

    def alter_audio_member(archive, filename, *args, **kwargs):
        result = real_write(archive, filename, *args, **kwargs)
        if Path(filename).resolve() == audio.resolve():
            if duplicate:
                archive.writestr('song.wav', _wave(600))
            return result
        return result

    if duplicate:
        monkeypatch.setattr(export_routes.zipfile.ZipFile, 'write', alter_audio_member)
    else:
        monkeypatch.setattr(export_routes.zipfile.ZipFile, 'write',
            lambda archive, filename, *args, **kwargs: None
            if Path(filename).resolve() == audio.resolve()
            else real_write(archive, filename, *args, **kwargs))
    try:
        if duplicate:
            with pytest.warns(UserWarning, match="Duplicate name: 'song.wav'"):
                response = client.get(base + '/export/osz')
        else:
            response = client.get(base + '/export/osz')
        assert response.status_code == 400
        assert '缺失或重复' in response.json['error']
    finally:
        app.extensions['executor'].shutdown()


def test_remap_osz_rejects_case_insensitive_duplicate_audio_members(tmp_path, monkeypatch):
    app, client, _, base, audio, _, _ = _audio_export_project(
        tmp_path, candidate_mode='remap')
    real_write = export_routes.zipfile.ZipFile.write

    def add_case_variant_duplicate(archive, filename, *args, **kwargs):
        result = real_write(archive, filename, *args, **kwargs)
        if Path(filename).resolve() == audio.resolve():
            archive.writestr('SONG.WAV', _wave(600))
        return result

    monkeypatch.setattr(export_routes.zipfile.ZipFile, 'write', add_case_variant_duplicate)
    try:
        response = client.get(base + '/export/osz')
        assert response.status_code == 400
        assert '缺失或重复' in response.json['error']
    finally:
        app.extensions['executor'].shutdown()


def test_remap_osz_rejects_osu_file_as_audio_reference(tmp_path):
    app, client, _, base, _, _, _ = _audio_export_project(
        tmp_path, candidate_mode='remap')
    try:
        state = app.extensions['projects'].load(base.rsplit('/', 1)[-1])
        chart_path = app.extensions['projects'].path(state['id']) / 'files' / state['original']
        chart = chart_path.read_text(encoding='utf-8')
        assert 'AudioFilename: song.wav' in chart
        chart_path.write_text(chart.replace(
            'AudioFilename: song.wav', 'AudioFilename: source.osu'), encoding='utf-8')

        response = client.get(base + '/export/osz')

        assert response.status_code == 400
        assert '不能引用 .osu' in response.json['error']
    finally:
        app.extensions['executor'].shutdown()


def test_remap_osz_rejects_existing_non_audio_reference_before_creating_archive(
        tmp_path, monkeypatch):
    app, client, _, base, _, _, _ = _audio_export_project(
        tmp_path, candidate_mode='remap')
    real_temporary_file = export_routes.tempfile.TemporaryFile
    opened = []

    def track_temporary_file(*args, **kwargs):
        handle = real_temporary_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(export_routes.tempfile, 'TemporaryFile', track_temporary_file)
    try:
        projects = app.extensions['projects']
        state = projects.load(base.rsplit('/', 1)[-1])
        project = projects.path(state['id'])
        chart_path = project / 'files' / state['original']
        chart = chart_path.read_text(encoding='utf-8')
        assert 'AudioFilename: song.wav' in chart
        chart_path.write_text(chart.replace(
            'AudioFilename: song.wav', 'AudioFilename: cover.png'), encoding='utf-8')
        (project / 'files' / 'cover.png').write_bytes(b'synthetic cover image')
        state_path = project / 'state.json'
        before_state = state_path.read_bytes()

        response = client.get(base + '/export/osz')

        assert response.status_code == 400
        assert '支持的音频格式' in response.json['error']
        assert not response.data.startswith(b'PK')
        assert state_path.read_bytes() == before_state
        assert opened == []
        assert not list(tmp_path.rglob('ManiaLab.osz'))
    finally:
        app.extensions['executor'].shutdown()


def test_regional_generation_rejects_changed_parent_source_without_mutating_exports(tmp_path):
    app, client, headers, base, _, music, candidate = _audio_export_project(tmp_path)
    try:
        projects = app.extensions['projects']
        state = projects.load(base.rsplit('/', 1)[-1])
        audio_hash = hashlib.sha256(music).hexdigest()
        parent_source = dict(source='provided', audio_sha256=audio_hash,
            set_sha256='a' * 64, voices=['other'])
        current_source = parent_source | {'set_sha256': 'b' * 64}
        candidate['source_separation'] = dict(
            identity=separation_identity(parent_source), separation=parent_source)
        state.update(analysis=dict(revision=14, duration=5,
            candidates=[Note(3500, 4250, id='tone', source='pitch', voice='other').data()],
            phrase_candidates=[], timing_attacks=[], separation=current_source),
            variants=[candidate], active=0)
        projects.save(state)
        state_path = projects.path(state['id']) / 'state.json'
        before_state = state_path.read_bytes()
        before_candidate = deepcopy(projects.load(state['id'])['variants'][0])
        routes = ('/export/json', '/export/osu', '/export/osz')
        before_exports = {}
        for route in routes:
            response = client.get(base + route)
            assert response.status_code == 200, response.json
            before_exports[route] = _export_snapshot(route, response)

        submitted = client.post(base + '/generate', headers=headers, json=dict(
            mode='audio', settings=candidate['settings'], bpm=120, offset=0,
            division=4, audio_shift_ms=0, timing_confirmed=True, region=[3000, 5000]))
        assert submitted.status_code == 200, submitted.json
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get('/api/jobs/' + submitted.json['id']).json
            if job['status'] in ('done', 'error', 'cancelled'):
                break
            time.sleep(.01)
        assert job['status'] == 'error', job
        assert '不能混用不同或未知的声轨来源' in job['message']

        assert state_path.read_bytes() == before_state
        assert projects.load(state['id'])['variants'][0] == before_candidate
        for route in routes:
            response = client.get(base + route)
            assert response.status_code == 200, response.json
            assert _export_snapshot(route, response) == before_exports[route]
    finally:
        app.extensions['executor'].shutdown()


@pytest.mark.parametrize('previous_audio_identity', ['missing', 'different'])
def test_regional_generation_rejects_unknown_or_changed_audio_sha_without_mutating_exports(
        tmp_path, previous_audio_identity):
    app, client, headers, base, _, music, candidate = _audio_export_project(tmp_path)
    try:
        projects = app.extensions['projects']
        state = projects.load(base.rsplit('/', 1)[-1])
        if previous_audio_identity == 'missing':
            candidate.pop('source_audio_sha256')
        else:
            candidate['source_audio_sha256'] = hashlib.sha256(music).hexdigest()
        state.update(analysis=dict(revision=14, duration=5,
            audio_sha256=hashlib.sha256(_wave(400)).hexdigest(),
            candidates=[Note(3500, 4250, id='tone', source='pitch').data()],
            phrase_candidates=[], timing_attacks=[]),
            variants=[candidate], active=0)
        projects.save(state)

        state_path = projects.path(state['id']) / 'state.json'
        before_state = state_path.read_bytes()
        before_candidate = deepcopy(projects.load(state['id'])['variants'][0])
        routes = ('/export/json', '/export/osu', '/export/osz')
        before_exports = {}
        for route in routes:
            response = client.get(base + route)
            expected_status = (400 if previous_audio_identity == 'missing'
                               and route == '/export/osz' else 200)
            assert response.status_code == expected_status, response.json
            before_exports[route] = _export_snapshot(route, response)

        submitted = client.post(base + '/generate', headers=headers, json=dict(
            mode='audio', settings=candidate['settings'], bpm=120, offset=0,
            division=4, audio_shift_ms=0, timing_confirmed=True, region=[3000, 5000]))
        assert submitted.status_code == 200, submitted.json
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get('/api/jobs/' + submitted.json['id']).json
            if job['status'] in ('done', 'error', 'cancelled'):
                break
            time.sleep(.01)
        assert job['status'] == 'error', job
        assert '不能混用不同或未知的声轨来源' in job['message']

        assert state_path.read_bytes() == before_state
        assert projects.load(state['id'])['variants'][0] == before_candidate
        for route in routes:
            response = client.get(base + route)
            assert _export_snapshot(route, response) == before_exports[route]
    finally:
        app.extensions['executor'].shutdown()


def test_regional_generation_keeps_legacy_unknown_identity_when_both_sources_lack_identity(tmp_path):
    app, client, headers, base, _, _, candidate = _audio_export_project(
        tmp_path, record_audio_identity=False)
    try:
        projects = app.extensions['projects']
        state = projects.load(base.rsplit('/', 1)[-1])
        state.update(analysis=dict(revision=14, duration=5,
            candidates=[Note(3500, 4250, id='tone', source='pitch').data()],
            phrase_candidates=[], timing_attacks=[]),
            variants=[candidate], active=0)
        projects.save(state)

        submitted = client.post(base + '/generate', headers=headers, json=dict(
            mode='audio', settings=candidate['settings'], bpm=120, offset=0,
            division=4, audio_shift_ms=0, timing_confirmed=True, region=[3000, 5000]))
        assert submitted.status_code == 200, submitted.json
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get('/api/jobs/' + submitted.json['id']).json
            if job['status'] in ('done', 'error', 'cancelled'):
                break
            time.sleep(.01)
        assert job['status'] == 'done', job
        saved = projects.load(state['id'])
        assert saved['variants'][saved['active']]['source_audio_sha256'] is None
    finally:
        app.extensions['executor'].shutdown()


@pytest.mark.parametrize('regional', [False, True])
def test_source_snapshots_survive_successive_edits_exports_and_reopening(tmp_path, regional):
    music = _wave()
    source_audio_sha256 = hashlib.sha256(music).hexdigest()
    old_stem = _wave(800)
    current_stem = _wave(2400)
    saved_source = dict(source='provided', audio_sha256=source_audio_sha256,
        set_sha256='a' * 64, voices=['other'],
        decoded_files={'other': {'wav_sha256': hashlib.sha256(old_stem).hexdigest()}})
    reference = dict(identity=separation_identity(saved_source), separation=saved_source)
    raw = dict(start=1020, end=1820, voice='other', release_reason='frame_decay')
    parent = dict(mode='audio', timing_revision=3, created=1., settings=parameters(),
        bpm=120, offset=0, division=4, audio_shift_ms=20, meter=3,
        notes=[Note(1000, 1875, 0, source='pitch', voice='other', id='tone',
                    evidence_start=1000, evidence_end=1800).data(),
               Note(2500, 2500, 1, source='transient', voice='other', id='tap').data()],
        report={}, selection={}, region=None, source_separation=reference,
        source_audio_sha256=source_audio_sha256,
        source_evidence={
            'tone': dict(detected=raw, expressed=dict(raw, end=1895)),
            'tap': dict(
                detected=dict(start=2490, end=2500, voice='other', source='transient'),
                expressed=dict(start=2490, end=2500),
            ),
        })
    if regional:
        parent['region'] = [0, 2000]
        parent['selection'] = dict(scope='regional_output', proposal={'selected': 1})
    current_analysis = dict(revision=14, duration=5,
        candidates=[Note(3500, 4250, id='tone', source='pitch', voice='other').data()],
        separation=dict(saved_source, set_sha256='b' * 64,
            decoded_files={'other': {'wav_sha256': hashlib.sha256(current_stem).hexdigest()}}))
    app = create_app(ROOT, tmp_path)
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(music), 'song.wav')})
        assert imported.status_code == 200, imported.json
        state = imported.json
        pid = state['id']
        base = '/api/projects/' + pid
        state.update(variants=[deepcopy(parent)], active=0, analysis=current_analysis)
        projects = app.extensions['projects']
        projects.save(state)
        for identity, audio in [('a' * 64, old_stem), ('b' * 64, current_stem)]:
            source_file = projects.path(pid) / 'stems-provided' / identity / 'other.wav'
            source_file.parent.mkdir(parents=True)
            source_file.write_bytes(audio)
        stored = projects.load(pid)
        parent, current_analysis = stored['variants'][0], stored['analysis']
        assert parent['source_audio_sha256'] == source_audio_sha256

        state, changed = _edit(client, base, headers, parent, op='update', note_id='tone',
            values=dict(start=1118, end=2376, lane=2))
        state, added = _edit(client, base, headers, changed, op='add',
            values=dict(start=3001, end=3501, lane=3))
        deleted_evidence = deepcopy(added['source_evidence']['tap'])
        state, final = _edit(client, base, headers, added, op='delete', note_id='tap')
        assert [(note['start'], note['end']) for note in final['notes']] == [(1125, 2375), (3000, 3500)]
        assert not any(note['id'] == 'tap' for note in final['notes'])
        assert final['source_evidence']['tap'] == deleted_evidence
        assert final['edits'][-1]['before']['id'] == 'tap'
        manual = next(note for note in final['notes'] if note['source'] == 'manual')
        assert manual['id'] not in final['source_evidence']
        for version in state['variants']:
            assert version['source_separation'] == reference
            assert version['source_evidence'] == parent['source_evidence']
        assert state['variants'][0] == parent
        assert final['meter'] == parent['meter']
        _assert_exports(client, base, final, current_analysis, music)
        evidence_export = client.get(base + '/export/json').json['variant']
        assert evidence_export['source_audio_sha256'] == source_audio_sha256
        assert evidence_export['source_evidence']['tap'] == deleted_evidence
        assert evidence_export['edits'][-1]['before']['id'] == 'tap'
        assert not any(note['id'] == 'tap' for note in evidence_export['notes'])
        deleted_chart_note = (2500, 2500, 1)
        chart = parse_osu(client.get(base + '/export/osu').get_data(as_text=True))
        assert deleted_chart_note not in [(note.start, note.end, note.lane) for note in chart['notes']]
        with zipfile.ZipFile(io.BytesIO(client.get(base + '/export/osz').data)) as archive:
            chart_name = next(name for name in archive.namelist() if name.endswith('.osu'))
            bundled_chart = parse_osu(archive.read(chart_name).decode())
        assert deleted_chart_note not in [
            (note.start, note.end, note.lane) for note in bundled_chart['notes']
        ]
        response = client.get(base + '/stem/other?context=chart&source_identity=' + reference['identity'])
        assert response.status_code == 200 and response.data == old_stem
        restored = client.post(base + '/variant', json={
            'index': 0, 'candidate_id': parent['candidate_id']}, headers=headers)
        assert restored.status_code == 200
        _assert_exports(client, base, parent, current_analysis, music)
    finally:
        app.extensions['executor'].shutdown()

    reopened = create_app(ROOT, tmp_path)
    try:
        client = reopened.test_client()
        headers = {'X-ManiaLab-Token': reopened.config['TOKEN']}
        history = client.get(base).json
        assert history['variants'] == state['variants']
        assert [variant['source_audio_sha256'] for variant in history['variants']] == [
            source_audio_sha256
        ] * len(history['variants'])
        restored = client.post(base + '/variant', json={
            'index': 3, 'candidate_id': history['variants'][3]['candidate_id']}, headers=headers)
        assert restored.status_code == 200
        reopened_export = client.get(base + '/export/json').json['variant']
        assert reopened_export['source_audio_sha256'] == source_audio_sha256
        _assert_exports(client, base, final, current_analysis, music)
        response = client.get(base + '/stem/other?context=chart&source_identity=' + reference['identity'])
        assert response.status_code == 200 and response.data == old_stem
    finally:
        reopened.extensions['executor'].shutdown()


@pytest.mark.parametrize(('audio_shift_ms', 'expected_region_times'), [
    (22.5, (3603, 4353)),
    (-75, (3728, 4478)),
])
def test_regional_ln_clock_evidence_audition_export_and_reopen_stay_aligned(
        tmp_path, audio_shift_ms, expected_region_times):
    """Keep regional LN evidence and chart playback aligned across persistence."""
    music = _wave()
    old_stem = _wave(800)
    separation = dict(source='provided', audio_sha256=hashlib.sha256(music).hexdigest(),
        set_sha256='a' * 64, voices=['other'],
        decoded_files={'other': {'wav_sha256': hashlib.sha256(old_stem).hexdigest()}})
    reference = dict(identity=separation_identity(separation), separation=separation)

    def assert_source_identity(candidate):
        source = candidate['source_separation']
        assert source['identity'] == reference['identity']
        assert separation_identity(source['separation']) == reference['identity']
        assert source['separation']['audio_sha256'] == separation['audio_sha256']
        assert source['separation']['decoded_files'] == separation['decoded_files']

    parent_candidates = [
        Note(1000, 1500, id='locked', source='pitch', voice='other',
             release_reason='frame_decay').data(),
        Note(1700, 2200, id='support', source='pitch', voice='other',
             release_reason='frame_decay').data(),
        Note(3500, 4200, id='region', source='pitch', voice='other',
             release_reason='frame_decay').data(),
    ]
    initial_structure = dict(boundaries=[0, 3000, 5000], segments=[
        dict(start_ms=0, end_ms=3000), dict(start_ms=3000, end_ms=5000),
    ])
    analysis = dict(revision=13, duration=5, candidates=deepcopy(parent_candidates),
        phrase_candidates=[Note(1000, 2800, id='locked', source='pitch', voice='vocals',
            phrase=True, support=('locked', 'support'), support_coverage=1,
            release_reason='vocal_activity_decay').data()],
        timing_attacks=[], separation=separation)
    settings = dict(ln_ratio=1, phrase_ln=1, min_ln=100, density=24, max_held=10)
    clock = dict(bpm=120, offset=103, division=4, audio_shift_ms=audio_shift_ms)
    app = create_app(ROOT, tmp_path)
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(music), 'song.wav')})
        assert imported.status_code == 200, imported.json
        state = imported.json
        pid = state['id']
        base = '/api/projects/' + pid
        projects = app.extensions['projects']
        state['analysis'] = analysis
        state['variants'] = []
        state['active'] = -1
        projects.save(state)
        (projects.path(pid) / 'preview.wav').write_bytes(music)
        stem_path = projects.path(pid) / 'stems-provided' / separation['set_sha256'] / 'other.wav'
        stem_path.parent.mkdir(parents=True)
        stem_path.write_bytes(old_stem)

        # Generate through the asynchronous API so persistence follows the app workflow.
        state, parent = _generate(client, base, headers, dict(mode='audio', settings=settings,
            timing_confirmed=True, **clock))
        assert parent['source_separation'] == reference
        assert parent['source_evidence']['locked']['detected']['end'] == 1500
        assert parent['source_evidence']['locked']['expressed']['phrase'] is True
        assert parent['source_evidence']['locked']['expressed']['support'] == ['locked', 'support']
        assert parent['source_evidence']['locked']['expressed']['end'] > 1500

        # Re-analysis moves only the regional candidate; the locked candidate and
        # separation identity remain the same, then a real regional child is made.
        changed_locked = Note(1000, 1900, id='locked', source='pitch', voice='other',
            release_reason='frame_decay').data()
        vocal_anchor = Note(3600, 3900, id='region', source='pitch', voice='vocals',
            release_reason='frame_decay').data()
        vocal_support = Note(4000, 4200, id='region-support', source='pitch', voice='vocals',
            release_reason='frame_decay').data()
        vocal_phrase = Note(3600, 4350, id='region', source='pitch', voice='vocals',
            phrase=True, support=('region-support',), support_coverage=1,
            release_reason='vocal_activity_decay').data()
        state['analysis']['structure'] = initial_structure
        projects.save(state)
        assert projects.load(pid)['analysis']['structure']['boundaries'] == [0, 3000, 5000]
        updated_structure = dict(boundaries=[0, 3200, 5000], segments=[
            dict(start_ms=0, end_ms=3200), dict(start_ms=3200, end_ms=5000),
        ])
        state['analysis'] = dict(state['analysis'], phrase_candidates=[vocal_phrase],
            candidates=[changed_locked,
                Note(1700, 2300, id='support', source='pitch', voice='other',
                    release_reason='frame_decay').data(),
                Note(2500, 2900, id='outside', source='pitch', voice='other',
                    release_reason='frame_decay').data(),
                vocal_anchor, vocal_support], structure=updated_structure)
        projects.save(state)

        # Refresh a full-song candidate against the changed sections, then
        # return to the older source snapshot before rewriting only one region.
        state, refreshed = _generate(client, base, headers, dict(mode='audio', settings=settings,
            timing_confirmed=True, **clock))
        assert refreshed['selection']['structure']['boundaries'] == [0, 3200, 5000]
        assert refreshed['source_evidence']['locked']['detected']['end'] == 1900
        assert_source_identity(refreshed)
        assert state['analysis']['structure']['boundaries'] != initial_structure['boundaries']
        switched = client.post(base + '/variant', json={
            'index': 0, 'candidate_id': state['variants'][0]['candidate_id']}, headers=headers)
        assert switched.status_code == 200
        state = client.get(base).json
        assert state['active'] == 0
        parent = state['variants'][0]
        assert parent['source_evidence']['locked']['detected']['end'] == 1500

        state, child = _generate(client, base, headers, dict(mode='audio', settings=settings,
            timing_confirmed=True, region=[3000, 5000], **clock))
        assert child['selection']['structure']['boundaries'] == [0, 3200, 5000]
        assert child['selection']['scope'] == 'regional_output'
        assert child['selection']['locked'] == 2 and child['selection']['generated'] == 1
        decisions = {row['id']: row for row in child['selection']['decisions']}
        assert decisions['locked']['reason'] == 'locked_existing'
        assert decisions['region']['reason'] == 'regional_regenerated'
        child_notes = {note['id']: note for note in child['notes']}
        assert child_notes['region']['voice'] == 'vocals'
        assert child_notes['locked']['end'] > child_notes['locked']['start']
        assert child_notes['region']['end'] > child_notes['region']['start']
        assert child['source_evidence']['region']['detected']['start'] == 3600
        assert child['source_evidence']['region']['detected']['end'] == 3900
        assert child['source_evidence']['region']['detected']['voice'] == 'vocals'
        assert_source_identity(child)
        assert child['source_evidence']['locked'] == parent['source_evidence']['locked']
        assert child['source_evidence']['support'] == parent['source_evidence']['support']
        assert child['source_evidence']['locked']['detected']['end'] == 1500
        assert child['selection']['scope'] == 'regional_output'
        assert child['selection']['proposal']['scope'] == 'full_song_proposal'
        proposal = child['selection']['proposal']
        assert proposal['candidates'] == 5
        assert proposal['selected'] == 4
        proposal_decisions = {row['id']: row for row in proposal['decisions']}
        assert proposal_decisions['region-support']['reason'] == 'represented_by_phrase'
        assert proposal_decisions['region-support']['represented_by'] == 'region'
        anchor_decision = proposal_decisions['region']
        assert anchor_decision['expression_end'] == 4350
        assert anchor_decision['decision'] == 'selected'
        assert anchor_decision['chart_end'] > anchor_decision['chart_start']
        expressed = child['source_evidence']['region']['expressed']
        assert expressed['start'] == 3600
        assert expressed['end'] == 4350
        assert expressed['phrase'] is True
        assert expressed['release_reason'] == 'vocal_activity_decay'
        assert expressed['support'] == ['region-support']
        assert expressed['support_coverage'] == 1
        region_chart_note = child_notes['region']
        assert (region_chart_note['start'], region_chart_note['end']) == expected_region_times
        tick_ms = 60000 / clock['bpm'] / clock['division']
        for value in (region_chart_note['start'], region_chart_note['end']):
            assert abs((value - clock['offset']) / tick_ms - round(
                (value - clock['offset']) / tick_ms)) < 1e-9
        assert child['selection']['selected'] == 3
        assert state['variants'][0] == parent
        projects.save(state)
        current_analysis = state['analysis']
    finally:
        app.extensions['executor'].shutdown()

    app = create_app(ROOT, tmp_path)
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        saved = client.get(base).json
        assert len(saved['variants']) == 3 and saved['active'] == 2
        child = saved['variants'][2]
        state, edited = _edit(client, base, headers, child, op='update', note_id='region',
            values=dict(start=3701, end=4401, lane=2))
        assert_source_identity(edited)
        assert edited['source_evidence'] == child['source_evidence']
        edited_region = next(note for note in edited['notes'] if note['id'] == 'region')
        assert edited_region['end'] > edited_region['start']
        assert edited_region['lane'] == 2
        assert edited_region['voice'] == 'vocals'
        assert edited['source_evidence']['locked'] == child['source_evidence']['locked']
        assert edited['source_evidence']['region'] == child['source_evidence']['region']

        audition_start = edited_region['start'] - 500
        audition = client.post(base + '/audition', headers=headers, json=dict(
            mode='chart', voice='mix', audio='song.wav', start_ms=audition_start,
            duration_ms=2000,
        ))
        assert audition.status_code == 200, audition.json
        sample_rate, audition_samples = wavfile.read(io.BytesIO(audition.data))
        assert sample_rate == 8000
        assert int(audition.headers['X-ManiaLab-Audition-Events']) >= 2
        for chart_time in (edited_region['start'], edited_region['end']):
            event_sample = round((chart_time - audition_start) * sample_rate / 1000)
            local = audition_samples[event_sample:event_sample + 8]
            assert len(local) == 8
            assert np.any(local != 0)

        evidence = client.get(base + '/export/json')
        assert evidence.status_code == 200
        assert evidence.json['variant'] == edited
        assert evidence.json['analysis']['revision'] == current_analysis['revision']
        assert evidence.json['analysis']['separation'] == current_analysis['separation']
        assert evidence.json['variant']['selection'] == edited['selection']
        chart = client.get(base + '/export/osu')
        assert chart.status_code == 200
        assert parse_osu(chart.get_data(as_text=True))['general']['AudioFilename'] == 'song.wav'
        assert [(note.start, note.end, note.lane)
            for note in parse_osu(chart.get_data(as_text=True))['notes']] == [
                (note['start'], note['end'], note['lane']) for note in edited['notes']]
        with zipfile.ZipFile(io.BytesIO(client.get(base + '/export/osz').data)) as archive:
            chart_name = next(name for name in archive.namelist() if name.endswith('.osu'))
            assert archive.read(chart_name) == chart.data
            assert archive.read(parse_osu(chart.data.decode())['general']['AudioFilename']) == music
        assert client.get(base + '/stem/other?context=chart&source_identity=' + reference['identity']).data == old_stem

        first_candidate_id = client.get(base).json['variants'][0]['candidate_id']
        assert client.post(base + '/variant', json={
            'index': 0, 'candidate_id': first_candidate_id}, headers=headers).status_code == 200
        parent = client.get(base).json['variants'][0]
        parent_json = client.get(base + '/export/json')
        assert parent_json.status_code == 200 and parent_json.json['variant'] == parent
        parent_chart = client.get(base + '/export/osu')
        assert parent_chart.status_code == 200
        assert [(note.start, note.end, note.lane)
            for note in parse_osu(parent_chart.get_data(as_text=True))['notes']] == [
                (note['start'], note['end'], note['lane']) for note in parent['notes']]
        with zipfile.ZipFile(io.BytesIO(client.get(base + '/export/osz').data)) as archive:
            chart_name = next(name for name in archive.namelist() if name.endswith('.osu'))
            assert archive.read(chart_name) == parent_chart.data
            assert archive.read(parse_osu(parent_chart.data.decode())['general']['AudioFilename']) == music
        third_candidate_id = client.get(base).json['variants'][2]['candidate_id']
        assert client.post(base + '/variant', json={
            'index': 2, 'candidate_id': third_candidate_id}, headers=headers).status_code == 200
        reopened = client.get(base).json
        assert_source_identity(reopened['variants'][2])
        assert reopened['variants'][2]['selection']['scope'] == 'regional_output'
        reopened_child = reopened['variants'][2]
        assert reopened_child['source_evidence']['locked'] == parent['source_evidence']['locked']
        assert reopened_child['source_evidence']['region']['detected']['voice'] == 'vocals'
        reopened_region = next(note for note in reopened_child['notes'] if note['id'] == 'region')
        assert reopened_region['voice'] == 'vocals'
        assert (reopened_region['start'], reopened_region['end']) == expected_region_times
        child_json = client.get(base + '/export/json')
        assert child_json.status_code == 200 and child_json.json['variant'] == reopened_child
        assert child_json.json['analysis']['revision'] == current_analysis['revision']
        assert child_json.json['analysis']['separation'] == current_analysis['separation']
        assert child_json.json['variant']['source_evidence']['region']['expressed'] == expressed
        child_chart = client.get(base + '/export/osu')
        assert child_chart.status_code == 200
        assert [(note.start, note.end, note.lane)
            for note in parse_osu(child_chart.get_data(as_text=True))['notes']] == [
                (note['start'], note['end'], note['lane']) for note in reopened_child['notes']]
        with zipfile.ZipFile(io.BytesIO(client.get(base + '/export/osz').data)) as archive:
            chart_name = next(name for name in archive.namelist() if name.endswith('.osu'))
            assert archive.read(chart_name) == child_chart.data
            assert archive.read(parse_osu(child_chart.data.decode())['general']['AudioFilename']) == music
        response = client.get(base + '/stem/other?context=chart&source_identity=' + reference['identity'])
        assert response.status_code == 200 and response.data == old_stem
    finally:
        app.extensions['executor'].shutdown()
