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
        'base_created': candidate['created'], **changes,
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
        assert archive.read('ManiaLab.osu') == chart.data
        assert archive.read('song.wav') == music
    parsed = parse_osu(chart.get_data(as_text=True))
    assert [(note.start, note.end, note.lane) for note in parsed['notes']] == [
        (note['start'], note['end'], note['lane']) for note in candidate['notes']
    ]
    assert parsed['general']['AudioFilename'] == 'song.wav'


@pytest.mark.parametrize('regional', [False, True])
def test_source_snapshots_survive_successive_edits_exports_and_reopening(tmp_path, regional):
    music = _wave()
    old_stem = _wave(800)
    current_stem = _wave(2400)
    saved_source = dict(source='provided', audio_sha256=hashlib.sha256(music).hexdigest(),
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
        assert evidence_export['source_evidence']['tap'] == deleted_evidence
        assert evidence_export['edits'][-1]['before']['id'] == 'tap'
        assert not any(note['id'] == 'tap' for note in evidence_export['notes'])
        deleted_chart_note = (2500, 2500, 1)
        chart = parse_osu(client.get(base + '/export/osu').get_data(as_text=True))
        assert deleted_chart_note not in [(note.start, note.end, note.lane) for note in chart['notes']]
        with zipfile.ZipFile(io.BytesIO(client.get(base + '/export/osz').data)) as archive:
            bundled_chart = parse_osu(archive.read('ManiaLab.osu').decode())
        assert deleted_chart_note not in [
            (note.start, note.end, note.lane) for note in bundled_chart['notes']
        ]
        response = client.get(base + '/stem/other?context=chart&source_identity=' + reference['identity'])
        assert response.status_code == 200 and response.data == old_stem
        restored = client.post(base + '/variant', json={'index': 0}, headers=headers)
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
        restored = client.post(base + '/variant', json={'index': 3}, headers=headers)
        assert restored.status_code == 200
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
    parent_candidates = [
        Note(1000, 1500, id='locked', source='pitch', voice='other',
             release_reason='frame_decay').data(),
        Note(1700, 2200, id='support', source='pitch', voice='other',
             release_reason='frame_decay').data(),
        Note(3500, 4200, id='region', source='pitch', voice='other',
             release_reason='frame_decay').data(),
    ]
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
        state['analysis'] = dict(analysis, phrase_candidates=[vocal_phrase],
            candidates=[changed_locked,
                Note(1700, 2300, id='support', source='pitch', voice='other',
                    release_reason='frame_decay').data(),
                Note(2500, 2900, id='outside', source='pitch', voice='other',
                    release_reason='frame_decay').data(),
                vocal_anchor, vocal_support],
            structure=dict(boundaries=[0, 5000], segments=[dict(start_ms=0, end_ms=5000)]))
        projects.save(state)
        state, child = _generate(client, base, headers, dict(mode='audio', settings=settings,
            timing_confirmed=True, region=[3000, 5000], **clock))
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
        assert child['source_separation'] == reference
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
        assert len(saved['variants']) == 2 and saved['active'] == 1
        child = saved['variants'][1]
        state, edited = _edit(client, base, headers, child, op='update', note_id='region',
            values=dict(start=3701, end=4401, lane=2))
        assert edited['source_separation'] == reference
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
            assert archive.read('ManiaLab.osu') == chart.data
            assert archive.read('song.wav') == music
        assert client.get(base + '/stem/other?context=chart&source_identity=' + reference['identity']).data == old_stem

        assert client.post(base + '/variant', json={'index': 0}, headers=headers).status_code == 200
        parent = client.get(base).json['variants'][0]
        parent_json = client.get(base + '/export/json')
        assert parent_json.status_code == 200 and parent_json.json['variant'] == parent
        parent_chart = client.get(base + '/export/osu')
        assert parent_chart.status_code == 200
        assert [(note.start, note.end, note.lane)
            for note in parse_osu(parent_chart.get_data(as_text=True))['notes']] == [
                (note['start'], note['end'], note['lane']) for note in parent['notes']]
        with zipfile.ZipFile(io.BytesIO(client.get(base + '/export/osz').data)) as archive:
            assert archive.read('ManiaLab.osu') == parent_chart.data
            assert archive.read('song.wav') == music
        assert client.post(base + '/variant', json={'index': 1}, headers=headers).status_code == 200
        reopened = client.get(base).json
        assert reopened['variants'][1]['source_separation'] == reference
        assert reopened['variants'][1]['selection']['scope'] == 'regional_output'
        reopened_child = reopened['variants'][1]
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
            assert archive.read('ManiaLab.osu') == child_chart.data
            assert archive.read('song.wav') == music
        response = client.get(base + '/stem/other?context=chart&source_identity=' + reference['identity'])
        assert response.status_code == 200 and response.data == old_stem
    finally:
        app.extensions['executor'].shutdown()
