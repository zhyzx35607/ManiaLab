"""Candidate edits and exports keep saved sources after analysis changes."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import time
import wave
import zipfile

import pytest

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
        source_evidence={'tone': dict(detected=raw, expressed=dict(raw, end=1895))})
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
        state, final = _edit(client, base, headers, added, op='delete', note_id='tap')
        assert [(note['start'], note['end']) for note in final['notes']] == [(1125, 2375), (3000, 3500)]
        manual = next(note for note in final['notes'] if note['source'] == 'manual')
        assert manual['id'] not in final['source_evidence']
        for version in state['variants']:
            assert version['source_separation'] == reference
            assert version['source_evidence'] == parent['source_evidence']
        assert state['variants'][0] == parent
        assert final['meter'] == parent['meter']
        _assert_exports(client, base, final, current_analysis, music)
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
