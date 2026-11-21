"""Phrase source evidence remains consistent through generation and reopen."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import time
import wave

from manialab.arrange import parameters
from manialab.domain import Note, parse_osu
from manialab.server import create_app
from manialab.structure import edit_boundaries
from manialab.voices import reconcile_phrase_sources


ROOT = Path(__file__).resolve().parents[1]


def _wave_bytes():
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
        audio.writeframes(b'\0\0' * (22050 * 4))
    return output.getvalue()


def _wait_for_job(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + job_id).json
        if job['status'] not in ('queued', 'running'):
            assert job['status'] == 'done', job
            return
        time.sleep(.01)
    raise AssertionError('Candidate generation did not finish')


def test_reconciled_phrase_evidence_survives_generation_json_and_reopen(tmp_path):
    music = _wave_bytes()
    audio_hash = hashlib.sha256(music).hexdigest()
    anchor = Note(1000, 1200, pitch=60, strength=.8, source='pitch',
                  voice='vocals', id='vocals:anchor')
    removed_support = Note(1500, 1600, pitch=64, strength=.4, source='pitch',
                           voice='vocals', id='vocals:removed')
    phrase = Note(1000, 2500, pitch=60, strength=.8, source='pitch',
                  voice='vocals', id='vocals:anchor', phrase=True,
                  support=(anchor.id, removed_support.id), support_coverage=1,
                  release_reason='vocal_activity_decay')
    surviving_phrase = reconcile_phrase_sources([phrase], [anchor])[0]
    assert surviving_phrase.support == (anchor.id,)
    assert surviving_phrase.support_coverage == phrase.support_coverage
    assert surviving_phrase.end == phrase.end

    app = create_app(ROOT, tmp_path)
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(music), 'phrase.wav')})
        assert imported.status_code == 200, imported.json
        state = imported.json
        state['analysis'] = dict(
            revision=14, duration=4, audio_sha256=audio_hash,
            tempo=dict(bpm=120, offset=0, division=4, stable=True),
            candidates=[anchor.data()],
            phrase_candidates=[surviving_phrase.data()],
            structure=edit_boundaries({}, [], dict(duration=4, candidates=[anchor.data()])),
            separation=dict(source='provided', voices=['vocals'],
                audio_sha256=audio_hash, set_sha256='a' * 64),
        )
        projects = app.extensions['projects']
        projects.save(state)
        base = '/api/projects/' + state['id']

        response = client.post(base + '/generate', headers=headers, json=dict(
            mode='audio', settings=parameters({'keys': 6, 'density': 8,
                'ln_ratio': 1, 'phrase_ln': 1}),
            bpm=120, offset=0, division=4, audio_shift_ms=0,
            timing_confirmed=True,
        ))
        assert response.status_code == 200, response.json
        _wait_for_job(client, response.json['id'])

        exported = client.get(base + '/export/json')
        assert exported.status_code == 200
        variant = exported.json['variant']
        evidence = variant['source_evidence'][anchor.id]
        assert evidence['detected']['end'] == anchor.end
        assert evidence['expressed']['phrase'] is True
        assert evidence['expressed']['support'] == [anchor.id]
        assert evidence['expressed']['support_coverage'] == 1
        assert evidence['expressed']['end'] == phrase.end
        chart = parse_osu(client.get(base + '/export/osu').get_data(as_text=True))
        assert [(note.start, note.end) for note in chart['notes']] == [(anchor.start, phrase.end)]
        saved_analysis = deepcopy(exported.json['analysis'])
        assert saved_analysis['phrase_candidates'][0]['support'] == [anchor.id]
        pid = state['id']
    finally:
        app.extensions['executor'].shutdown()

    reopened = create_app(ROOT, tmp_path)
    try:
        client = reopened.test_client()
        history = client.get('/api/projects/' + pid).json
        assert history['analysis']['phrase_candidates'][0]['support'] == [anchor.id]
        reopened_export = client.get('/api/projects/' + pid + '/export/json')
        assert reopened_export.status_code == 200
        reopened_variant = reopened_export.json['variant']
        assert reopened_variant['source_evidence'][anchor.id] == evidence
        assert reopened_variant['notes'] == variant['notes']
        reopened_chart = parse_osu(client.get('/api/projects/' + pid + '/export/osu').get_data(as_text=True))
        assert [(note.start, note.end) for note in reopened_chart['notes']] == [(anchor.start, phrase.end)]
    finally:
        reopened.extensions['executor'].shutdown()
