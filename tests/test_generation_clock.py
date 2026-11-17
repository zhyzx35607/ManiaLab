"""Regressions for musical section membership across the two clocks."""
import json
import io
from pathlib import Path
import time

import pytest

from manialab.arrange import parameters
from manialab.domain import Note, note_from, parse_osu
from manialab.preferences import ChartPreferences, row_settings, settings_at
from manialab.server import create_app
from manialab.timing import align_candidates, chart_clock_notes


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('shift,offset', [(0, 0), (22.5, -22), (50, -50), (-50, 50), (0, -50)])
def test_generation_keeps_original_section_after_shift_and_quantization(tmp_path, structured, shift, offset):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    try:
        state = client.post('/api/import', data={'files': (io.BytesIO(b'fixture'), 'song.wav')}, headers=headers).json
        notes = [Note(5990, 5990, source='pitch', voice='vocals', id='muted').data(),
                 Note(6000, 6500, source='pitch', voice='vocals', id='boundary').data(),
                 Note(7000, 7500, source='pitch', voice='vocals', id='late').data()]
        analysis = dict(revision=3, duration=12, candidates=notes, phrase_candidates=[], timing_attacks=[],
                        tempo=dict(bpm=120, offset=0, division=4, stable=True))
        intervals = [dict(start_ms=0, end_ms=6000), dict(start_ms=6000, end_ms=12000)]
        if structured:
            analysis['structure'] = dict(segments=intervals)
        state['analysis'] = analysis
        state['plan'] = dict(segments=[dict(s, preferences={'vocals_weight': w}) for s, w in zip(intervals, [0, 1])])
        original = json.loads(json.dumps(analysis))
        projects = app.extensions['projects']
        projects.save(state)
        url = '/api/projects/' + state['id']
        response = client.post(url + '/generate', json=dict(mode='audio', settings={'ln_ratio': 1},
            bpm=120, offset=offset, division=4, audio_shift_ms=shift, timing_confirmed=True), headers=headers)
        assert response.status_code == 200, response.json
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get('/api/jobs/' + response.json['id']).json
            if job['status'] in ('done', 'error'):
                break
            time.sleep(.01)
        assert job['status'] == 'done', job
        saved = client.get(url).json
        candidate = saved['variants'][-1]
        assert {n['id'] for n in candidate['notes']} == {'boundary', 'late'}
        assert {n['id']: n['audio_start'] for n in candidate['notes']} == {'boundary': 6000, 'late': 7000}
        assert saved['analysis'] == original
        assert candidate['report']['grid']['off_grid'] == 0
        exported = client.get(url + '/export/osu')
        assert exported.status_code == 200
        assert len(parse_osu(exported.data.decode())['notes']) == 2
    finally:
        app.extensions['executor'].shutdown()


def test_zero_clipping_and_quantization_do_not_lose_audio_membership():
    prefs = parameters()
    segments = [dict(start_ms=0, end_ms=20, preferences={'vocals_weight': 0, 'chord_size': 2}),
                dict(start_ms=20, end_ms=1000, preferences={'vocals_weight': 1, 'chord_size': 4})]
    notes = [Note(10, 10, id='before'), Note(20, 20, id='after')]
    shifted = chart_clock_notes(notes, 50)
    aligned, _ = align_candidates(shifted, {'bpm': 120, 'offset': 0, 'stable': True})
    timeline = ChartPreferences(segments, 50)
    assert [n.start for n in aligned] == [0, 0]
    assert [settings_at(prefs, n, timeline)['vocals_weight'] for n in aligned] == [0, 1]
    assert row_settings(prefs, aligned, timeline)['chord_size'] == 2
    assert settings_at(prefs, 0, timeline)['vocals_weight'] == 1
    assert [n.audio_start for n in notes] == [None, None]


@pytest.mark.parametrize('engine', ['beam', 'classic'])
def test_layout_uses_original_onset_for_section_preferences(engine):
    from manialab.arrange import assign_lanes
    notes = chart_clock_notes([Note(6000, 6000, id='boundary', source='pitch')], 50)
    prefs = parameters({'layout_engine': engine, 'pattern_prior': 0})
    segments = [dict(start_ms=0, end_ms=6000, preferences={'pitch_follow': 0}),
                dict(start_ms=6000, end_ms=12000, preferences={'pitch_follow': 1})]
    timeline = ChartPreferences(segments, 50)
    assert row_settings(prefs, notes, timeline)['pitch_follow'] == 1
    local = prefs | {'pitch_follow': 1}
    assert assign_lanes(notes, prefs, segment_preferences=timeline) == assign_lanes(notes, local)


@pytest.mark.parametrize('bad', [-1, True, '6000', float('nan')])
def test_invalid_original_audio_onset_is_rejected(bad):
    with pytest.raises(ValueError, match='原始音频起点'):
        note_from(dict(start=100, end=100, audio_start=bad))
