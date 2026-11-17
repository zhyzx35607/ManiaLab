"""Diagnostics belong to original audio windows even when the chart shifts."""
from copy import deepcopy
import io
from pathlib import Path

import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.generation import generate_variant
from manialab.server import create_app
from manialab.timing import analysis_timing_diagnostics, chart_clock_notes, grid_diagnostics

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('times,shift', [([12010,12100,12200,12300,12400],22.5),
    ([11600,11700,11800,11900,11990],-50)])
def test_diagnostic_window_membership_survives_chart_shift(times,shift):
    notes = [Note(t,t,id=str(t)) for t in times]
    original = grid_diagnostics(notes,120,0,4)
    shifted = grid_diagnostics(chart_clock_notes(notes,shift),120,-round(shift),4)
    assert len(shifted) == len(original) == 1
    assert [(s['start_ms'],s['end_ms'],s['attacks']) for s in shifted] == [
        (s['start_ms'],s['end_ms'],s['attacks']) for s in original]
    assert shifted[0]['suspect'] == original[0]['suspect']


def analysis_for_diagnostics(times=(12010,12100,12200,12300,12400),duration=24):
    return dict(revision=8,duration=duration,candidates=[Note(12500,13000,id='tone',source='pitch').data()],
        phrase_candidates=[],timing_attacks=[Note(t,t,id=str(t)).data() for t in times],
        tempo=dict(bpm=120,offset=0,division=4,stable=False,evidence_source='drums',
            timing_coverage_revision=1,sections=[dict(start_ms=12000,end_ms=24000,
                attacks=len(times),coverage=.1,phase_error_ms=99,suspect=True,fallback=True,evidence_source='mix')]))


@pytest.fixture
def project(tmp_path):
    app = create_app(ROOT,tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token':app.config['TOKEN']}
    state = client.post('/api/import',data={'files':(io.BytesIO(b'fixture'),'song.wav')},headers=headers).json
    state['analysis'] = analysis_for_diagnostics()
    app.extensions['projects'].save(state)
    yield app,client,headers,state
    app.extensions['executor'].shutdown()


@pytest.mark.parametrize('route', ['timing-check','timing-refit'])
def test_http_diagnostic_keeps_audio_windows_and_source_but_recalculates_quality(project,route):
    app,client,headers,state = project
    before = deepcopy(app.extensions['projects'].load(state['id'])['analysis'])
    result = client.post(f"/api/projects/{state['id']}/{route}",json=dict(bpm=120,offset=-22,
        division=4,audio_shift_ms=22.5),headers=headers)
    assert result.status_code == 200,result.json
    report = result.json
    assert len(report['sections']) == 1
    section = report['sections'][0]
    assert section['start_ms'] == 12000 and section['attacks'] == 5
    assert section['fallback'] is True and section['evidence_source'] == 'mix'
    assert section['phase_error_ms'] != 99 and section['coverage'] != .1
    assert report['window_clock'] == 'audio'
    assert report['clock']['audio_shift_ms'] == 22.5
    assert report['status'] == 'partial'
    assert report['insufficient_windows'] == [dict(start_ms=0,end_ms=12000,attacks=0)]
    assert app.extensions['projects'].load(state['id'])['analysis'] == before


def test_sparse_http_check_reports_unassessed_evidence(project):
    app,client,headers,state = project
    state['analysis'] = analysis_for_diagnostics(times=[1000],duration=12)
    app.extensions['projects'].save(state)
    report = client.post(f"/api/projects/{state['id']}/timing-check",json={},headers=headers).json
    assert report['sections'] == [] and report['status'] == 'insufficient'
    assert report['insufficient_windows'] == [dict(start_ms=0,end_ms=12000,attacks=1)]


def test_generated_diagnostics_match_http_check_without_changing_evidence(project):
    app,client,headers,state = project
    tempo = dict(bpm=120,offset=-22,division=4,audio_shift_ms=22.5,stable=False)
    report = client.post(f"/api/projects/{state['id']}/timing-check",json=tempo,headers=headers).json
    original = deepcopy(state['analysis'])
    generate_variant(state,dict(mode='audio',snap=True,timing_confirmed=True),parameters(),tempo,
        lambda *_:None,lambda:False)
    variant = state['variants'][-1]
    assert variant['timing_sections'] == report['sections']
    assert variant['timing_diagnostics'] == report
    assert state['analysis'] == original


def test_unordered_attacks_do_not_drop_later_windows():
    notes = [Note(t,t) for t in [12500,13000,13500,14000,14500,1000]]
    assert grid_diagnostics(notes,120,0)[0]['start_ms'] == 12000


def test_complete_diagnostics_do_not_mean_automatic_timing_acceptance():
    from manialab.timing import align_candidates
    analysis = analysis_for_diagnostics(times=[1000,1500,2000,2500,3000],duration=6)
    tempo = dict(bpm=120,offset=0,division=4,audio_shift_ms=0,stable=False)
    original = deepcopy(analysis)
    report = analysis_timing_diagnostics(analysis,tempo)
    assert report['status'] == 'available' and report['suspect'] == 0
    assert not report['insufficient_windows']
    with pytest.raises(ValueError,match='Timing'):
        align_candidates([Note(1000,1000)],tempo)
    assert analysis == original and tempo['stable'] is False


def test_partial_last_window_reports_its_actual_duration():
    analysis = analysis_for_diagnostics(times=[1000,1500,2000,2500,3000,12500],duration=13)
    report = analysis_timing_diagnostics(analysis,dict(bpm=120,offset=0))
    assert report['status'] == 'partial'
    assert report['insufficient_windows'] == [dict(start_ms=12000,end_ms=13000,attacks=1)]


def test_legacy_analysis_without_source_labels_does_not_invent_provenance():
    analysis = analysis_for_diagnostics()
    analysis['tempo'].pop('sections')
    report = analysis_timing_diagnostics(analysis,dict(bpm=120,offset=0))
    assert 'fallback' not in report['sections'][0]
    assert 'evidence_source' not in report['sections'][0]


@pytest.mark.parametrize('shift', [0,22.5,50,-50])
def test_real_fallback_assembly_keeps_source_after_shift(shift):
    import math
    from manialab.timing import cover_missing_timing_windows
    primary = [Note(t,t,id=f'd{t}') for t in range(12000,24000,500)]
    mixed = [Note(t,t,id=f'm{t}') for t in range(20,24000,500)]
    tempo,attacks = cover_missing_timing_windows(dict(bpm=120,offset=0,division=4,stable=True),primary,mixed)
    analysis = dict(duration=24,tempo=tempo,timing_attacks=[n.data() for n in attacks])
    report = analysis_timing_diagnostics(analysis,dict(bpm=120,offset=math.floor(-shift+.5),audio_shift_ms=shift))
    assert report['status'] == 'available'
    assert [(s['start_ms'],s['attacks'],s['evidence_source'],s['fallback']) for s in report['sections']] == [
        (0,24,'mix',True),(12000,24,'drums',False)]
