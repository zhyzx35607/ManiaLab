from dataclasses import replace
import numpy as np
import pytest

from manialab.audio import rhythmic_evidence, SR, fuse_attacks
from manialab.domain import Note
from manialab.timing import BeatClock, chart_clock_notes, estimate_timing, align_candidates, normalize_bpm, refit_offset, grid_report


def test_original_red_lines_change_period_and_ignore_inherited_velocity():
    clock = BeatClock(['1000,500,4,2,0,70,1,0', '2000,-50,4,2,0,70,0,0',
                       '3000,250,4,2,0,70,1,0', 'bad', '4000,nan,4,2,0,70,1,0'])
    assert [clock(t) for t in (0, 1000, 2999, 3000, 5000)] == [500, 500, 500, 250, 250]
    assert BeatClock([], 400)(1000) == 400


def test_mapping_clock_is_explicit_and_keeps_raw_audio_evidence():
    notes = [Note(115, 865, id='tone'), Note(5, 5, id='intro')]
    before = [n.data() for n in notes]
    converted = chart_clock_notes(notes, 22.5)
    assert [(n.start,n.end) for n in converted] == [(93,843),(0,0)]
    assert [n.data() for n in notes] == before
    assert chart_clock_notes(notes,0) == notes
    with pytest.raises(ValueError):
        chart_clock_notes(notes,float('nan'))


def click_track(bpm, seconds=100, offset=.347):
    y = np.zeros(SR * seconds, dtype=np.float32)
    times = np.arange(offset, seconds - .1, 60 / bpm)
    t = np.arange(500) / SR
    click = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 120)).astype(np.float32)
    for at in times:
        i = round(at * SR)
        y[i:i + len(click)] += click
    return y, times * 1000


@pytest.mark.parametrize('bpm', [90, 120, 137, 174.5, 273.25])
def test_precision_and_last_beat_drift(bpm):
    y, expected = click_track(bpm)
    notes, result = rhythmic_evidence(y)
    assert abs(result['bpm'] - bpm) < .01
    beat = 60000 / result['bpm']
    errors = (expected - result['offset'] + beat / 2) % beat - beat / 2
    assert np.max(np.abs(errors)) < 5
    assert result['stable']
    assert len(notes) >= len(expected) - 1


def test_subdivisions_and_silence_intro():
    beat = 60000 / 133.7
    notes = [Note(round(7347 + i * beat / 2), round(7347 + i * beat / 2),
                  strength=1 if i % 2 == 0 else .3) for i in range(400)]
    result = estimate_timing(notes, 100)
    # A deliberately nonstandard tempo can still be entered exactly by hand.
    match = refit_offset(notes, 133.7)
    assert match['bpm'] == 133.7
    assert match['coverage'] > .99
    assert all(c['bpm'] == normalize_bpm(c['bpm']) for c in result['candidates'])


def test_variable_tempo_does_not_claim_safe_grid():
    times = list(np.arange(347, 60000, 500)) + list(np.arange(60347, 120000, 60000 / 137))
    notes = [Note(round(t), round(t), strength=1) for t in times]
    result = estimate_timing(notes, 120)
    assert not result['stable']
    assert any(s['suspect'] for s in result['sections'])
    with pytest.raises(ValueError, match='Timing'):
        align_candidates(notes, result)


def test_all_endpoints_snap_including_far_attacks_and_releases():
    notes = [Note(1014, 2020, source='pitch'), Note(1170, 1170), Note(1508, 1508), Note(1493, 1498)]
    tempo = dict(bpm=120, offset=0, division=4, stable=True)
    aligned, report = align_candidates(notes, tempo)
    assert aligned[0].start == 1000 and aligned[0].end == 2000
    assert aligned[1].start == 1125
    assert aligned[2].start == aligned[2].end == 1500
    assert aligned[3].start == aligned[3].end == 1500
    assert report['large_attack_shifts'] == 1
    assert report['collapsed_holds'] == 1
    assert grid_report(aligned, tempo)['off_grid'] == 0


@pytest.mark.parametrize('raw,expected', [(210.00073,210),(104.9996,105),(174.501,174.5),
    (166.66687,166.667),(133.3328,133.333),(273.251,273.25),(99.749,99.75),
    (133.7,134),(120.1,120),(119.998,120)])
def test_bpm_music_rounding(raw,expected):
    assert normalize_bpm(raw) == expected


def test_offset_refit_after_bpm_rounding():
    notes = [Note(round(91+i*60000/210),round(91+i*60000/210)) for i in range(1500)]
    result = refit_offset(notes,normalize_bpm(210.00073))
    assert result['bpm'] == 210
    assert result['offset'] == 91
    assert result['coverage'] == 1


def test_real_fractional_tempo_is_not_hidden_by_rounding():
    rng = np.random.default_rng(19)
    expected = np.arange(347, 180000, 60000 / 219.2)
    attacks = [Note(round(t + noise), round(t + noise), strength=1)
               for t, noise in zip(expected, rng.normal(0, 1.5, len(expected)))]
    result = estimate_timing(attacks, 180)
    assert result['bpm'] == 219
    assert not result['stable']
    assert '漂移' in result['reliability']
    precise = next(c for c in result['precise_candidates'] if c['bpm'] == 219.2)
    assert precise['rounding_drift_ms'] > 100
    assert precise['coverage'] > .98
    period = 60000 / precise['bpm']
    errors = (expected - precise['offset'] + period / 2) % period - period / 2
    assert max(abs(errors)) < 3


def test_numerical_noise_does_not_request_fractional_bpm():
    notes = [Note(round(91 + i * 60000 / 210.00073), round(91 + i * 60000 / 210.00073))
             for i in range(800)]
    result = estimate_timing(notes, 230)
    assert result['bpm'] == 210
    assert result['stable']
    assert not result['precise_candidates']


def test_meter_model_guides_beat_unit_without_setting_the_offset():
    attacks = [Note(round(91 + i * 400), round(91 + i * 400), strength=1)
               for i in range(400)]
    # Model estimates deliberately have a 27ms latency. Only spacing is used.
    model_beats = [118 + i * 800 for i in range(200)]
    result = estimate_timing(attacks, 161, model_beats)
    assert result['bpm'] == 75
    assert result['meter_model_used']
    period = 60000 / result['bpm']
    assert abs((result['offset'] - 91 + period / 4) % (period / 2) - period / 4) < 2
    assert any(c['bpm'] == 150 for c in result['candidates'])


def test_sparse_or_irregular_model_beats_cannot_overrule_audio():
    attacks = [Note(round(91 + i * 500), round(91 + i * 500), strength=1)
               for i in range(200)]
    result = estimate_timing(attacks, 101, [100, 900, 4000])
    assert result['bpm'] == 120 and not result['meter_model_used']


@pytest.mark.parametrize('bpm,offset,division', [(210,91,4),(105,91,4),(166.667,-91,3),(174.5,91,4),
    (105,91,8),(174.5,-91,16),(137,91,6),(166.667,25,12)])
def test_serialized_chart_endpoints_on_exported_red_line(tmp_path,bpm,offset,division):
    from pathlib import Path
    from manialab.domain import export_osu,parse_osu
    tempo=dict(bpm=bpm,offset=offset,division=division,stable=True)
    tick=60000/bpm/division
    candidates=[Note(round(1000+i*503),round(1271+i*503),lane=i%10) for i in range(300)]
    notes,report=align_candidates(candidates,tempo)
    path=Path(__file__).parents[1]/'vendor/mapperatorinator/template.osu'
    text=export_osu(notes,10,path,bpm=bpm,offset=offset,grid_division=division)
    assert f'BeatDivisor: {division}' in text
    parsed=parse_osu(text)
    red=next(p.split(',') for p in parsed['timing'] if p.strip())
    export_offset,export_tick=float(red[0]),float(red[1])/division
    for n in parsed['notes']:
        for t in [n.start,n.end]:
            nearest=export_offset+round((t-export_offset)/export_tick)*export_tick
            assert abs(t-nearest) <= .501
    # A corrupted tail must fail the *export*, even if no lane overlaps exist.
    corrupt=[replace(notes[0],end=notes[0].end+8)]+notes[1:]
    with pytest.raises(ValueError,match='落线检查失败'):
        export_osu(corrupt,10,path,bpm=bpm,offset=offset,grid_division=division)


def test_nonnegative_first_grid_and_disabled_bypass():
    notes=[Note(0,0),Note(1,40),Note(96,99)]
    tempo=dict(bpm=105,offset=91,division=4,stable=True)
    aligned,_=align_candidates(notes,tempo)
    assert all(n.start==91 and n.end==91 for n in aligned)
    assert grid_report(aligned,tempo)['off_grid']==0
    with pytest.raises(ValueError,match='完整拍线量化'):
        align_candidates(notes,tempo,enabled=False)


def test_attack_fusion_keeps_independent_transients():
    pitch = [Note(1018, 3000, pitch=60, source='pitch'), Note(1022, 4000, pitch=67, source='pitch')]
    transients = [Note(1000, 1000, source='transient'), Note(2000, 2000, source='transient')]
    merged, unmatched, fused = fuse_attacks(pitch, transients)
    assert [n.start for n in merged] == [1000, 1000]
    assert [n.end for n in merged] == [3000, 4000]
    assert unmatched == transients[1:] and fused == 1
