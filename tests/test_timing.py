from dataclasses import replace
import numpy as np
import pytest

from manialab.audio import rhythmic_evidence, SR, fuse_attacks
from manialab.domain import Note
from manialab.timing import estimate_timing, align_candidates


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
    # Meter ambiguity must remain available, and fitted candidates cannot drift.
    match = min(result['candidates'], key=lambda c: abs(c['bpm'] - 133.7))
    assert abs(match['bpm'] - 133.7) < .01
    assert match['coverage'] > .99


def test_variable_tempo_does_not_claim_safe_grid():
    times = list(np.arange(347, 60000, 500)) + list(np.arange(60347, 120000, 60000 / 137))
    notes = [Note(round(t), round(t), strength=1) for t in times]
    result = estimate_timing(notes, 120)
    assert not result['stable']
    assert any(s['suspect'] for s in result['sections'])
    aligned, report = align_candidates(notes, result)
    assert aligned == notes and not report['enabled']


def test_snap_is_bounded_and_sustain_not_extended():
    notes = [Note(1014, 2020, source='pitch'), Note(1170, 1170), Note(1508, 1508), Note(1493, 1498)]
    tempo = dict(bpm=120, offset=0, division=4, stable=True)
    aligned, report = align_candidates(notes, tempo)
    assert aligned[0].start == 1000 and aligned[0].end == 2000
    assert aligned[1].start == 1170
    assert aligned[2].start == aligned[2].end == 1500
    assert report['max_shift_ms'] <= 30
    assert all(a.end <= n.end for a, n in zip(aligned, notes))


def test_attack_fusion_keeps_independent_transients():
    pitch = [Note(1018, 3000, pitch=60, source='pitch'), Note(1022, 4000, pitch=67, source='pitch')]
    transients = [Note(1000, 1000, source='transient'), Note(2000, 2000, source='transient')]
    merged, unmatched, fused = fuse_attacks(pitch, transients)
    assert [n.start for n in merged] == [1000, 1000]
    assert [n.end for n in merged] == [3000, 4000]
    assert unmatched == transients[1:] and fused == 1
