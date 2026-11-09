from copy import deepcopy

import pytest

from scripts.transkun_onset_evidence import assign_onsets, collect_onset_evidence


def note(start, end, pitch=60, *, onset=True, offset=True):
    return dict(onset=start, offset=end, pitch=pitch, velocity=64,
                has_onset=onset, has_offset=offset)


def trace(final, *windows):
    return dict(sample_rate=44100, duration_seconds=10., final_events=final, pedal_events=[],
                windows=[dict(index=i, origin_seconds=0., events=[batch])
                         for i, batch in enumerate(windows)])


def test_replaced_onset_survives_with_explicitly_unresolved_release():
    source = trace([note(2., 3.)], [note(1., 2.5)], [note(2., 3.)])
    before = deepcopy(source)
    result = collect_onset_evidence(source)
    assert [c['note']['onset'] for c in result['candidates']] == [1., 2.]
    assert not any(c['ln_release_allowed'] for c in result['candidates'])
    assert result['candidates'][0]['note']['offset'] == 2.5  # estimate, not LN
    assert source == before


def test_incomplete_nonfinal_event_keeps_onset():
    result = collect_onset_evidence(trace([note(4., 5.)], [note(1., 2., offset=False), note(4., 5.)]))
    candidate = result['candidates'][0]
    assert candidate['note']['onset'] == 1.
    assert not candidate['note']['has_offset']
    assert candidate['display_interval_only'] and not candidate['physical_release_confirmed']


def test_cross_window_duplicate_uses_fixed_anchor_and_preserves_official_time():
    result = collect_onset_evidence(trace([note(1., 2.)], [note(1.02, 2.1)], [note(1.04, 2.2)]))
    assert [c['note']['onset'] for c in result['candidates']] == [1., 1.04]
    assert result['candidates'][0]['onset_sources'][0]['onset'] == 1.02


def test_two_close_reattacks_in_one_window_do_not_merge():
    result = collect_onset_evidence(trace([note(1., 2.)], [note(1., 1.01), note(1.01, 2.)]))
    assert len(result['candidates']) == 2
    assert all(len(c['onset_sources']) == 1 for c in result['candidates'])


def test_global_clock_retains_padding_and_pedals_separate():
    source = trace([], [note(.5, 1.), note(12., 12.1), note(1., 2., -64)])
    source['windows'][0]['origin_seconds'] = -1.
    source['pedal_events'] = [note(3., 4., -64)]
    result = collect_onset_evidence(source)
    assert [c['note']['onset'] for c in result['candidates']] == [-.5, 11.]
    assert all(c['onset_outside_input'] for c in result['candidates'])
    assert result['pedal_events'] == source['pedal_events']


def test_orphan_continuation_is_not_an_onset():
    assert not collect_onset_evidence(trace([], [note(1., 2., onset=False)]))['candidates']


def test_maximum_assignment_beats_nearest_first_greedy():
    proposed = [note(.018, 1.), note(.04, 1.)]
    existing = [note(0., 1.), note(.022, 1.)]
    assert assign_onsets(proposed, existing, .023) == [(0, 0), (1, 1)]


def test_official_forced_offset_never_becomes_release_permission():
    result = collect_onset_evidence(trace([note(1., 2.)], [note(1., 2., offset=False)]))
    candidate = result['candidates'][0]
    assert candidate['note']['has_offset'] and not candidate['onset_sources'][0]['has_offset']
    assert not candidate['ln_release_allowed']


@pytest.mark.parametrize('mutate', [
    lambda t: t.update(sample_rate=16000),
    lambda t: t['windows'][0].update(index=1),
    lambda t: t['windows'][0].update(origin_seconds=float('nan')),
    lambda t: t['windows'][0]['events'][0][0].update(onset=float('nan')),
    lambda t: t['windows'][0]['events'][0][0].update(pitch=109),
    lambda t: t['windows'][0]['events'][0][0].update(offset=0.),
])
def test_invalid_inputs_rejected(mutate):
    source = trace([], [note(1., 2.)])
    mutate(source)
    with pytest.raises(ValueError):
        collect_onset_evidence(source)
