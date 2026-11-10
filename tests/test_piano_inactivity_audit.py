import numpy as np

from scripts.audit_piano_inactivity_context import candidate_gaps, coverage, sampled_low_intervals


def test_reference_coverage_is_union_clipped_to_the_gap():
    labels = [dict(onset=.1, offset=.3, key_offset=.15),
              dict(onset=.2, offset=.6, key_offset=.5)]
    assert coverage(labels, 0, 1000, 'offset') == 500
    assert coverage(labels, 250, 400, 'offset') == 150
    assert coverage(labels, 0, 1000, 'key_offset') == 350
    assert coverage(labels, 600, 1000, 'offset') == 0


def test_reference_key_and_pedal_activity_are_not_equated():
    labels = [dict(onset=0, offset=2, key_offset=.5)]
    assert coverage(labels, 600, 1000, 'key_offset') == 0
    assert coverage(labels, 600, 1000, 'offset') == 400


def test_candidate_gap_audit_cannot_skip_a_long_earlier_overlap():
    notes = [dict(start=0, end=4000), dict(start=1000, end=1500), dict(start=2000, end=2500)]
    assert candidate_gaps(notes, [[0, 5000]], 5000) == []


def test_candidate_gap_requires_internal_supported_time_and_next_attack_in_window():
    notes = [dict(start=100, end=300), dict(start=600, end=900), dict(start=2000, end=2200)]
    assert candidate_gaps(notes, [[300, 379]], 2000) == []
    assert candidate_gaps(notes, [[300, 400], [1000, 1500]], 2000) == [
        dict(candidate_gap=[300, 600], inactive_intervals=[[300, 400]])]


def test_independent_frame_scan_keeps_exact_sampled_endpoints():
    frame, onset = np.zeros((30, 88)), np.zeros((30, 88))
    frame[9, 10] = .11;onset[20, 60] = .31
    assert sampled_low_intervals(frame, onset) == [[0, 80], [100, 190], [210, 290]]
