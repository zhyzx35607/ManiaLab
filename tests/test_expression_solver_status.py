import copy

import pytest

from manialab import expression
from manialab.domain import Note


def run_selection():
    notes = [Note(0, 1000, strength=1., source='pitch', voice='bass', id='bass',
                  release_reason='model_predicted_offset'),
             Note(0, 0, strength=.5, source='transient', voice='drums', id='d0'),
             Note(400, 400, strength=.5, source='transient', voice='drums', id='d1')]
    metadata = {n.id: dict(section=0, role='support') for n in notes}
    trace = {n.id: dict(id=n.id, start=n.start, mode='sustain' if n.hold else 'onsets',
                       reason='supported_sustain' if n.hold else 'section_onsets') for n in notes}
    return expression.select_expression(notes, metadata, trace, dict(density=2), 500, sustain_coverage=.25)


def test_short_group_tradeoff_is_not_allowed_a_song_length_relative_gap(monkeypatch):
    original, calls = expression._solve_binary, []
    def capture(*args, **kwargs):
        calls.append(copy.deepcopy(kwargs['options']))
        return original(*args, **kwargs)
    monkeypatch.setattr(expression, '_solve_binary', capture)
    notes, report = run_selection()
    assert {n.id for n in notes} == {'bass', 'd0'}
    assert report['drum_fallback_solver']['selected'] == 1
    assert calls == [dict(time_limit=15., mip_rel_gap=0.), dict(time_limit=5., mip_rel_gap=0.)]


@pytest.mark.parametrize('stage,status,gap,optimal', [
    (0, 0, .004, False), (1, 0, .004, False), (0, 1, 0., False), (0, 0, 0., True)])
def test_solver_success_with_nonzero_gap_is_not_reported_as_proven_optimal(
        monkeypatch, stage, status, gap, optimal):
    original, calls = expression._solve_binary, []
    def annotate(*args, **kwargs):
        solved = original(*args, **kwargs)
        if len(calls) == stage:
            solved.status, solved.mip_gap = status, gap
        calls.append(solved)
        return solved
    monkeypatch.setattr(expression, '_solve_binary', annotate)
    notes, report = run_selection()
    assert notes and report['attack_optimal'] is optimal
