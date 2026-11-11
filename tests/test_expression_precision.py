from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from scripts.check_expression_precision import reconstruct


def decision(identity, group, selected=True, reason='supported_sustain'):
    return dict(id=identity, group=group, decision='selected' if selected else 'omitted', reason=reason)


def test_historical_witness_recovers_chord_depth_and_excludes_later_drum_fallback():
    a, b, drum = [SimpleNamespace(id=identity) for identity in ('a', 'b', 'd')]
    units = [[a], [a, b], [drum]]
    # First two columns are mutually exclusive chord layers. The last
    # column rewards either selected chord layer, never a missing one.
    matrix = csc_matrix([[1, 1, 0, 0], [-1, -1, 0, 1]])
    cost = np.array([-.15, -.3, -1., -.85])
    report = dict(decisions=[decision('a', 0), decision('b', 0),
        decision('d', 1, reason='drum_group_fallback')], selection_solver=dict(objective=1.15))
    chosen, score = reconstruct(report, units, cost, matrix, np.array([1., 0.]))
    np.testing.assert_array_equal(chosen, [0, 1, 0, 1])
    assert score == pytest.approx(1.15)


def test_unselected_strong_option_cannot_lend_its_threshold_reward():
    a, b = [SimpleNamespace(id=identity) for identity in ('weak', 'strong')]
    matrix = csc_matrix([[-1, -1, 1, 0], [0, -1, 0, 1], [1, 1, 0, 0]])
    cost = np.array([-.1, -.2, -.4, -.5])
    report = dict(decisions=[decision('weak', 0), decision('strong', 1, False)],
        selection_solver=dict(objective=.5))
    chosen, _ = reconstruct(report, [[a], [b]], cost, matrix, np.array([0., 0., 1.]))
    np.testing.assert_array_equal(chosen, [1, 0, 1, 0])


@pytest.mark.parametrize('cap, objective', [(0., 1.), (1., 2.)])
def test_infeasible_or_wrong_objective_is_not_accepted_as_a_witness(cap, objective):
    report = dict(decisions=[decision('a', 0)], selection_solver=dict(objective=objective))
    with pytest.raises(AssertionError):
        reconstruct(report, [[SimpleNamespace(id='a')]], np.array([-1.]),
            csc_matrix([[1.]]), np.array([cap]))
