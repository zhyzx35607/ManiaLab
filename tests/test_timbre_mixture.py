import numpy as np
import pytest

from manialab.timbre_mixture import decompose, prototypes


def test_quiet_overlapping_component_survives_louder_drum():
    atoms = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    examples = np.asarray([[10., 0., 2., 0.], [0., 4., 0., 0.], [0., 0., 0., 3.], [0., 0., 0., 0.]])
    scores, residual, weights = decompose(examples, atoms, np.arange(3), 3)
    np.testing.assert_allclose(scores[0], [10 / np.sqrt(104), 0, 2 / np.sqrt(104)])
    np.testing.assert_allclose(scores[1], [0, 1, 0])
    np.testing.assert_allclose(scores[2:], 0)
    np.testing.assert_allclose(residual, [0, 0, 1, 0])
    np.testing.assert_allclose((weights @ atoms)[:2], examples[:2] / np.linalg.norm(examples[:2], axis=1)[:, None])
    np.testing.assert_allclose(decompose(examples * 3, atoms, np.arange(3), 3)[0], scores)


def test_multiple_timbres_share_one_category_without_exclusive_softmax():
    atoms = np.eye(3)
    scores, _, _ = decompose([[1, 1, 1]], atoms, np.array([0, 1, 1]), 2)
    np.testing.assert_allclose(scores[0], [1 / np.sqrt(3), 2 / np.sqrt(3)])
    assert scores[0].sum() > 1
    result = prototypes([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0]], 2)
    assert len(result) == 2
    np.testing.assert_allclose(result @ result.T, np.eye(2))


def test_invalid_evidence_is_not_coerced_to_silence():
    with pytest.raises(ValueError):
        decompose([[np.nan, 1]], [[1, 0]], np.array([0]), 1)
    with pytest.raises(ValueError):
        decompose([[1, -1]], [[1, 0]], np.array([0]), 1)
    with pytest.raises(ValueError):
        decompose([[1, 1]], [[1, 0]], np.array([1]), 1)
    with pytest.raises(ValueError):
        prototypes([[0, 0]], 1)
