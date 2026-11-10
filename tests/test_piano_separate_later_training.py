"""Exercise the runner's two coefficients together with the real base objective."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from torch.nn import functional as F
from scripts.adapt_piano_separate_later_ranking import combined_loss


@pytest.mark.parametrize(('known', 'derived'), [(True, True), (True, False), (False, True)])
def test_runner_keeps_base_and_weights_each_available_auxiliary_once(known, derived):
    voice = torch.tensor([.3, -.2, .7], dtype=torch.float64, requires_grad=True)
    staff = torch.tensor([[.2, .4]], dtype=torch.float64, requires_grad=True)
    mask = np.asarray([True, known, False])
    targets = dict(voice=torch.tensor([1., 0., 0.], dtype=torch.float64),
                   voice_mask=torch.as_tensor(mask), staff=torch.tensor([1]),
                   staff_mask=torch.tensor([True]), chord_mask=torch.tensor([False]))
    item = dict(payload=dict(event_ids=np.asarray(['s', 'p', 'old', 'added']),
                             potential_edges=np.asarray([[0, 0, 0], [1, 2, 3]]),
                             notes=np.asarray([(0,), (120,), (240,), (360,)], dtype=[('onset_div', 'i8')]),
                             voice=np.asarray([1., 0., 0.]), voice_mask=mask),
                derived_later_mask=np.asarray([False, False, derived]))
    total, base, old, new = combined_loss((voice, staff, None, None), targets, item)
    expected_base = F.binary_cross_entropy_with_logits(voice[mask], targets['voice'][mask],
                                                      pos_weight=voice.new_tensor(5.)) + F.cross_entropy(staff, targets['staff'])
    expected = expected_base
    if known:
        expected = expected + .1 * F.softplus(voice[1]-voice[0])
    if derived:
        expected = expected + .1 * F.softplus(voice[2]-voice[0])
    assert torch.equal(base, expected_base)
    assert torch.equal(total, expected)
    assert (old is not None) == known and (new is not None) == derived
    actual_gradient = torch.autograd.grad(total, (voice, staff), retain_graph=True)
    expected_gradient = torch.autograd.grad(expected, (voice, staff))
    assert all(torch.equal(a, b) for a, b in zip(actual_gradient, expected_gradient))


def test_runner_returns_no_loss_without_any_supervision():
    voice = torch.tensor([.5])
    targets = dict(voice=torch.tensor([0.]), voice_mask=torch.tensor([False]),
                   staff=torch.tensor([0]), staff_mask=torch.tensor([False]), chord_mask=torch.tensor([False]))
    item = dict(payload=dict(event_ids=np.asarray(['s', 't']), potential_edges=np.asarray([[0], [1]]),
                             notes=np.asarray([(0,), (120,)], dtype=[('onset_div', 'i8')]),
                             voice=np.asarray([0.]), voice_mask=np.asarray([False])),
                derived_later_mask=np.asarray([False]))
    assert combined_loss((voice, torch.zeros((1, 2)), None, None), targets, item) == (None, None, None, None)
