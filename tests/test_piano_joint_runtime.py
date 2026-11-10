from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from scripts.piano_joint_runtime import OUTPUTS, forward_logits, heads_copy, replay_outputs


class ToyModel(torch.nn.Module):
    """A tiny GRU with the real window length and 88-pitch conditioning contract."""
    def __init__(self):
        super().__init__()
        self.note_model = torch.nn.Module()
        self.note_model.reg_onset_fc = torch.nn.Linear(5, 88)
        self.note_model.frame_gru = torch.nn.GRU(264, 3, batch_first=True, bidirectional=True)
        self.note_model.frame_fc = torch.nn.Linear(6, 88)


@pytest.fixture
def sample():
    old_rng = torch.get_rng_state()
    old_threads = torch.get_num_threads()
    torch.manual_seed(19)
    torch.set_num_threads(2)
    try:
        model = ToyModel().eval().requires_grad_(False)
        inputs = (torch.randn(2, 1001, 5), torch.rand(2, 1001, 88), torch.rand(2, 1001, 88))
        yield model, heads_copy(model), inputs
    finally:
        torch.set_rng_state(old_rng)
        torch.set_num_threads(old_threads)


def snapshot(model):
    return {name: value.clone() for name, value in model.state_dict().items()}


def assert_unchanged(model, before):
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_heads_are_independent_trainable_copies_and_eval_replays_original(sample):
    model, heads, (h, f0, offset) = sample
    before = snapshot(model)
    assert isinstance(heads, torch.nn.ModuleDict) and set(heads) == {'onset', 'frame'}
    assert all(parameter.requires_grad for parameter in heads.parameters())
    for name, original in (('onset', model.note_model.reg_onset_fc),
                           ('frame', model.note_model.frame_fc)):
        for copied, frozen in zip(heads[name].parameters(), original.parameters()):
            assert torch.equal(copied, frozen) and copied.data_ptr() != frozen.data_ptr()
    onset, frame = forward_logits(model, heads, h, f0, offset)
    with torch.no_grad():
        original_onset = model.note_model.reg_onset_fc(h)
        values, _ = model.note_model.frame_gru(torch.cat((f0, original_onset.sigmoid(), offset), dim=2))
        original_frame = model.note_model.frame_fc(values)
    assert torch.equal(onset, original_onset) and torch.equal(frame, original_frame)
    assert_unchanged(model, before)


@pytest.mark.parametrize('loss_head', ['onset', 'frame'])
def test_each_loss_updates_only_its_head_and_cached_inputs_stay_frozen(sample, loss_head):
    model, heads, inputs = sample
    inputs = tuple(value.detach().requires_grad_(True) for value in inputs)
    before = snapshot(model)
    onset, frame = forward_logits(model, heads, *inputs)
    logits = onset if loss_head == 'onset' else frame
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
    for name, head in heads.items():
        for parameter in head.parameters():
            if name == loss_head:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert torch.count_nonzero(parameter.grad) > 0
            else:
                assert parameter.grad is None
    assert all(value.grad is None for value in inputs)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert_unchanged(model, before)


def test_joint_step_changes_both_heads_and_no_original_model_state(sample):
    model, heads, inputs = sample
    before, previous_heads = snapshot(model), snapshot(heads)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-5, weight_decay=1e-4)
    onset, frame = forward_logits(model, heads, *inputs)
    (onset.square().mean() + frame.square().mean()).backward()
    torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.)
    optimizer.step()
    assert all(not torch.equal(value, previous_heads[name]) for name, value in heads.state_dict().items())
    assert_unchanged(model, before)


def test_frame_features_are_recomputed_for_current_onset_with_full_context(sample):
    model, heads, inputs = sample
    seen = []
    def capture(module, args, output):
        seen.append((args[0].detach().clone(), output[0].detach().clone(), torch.is_grad_enabled()))
    handle = model.note_model.frame_gru.register_forward_hook(capture)
    try:
        onset0, frame0 = forward_logits(model, heads, *inputs)
        with torch.no_grad():
            heads['onset'].bias.add_(2)
        onset1, frame1 = forward_logits(model, heads, *inputs)
        onset2, frame2 = forward_logits(model, heads, *inputs)
    finally:
        handle.remove()
    assert len(seen) == 3 and all(row[0].shape == (2, 1001, 264) for row in seen)
    assert all(not row[2] for row in seen)
    assert torch.equal(seen[0][0][:, :, 88:176], onset0.detach().sigmoid())
    assert torch.equal(seen[1][0][:, :, 88:176], onset1.detach().sigmoid())
    assert not torch.equal(seen[0][1], seen[1][1])
    assert not torch.equal(frame0, frame1)
    assert torch.equal(onset1, onset2) and torch.equal(frame1, frame2)


def test_replay_copies_seven_outputs_and_does_not_modify_cached_evidence(sample):
    model, heads, (h, f0, offset) = sample
    cached = {key: np.full((2, 1001, 1 if 'pedal' in key else 88), index / 10, np.float32)
              for index, key in enumerate(OUTPUTS)}
    cached.update(features=h.numpy(), frame_base=f0.numpy(), reg_offset_output=offset.numpy())
    before = deepcopy(cached)
    expected_onset, expected_frame = forward_logits(model, heads, h, f0, offset)
    result = replay_outputs(model, heads, cached)
    assert set(result) == set(OUTPUTS)
    np.testing.assert_array_equal(result['reg_onset_output'], expected_onset.detach().sigmoid().numpy())
    np.testing.assert_array_equal(result['frame_output'], expected_frame.detach().sigmoid().numpy())
    for key, value in result.items():
        assert not np.shares_memory(value, cached[key])
        if key not in ('reg_onset_output', 'frame_output'):
            np.testing.assert_array_equal(value, cached[key])
    for key in cached:
        np.testing.assert_array_equal(cached[key], before[key])


@pytest.mark.parametrize('failure', ['short', 'wrong_pitch', 'nonfinite', 'dtype', 'training', 'unfrozen'])
def test_invalid_windows_or_mutable_original_model_are_rejected(sample, failure):
    model, heads, inputs = sample
    h, f0, offset = inputs
    if failure == 'short':
        h, f0, offset = h[:, 250:750], f0[:, 250:750], offset[:, 250:750]
    elif failure == 'wrong_pitch':
        f0 = f0[:, :, :87]
    elif failure == 'nonfinite':
        h[0, 0, 0] = float('nan')
    elif failure == 'dtype':
        h = h.double()
    elif failure == 'training':
        model.note_model.frame_gru.train()
    else:
        model.note_model.frame_gru.requires_grad_(True)
    with pytest.raises(ValueError):
        forward_logits(model, heads, h, f0, offset)
