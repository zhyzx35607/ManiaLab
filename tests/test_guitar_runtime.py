import hashlib
import json

import numpy as np
import pytest

from manialab import guitar
from manialab.domain import UNOBSERVED_RELEASES
from scripts.research_hf_instrument import stitch_windows, physical_heads


def heads(frames):
    return {key: np.zeros((frames, 88), dtype=np.float32) for key in guitar.OUTPUTS}


def event(**updates):
    result = dict(onset=1., offset=2., pitch=60, velocity=100, onset_score=.31,
                  release_reason='frame_decay', release_observed=True)
    result.update(updates)
    return result


@pytest.mark.parametrize('samples', [160000, 160001, 320000, 357191])
def test_guitar_overlap_uses_original_integer_clock_and_preserves_full_gain(samples):
    observed, windows = [], {key: [] for key in guitar.OUTPUTS}
    class Clock:
        def run(self, names, data):
            chunk = data['audio']
            observed.append(chunk.copy())
            start = (len(observed) - 1) * 500
            output = np.broadcast_to(((start + np.arange(1001)) / 4000)[None, :, None],
                                      (1, 1001, 88)).astype(np.float32).copy()
            for key in names:
                windows[key].append(output[0])
            return [output for key in names]
    pcm = np.full(samples, 1.3, dtype=np.float32)
    result = guitar.infer(pcm, 16000, Clock())
    assert observed[0][0, 0] == np.float32(1.3)
    original = {key: np.stack(value) for key, value in windows.items()}
    expected, _, mask = physical_heads(stitch_windows(original), samples, 16000)
    for key in guitar.OUTPUTS:
        np.testing.assert_array_equal(result[key], expected[key])
        assert len(result[key]) == int(mask.sum())


def test_guitar_does_not_normalize_or_quantize_float_during_resampling():
    from scipy.signal import resample_poly
    seen = []
    class Capture:
        def run(self, names, inputs):
            seen.append(inputs['audio'].copy())
            return [np.zeros((1, 1001, 88), dtype=np.float32) for key in names]
    pcm = np.array([1.35, -.17, .00417] * 1000, dtype=np.float32)
    result = guitar.infer(pcm, 22050, Capture())
    expected = resample_poly(pcm, 320, 441).astype(np.float32)
    np.testing.assert_array_equal(seen[0][0, :len(expected)], expected)
    assert len(result[guitar.OUTPUTS[0]]) == (len(pcm) * 100 + 22049) // 22050


@pytest.mark.parametrize('reason', ['frame_decay', 'release_regression', 'reattack',
                                  'input_boundary', 'duration_limit', 'frame_inactive'])
@pytest.mark.parametrize('allowed', [False, True])
def test_tail_mode_keeps_every_attack_and_raw_interval_without_promoting_unknowns(reason, allowed):
    observed = reason in guitar.OBSERVED
    raw = event(release_reason=reason, release_observed=observed)
    notes, trace = guitar.candidates([raw], allow_predicted_releases=allowed)
    assert len(notes) == 1 and (notes[0].start, notes[0].end, notes[0].pitch) == (1000, 2000, 60)
    assert trace[0]['raw_event'] == raw and notes[0].strength == 1.
    assert trace[0]['weight_semantics'] == 'unit_weight_not_confidence'
    if observed and allowed:
        assert notes[0].release_reason == 'model_predicted_offset'
    else:
        assert notes[0].release_reason in UNOBSERVED_RELEASES
    assert trace[0]['authored_ln_verified'] is False


def test_unsupported_tail_cannot_be_marked_observed():
    with pytest.raises(ValueError, match='音尾来源'):
        guitar.candidates([event(release_reason='reattack', release_observed=True)])


def setup_runtime(tmp_path, monkeypatch):
    import onnxruntime as ort
    detector = guitar.GuitarDetector(tmp_path)
    detector.model.parent.mkdir(parents=True)
    detector.model.write_bytes(b'model')
    monkeypatch.setattr(guitar, 'MODEL_SHA256', hashlib.sha256(b'model').hexdigest())
    monkeypatch.setattr(ort, 'InferenceSession', lambda *a, **k: object())
    calls = []
    def predict(pcm, rate, *args):
        calls.append(pcm.copy())
        return heads((len(pcm) * 100 + rate - 1) // rate)
    monkeypatch.setattr(guitar, 'infer', predict)
    return detector, calls


def test_cache_reuses_unchanged_pcm_but_separates_policy_source_and_decoder(tmp_path, monkeypatch):
    detector, calls = setup_runtime(tmp_path, monkeypatch)
    pcm = np.full(22050, 1.25, dtype=np.float32)
    cache = tmp_path / 'guitar-cache'
    original = tmp_path / 'guitar-activations.npz'
    original.write_bytes(b'old basic pitch')
    _, first = detector.analyze(pcm, 22050, cache, 'same')
    _, second = detector.analyze(pcm, 22050, cache, 'same')
    assert not first['cached'] and second['cached'] and len(calls) == 1
    np.testing.assert_array_equal(calls[0], pcm)
    detector.analyze(pcm, 22050, cache, 'same', allow_predicted_releases=True)
    detector.analyze(pcm, 22050, cache, 'new-source')
    pcm[0] = .062341
    detector.analyze(pcm, 22050, cache, 'same')
    assert len(calls) == 4
    assert original.read_bytes() == b'old basic pitch'
    old_sha = guitar._sha
    def changed_decoder(path):
        return 'f' * 64 if str(path).endswith('piano.py') else old_sha(path)
    monkeypatch.setattr(guitar, '_sha', changed_decoder)
    detector.analyze(pcm, 22050, cache, 'same')
    assert len(calls) == 5


def test_corrupted_cache_is_recomputed_and_cancellation_does_not_replace_marker(tmp_path, monkeypatch):
    detector, calls = setup_runtime(tmp_path, monkeypatch)
    pcm = np.zeros(16000, dtype=np.float32)
    cache = tmp_path / 'cache'
    _, report = detector.analyze(pcm, 16000, cache, 'same')
    marker = __import__('pathlib').Path(report['evidence_cache'])
    saved = json.loads(marker.read_text())
    target = marker.parent / saved['file']
    target.write_bytes(b'corrupt')
    _, rebuilt = detector.analyze(pcm, 16000, cache, 'same')
    assert len(calls) == 2 and not rebuilt['cached'] and rebuilt['cache_rejection']
    previous = marker.read_bytes()
    target = marker.parent / json.loads(previous)['file']
    target.write_bytes(b'corrupt again')
    cancelled = [False]
    def cancelling(*args):
        cancelled[0] = True
        return heads(100)
    monkeypatch.setattr(guitar, 'infer', cancelling)
    with pytest.raises(InterruptedError):
        detector.analyze(pcm, 16000, cache, 'same', cancelled=lambda: cancelled[0])
    assert marker.read_bytes() == previous


def test_model_and_invalid_input_fail_before_inference(tmp_path, monkeypatch):
    detector, calls = setup_runtime(tmp_path, monkeypatch)
    cache = tmp_path / 'cache'
    for pcm, rate in ((np.zeros(5), 16000), (np.zeros((16000, 2)), 16000),
                      (np.full(16000, np.nan), 16000), (np.zeros(16000), True)):
        with pytest.raises(ValueError):
            detector.analyze(pcm, rate, cache, 'same')
    detector.model.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='模型文件'):
        detector.analyze(np.zeros(16000), 16000, cache, 'same')
    assert calls == []
