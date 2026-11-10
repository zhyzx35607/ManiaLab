"""Runtime/cache boundary review; no neural runtime or model execution."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from manialab import guitar


def _heads(frames):
    return {name: np.zeros((frames, 88), dtype=np.float32) for name in guitar.OUTPUTS}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    detector = guitar.GuitarDetector(tmp_path)
    detector.model.parent.mkdir(parents=True)
    detector.model.write_bytes(b'review-only-model-placeholder')
    monkeypatch.setattr(guitar, 'MODEL_SHA256', hashlib.sha256(detector.model.read_bytes()).hexdigest())
    sessions, calls = [], []

    def session(*args, **kwargs):
        sessions.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=session))
    versions = {name: 'review-version-1' for name in ('onnxruntime', 'numpy', 'scipy')}
    monkeypatch.setattr(guitar.importlib.metadata, 'version', lambda name: versions[name])

    def infer(pcm, rate, *args):
        calls.append((pcm.copy(), rate))
        return _heads((len(pcm) * 100 + rate - 1) // rate)

    monkeypatch.setattr(guitar, 'infer', infer)
    return SimpleNamespace(detector=detector, pcm=np.zeros(16000, dtype=np.float32),
        cache=tmp_path / 'cache', versions=versions, calls=calls, sessions=sessions)


def _analyze(runtime, **kwargs):
    return runtime.detector.analyze(runtime.pcm, kwargs.pop('sample_rate', 16000),
        runtime.cache, 'review-stem', **kwargs)


@pytest.mark.parametrize('change', ['sample_rate', 'threshold', 'onnxruntime', 'numpy',
                                   'scipy', 'detector', 'checkpoint'])
def test_cache_identity_includes_every_execution_dependency(runtime, monkeypatch, change):
    _, original = _analyze(runtime)
    kwargs = {}
    if change == 'sample_rate':
        kwargs['sample_rate'] = 22050
    elif change == 'threshold':
        monkeypatch.setattr(guitar, 'THRESHOLDS', dict(guitar.THRESHOLDS, onset_threshold=.31))
    elif change in runtime.versions:
        runtime.versions[change] = 'review-version-2'
    elif change == 'detector':
        old_sha = guitar._sha
        monkeypatch.setattr(guitar, '_sha', lambda path:
            'e' * 64 if Path(path) == Path(guitar.__file__) else old_sha(path))
    else:
        monkeypatch.setattr(guitar, 'CHECKPOINT_SHA256', 'f' * 64)
    _, changed = _analyze(runtime, **kwargs)
    assert not changed['cached'] and len(runtime.calls) == 2
    assert original['evidence_cache'] != changed['evidence_cache']
    assert Path(original['evidence_cache']).is_file()
    assert len(list(runtime.cache.glob('*/current.json'))) == 2


@pytest.mark.parametrize('damage', ['json', 'identity', 'missing_file', 'file_path',
    'inner_identity', 'missing_head', 'shape', 'nan', 'negative', 'above_one'])
def test_structurally_corrupted_cache_recomputes_without_reusing_bad_heads(runtime, damage):
    _, initial = _analyze(runtime)
    marker = Path(initial['evidence_cache'])
    saved = json.loads(marker.read_text(encoding='utf-8'))
    target = marker.parent / saved['file']
    if damage == 'json':
        marker.write_text('{incomplete', encoding='utf-8')
    elif damage == 'identity':
        saved['identity']['normalization'] = True
        marker.write_text(json.dumps(saved), encoding='utf-8')
    elif damage == 'missing_file':
        target.unlink()
    elif damage == 'file_path':
        saved['file'] = 'activations-../outside.npz'
        marker.write_text(json.dumps(saved), encoding='utf-8')
    else:
        with np.load(target, allow_pickle=False) as data:
            values = {name: data[name] for name in data.files}
        first = guitar.OUTPUTS[0]
        if damage == 'inner_identity':
            values['identity'] = 'different-signature'
        elif damage == 'missing_head':
            del values[first]
        elif damage == 'shape':
            values[first] = values[first][:-1]
        else:
            values[first][0, 0] = {'nan': np.nan, 'negative': -.1, 'above_one': 1.1}[damage]
        with target.open('wb') as stream:
            np.savez_compressed(stream, **values)
        # Keep the container hash coherent to exercise semantic validation too.
        saved['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        marker.write_text(json.dumps(saved), encoding='utf-8')
    _, repaired = _analyze(runtime)
    assert not repaired['cached'] and repaired['cache_rejection']
    assert len(runtime.calls) == 2
    assert json.loads(marker.read_text(encoding='utf-8'))['file'] != saved['file']
    _, reused = _analyze(runtime)
    assert reused['cached'] and len(runtime.calls) == 2


def test_hot_cache_never_initializes_a_session_and_retains_cpu_configuration(runtime):
    _analyze(runtime)
    _analyze(runtime)
    assert len(runtime.sessions) == 1
    options = runtime.sessions[0]['sess_options']
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (2, 1)
    assert runtime.sessions[0]['providers'] == ['CPUExecutionProvider']


@pytest.mark.parametrize('cached', [False, True])
def test_cancellation_during_decode_cannot_return_a_completed_result(runtime, monkeypatch, cached):
    if cached:
        _, report = _analyze(runtime)
        marker = Path(report['evidence_cache'])
        before = marker.read_bytes()
    cancelled = [False]
    decode = guitar.decode

    def cancelling_decode(*args, **kwargs):
        result = decode(*args, **kwargs)
        cancelled[0] = True
        return result

    monkeypatch.setattr(guitar, 'decode', cancelling_decode)
    with pytest.raises(InterruptedError):
        _analyze(runtime, cancelled=lambda: cancelled[0])
    assert len(runtime.calls) == 1
    if cached:
        assert marker.read_bytes() == before
    else:
        assert not list(runtime.cache.glob('*/current.json'))


def test_inference_failure_cannot_publish_a_current_cache(runtime, monkeypatch):
    def broken(*args):
        raise RuntimeError('synthetic inference failure')

    monkeypatch.setattr(guitar, 'infer', broken)
    with pytest.raises(RuntimeError, match='synthetic inference failure'):
        _analyze(runtime)
    assert not list(runtime.cache.glob('*/current.json'))


@pytest.mark.parametrize('cached', [False, True])
def test_cancellation_in_final_progress_cannot_report_success(runtime, cached):
    if cached:
        _analyze(runtime)
    cancelled = [False]

    def progress(percent, stage):
        if percent == 100:
            cancelled[0] = True

    with pytest.raises(InterruptedError):
        _analyze(runtime, progress=progress, cancelled=lambda: cancelled[0])
    assert len(runtime.calls) == 1


def test_zero_velocity_event_still_has_unit_weight_and_exact_raw_trace():
    raw = dict(onset=.1006, offset=.8004, pitch=40, velocity=0, onset_score=.30001,
               release_reason='release_regression', release_observed=True)
    for allowed in (False, True):
        notes, trace = guitar.candidates([raw], allow_predicted_releases=allowed)
        assert (notes[0].start, notes[0].end, notes[0].strength) == (101, 800, 1.)
        assert notes[0].voice == 'guitar' and notes[0].id == 'guitar:fl:0'
        assert trace[0]['raw_event'] == raw
        assert trace[0]['model_predicted_release_hypothesis'] is allowed
        assert trace[0]['authored_ln_verified'] is False
        assert trace[0]['weight_semantics'] == 'unit_weight_not_confidence'
