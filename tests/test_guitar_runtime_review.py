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


def _seed_cache(runtime):
    _, report = _analyze(runtime)
    marker = Path(report['evidence_cache'])
    target = marker.parent / json.loads(marker.read_text(encoding='utf-8'))['file']
    return marker, target, marker.read_bytes(), target.read_bytes()


def _force_cache_recompute(monkeypatch, old_target):
    original_sha = guitar._sha

    def stale_old_cache(path):
        if Path(path) == old_target:
            return 'stale-cache-hash'
        return original_sha(path)

    monkeypatch.setattr(guitar, '_sha', stale_old_cache)


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


@pytest.mark.parametrize('failure', ['npz_write', 'target_hash', 'marker_write', 'cancel', 'replace'])
def test_failed_cache_publication_cleans_only_new_files_and_keeps_old_cache(
        runtime, monkeypatch, failure):
    marker, old_target, old_marker_bytes, old_payload_bytes = _seed_cache(runtime)
    _force_cache_recompute(monkeypatch, old_target)

    if failure == 'npz_write':
        def partial_npz(stream, **kwargs):
            stream.write(b'partial archive')
            raise OSError('synthetic NPZ write failure')

        monkeypatch.setattr(guitar.np, 'savez_compressed', partial_npz)
        error = OSError
    elif failure == 'target_hash':
        sha = guitar._sha

        def broken_new_target(path):
            if Path(path).name.startswith('activations-') and Path(path) != old_target:
                raise OSError('synthetic target hash failure')
            return sha(path)

        monkeypatch.setattr(guitar, '_sha', broken_new_target)
        error = OSError
    elif failure == 'marker_write':
        open_path = Path.open

        class PartialWriter:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def write(self, value):
                self.stream.write(value[:8])
                raise OSError('synthetic marker write failure')

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

        def partial_marker(path, mode='r', *args, **kwargs):
            stream = open_path(path, mode, *args, **kwargs)
            if path.name.startswith('current-') and mode == 'x':
                return PartialWriter(stream)
            return stream

        monkeypatch.setattr(Path, 'open', partial_marker)
        error = OSError
    elif failure == 'cancel':
        polls = [0]

        def cancel_before_replace():
            polls[0] += 1
            return polls[0] >= 4

        with pytest.raises(InterruptedError):
            _analyze(runtime, cancelled=cancel_before_replace)
        error = None
    else:
        replace = Path.replace

        def failed_replace(path, destination):
            if path.name.startswith('current-') and Path(destination) == marker:
                raise OSError('synthetic marker replace failure')
            return replace(path, destination)

        monkeypatch.setattr(Path, 'replace', failed_replace)
        error = OSError

    if error is not None:
        with pytest.raises(error):
            _analyze(runtime)

    assert marker.read_bytes() == old_marker_bytes
    assert old_target.read_bytes() == old_payload_bytes
    assert list(marker.parent.glob('activations-*.npz')) == [old_target]
    assert not list(marker.parent.glob('current-*.tmp'))


def test_cleanup_continues_after_unlink_failure_and_reports_it(runtime, monkeypatch):
    marker, old_target, old_marker_bytes, old_payload_bytes = _seed_cache(runtime)
    _force_cache_recompute(monkeypatch, old_target)
    open_path = Path.open
    unlink = Path.unlink
    attempted = []

    class PartialWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def write(self, value):
            self.stream.write(value[:8])
            raise OSError('synthetic marker write failure')

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

    def partial_marker(path, mode='r', *args, **kwargs):
        stream = open_path(path, mode, *args, **kwargs)
        if path.name.startswith('current-') and mode == 'x':
            return PartialWriter(stream)
        return stream

    def failing_unlink(path, *args, **kwargs):
        if path.name.startswith(('activations-', 'current-')) and path != old_target:
            attempted.append(path)
            if path.name.startswith('activations-'):
                raise PermissionError('synthetic unlink failure')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', partial_marker)
    monkeypatch.setattr(Path, 'unlink', failing_unlink)
    with pytest.raises(OSError, match='synthetic marker write failure') as caught:
        _analyze(runtime)

    assert len(attempted) == 2
    assert attempted[0].name.startswith('current-')
    assert attempted[1].name.startswith('activations-')
    assert any('synthetic unlink failure' in note for note in caught.value.__notes__)
    assert marker.read_bytes() == old_marker_bytes
    assert old_target.read_bytes() == old_payload_bytes
    assert len(list(marker.parent.glob('activations-*.npz'))) == 2
    assert not list(marker.parent.glob('current-*.tmp'))


def test_uuid_collision_does_not_delete_an_existing_payload(runtime, monkeypatch):
    marker, old_target, old_marker_bytes, old_payload_bytes = _seed_cache(runtime)
    _force_cache_recompute(monkeypatch, old_target)
    collision_id = old_target.stem.removeprefix('activations-')
    monkeypatch.setattr(guitar.uuid, 'uuid4', lambda: SimpleNamespace(hex=collision_id))

    with pytest.raises(FileExistsError):
        _analyze(runtime)

    assert marker.read_bytes() == old_marker_bytes
    assert old_target.read_bytes() == old_payload_bytes
    assert list(marker.parent.glob('activations-*.npz')) == [old_target]
    assert not list(marker.parent.glob('current-*.tmp'))


def test_replace_that_publishes_then_raises_keeps_new_cache(runtime, monkeypatch):
    marker, old_target, _, _ = _seed_cache(runtime)
    _force_cache_recompute(monkeypatch, old_target)
    original_sha = guitar._sha
    original_replace = Path.replace

    def replace_then_raise(source, destination):
        result = original_replace(source, destination)
        if Path(destination) == marker:
            raise KeyboardInterrupt('synthetic interruption after marker replace')
        return result

    monkeypatch.setattr(Path, 'replace', replace_then_raise)
    with pytest.raises(KeyboardInterrupt, match='after marker replace'):
        _analyze(runtime)

    marker_data = json.loads(marker.read_text(encoding='utf-8'))
    target = marker.parent / marker_data['file']
    assert target.is_file()
    assert marker_data['sha256'] == original_sha(target)

    monkeypatch.setattr(guitar, '_sha', original_sha)
    assert _analyze(runtime)[1]['cached']


@pytest.mark.parametrize('cached', [False, True])
def test_cancellation_in_final_progress_cannot_report_success(runtime, cached):
    if cached:
        _, report = _analyze(runtime)
        marker = Path(report['evidence_cache'])
        before = marker.read_bytes()
    cancelled = [False]

    def progress(percent, stage):
        if percent == 100:
            cancelled[0] = True

    with pytest.raises(InterruptedError):
        _analyze(runtime, progress=progress, cancelled=lambda: cancelled[0])
    assert len(runtime.calls) == 1
    if cached:
        assert marker.read_bytes() == before
    else:
        marker = next(runtime.cache.glob('*/current.json'))
        assert _analyze(runtime)[1]['cached']
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
