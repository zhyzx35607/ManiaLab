import hashlib
import os

from manialab import piano
from manialab.model_identity import VerifiedModel


def test_status_checks_identity_once_and_invalidates_on_change(tmp_path, monkeypatch):
    model = tmp_path / 'model.onnx'
    good = b'correct model'
    identity = VerifiedModel(model, len(good), hashlib.sha256(good).hexdigest())
    assert identity.state() == 'missing'
    model.write_bytes(good)
    digest = hashlib.file_digest
    calls = []
    def count(*args):
        calls.append(True)
        return digest(*args)
    monkeypatch.setattr(hashlib, 'file_digest', count)
    assert identity.state() == identity.state() == 'verified'
    assert len(calls) == 1
    before = model.stat().st_mtime_ns
    model.write_bytes(b'x' * len(good))
    os.utime(model, ns=(before + 1_000_000_000, before + 1_000_000_000))
    assert identity.state() == 'mismatched'
    assert len(calls) == 2
    model.unlink()
    assert identity.state() == 'missing'
    model.write_bytes(b'short')
    assert identity.state() == 'mismatched'
    assert len(calls) == 2


def test_forced_identity_check_detects_same_stat_in_place_change(tmp_path):
    path = tmp_path / 'model'
    path.write_bytes(b'first')
    identity = VerifiedModel(path, 5, hashlib.sha256(b'first').hexdigest())
    assert identity.state() == 'verified'
    before = path.stat().st_mtime_ns
    path.write_bytes(b'other')
    os.utime(path, ns=(before, before))
    assert identity.state(refresh=True) == 'mismatched'


def test_versioned_piano_runtime_preserves_historical_model(tmp_path, monkeypatch):
    good = b'current graph'
    sha = hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(piano, 'MODEL_SHA256', sha)
    monkeypatch.setattr(piano, 'MODEL_BYTES', len(good))
    folder = tmp_path / 'assets/deep'
    folder.mkdir(parents=True)
    old = folder / 'piano-regression.onnx'
    old.write_bytes(b'historical graph')
    assert piano.PianoDetector(tmp_path).status()['state'] == 'mismatched'
    current = folder / f'piano-regression-{sha}.onnx'
    current.write_bytes(good)
    detector = piano.PianoDetector(tmp_path)
    assert detector.status()['available']
    assert detector.model == current
    assert old.read_bytes() == b'historical graph'
