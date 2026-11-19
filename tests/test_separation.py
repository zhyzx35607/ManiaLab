import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from manialab import separation


def test_separator_cleans_temporary_input_when_worker_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setenv('MANIALAB_DEEP_PYTHON', sys.executable)
    separator = separation.Separator(tmp_path)
    separator.model.parent.mkdir(parents=True)
    separator.model.write_bytes(b"synthetic model")
    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic audio")
    wave = tmp_path / "separation-input.wav"

    def prepare_input(*args, **kwargs):
        wave.write_bytes(b"synthetic converted audio")
        return SimpleNamespace(returncode=0)

    def fail_to_start(*args, **kwargs):
        raise OSError("worker cannot start")

    monkeypatch.setattr(separation, "run_hidden_cancellable", prepare_input)
    monkeypatch.setattr(separation.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(separation.subprocess, "Popen", fail_to_start)

    with pytest.raises(OSError, match="worker cannot start"):
        separator.separate(source, tmp_path)

    assert not wave.exists()


@pytest.mark.parametrize('profile', ['four', 'six'])
def test_separator_preserves_float_input_and_rebuilds_legacy_cache(tmp_path, monkeypatch, profile):
    monkeypatch.setenv('MANIALAB_DEEP_PYTHON', sys.executable)
    monkeypatch.delenv('MANIALAB_DEEP_MODEL', raising=False)
    separator = separation.Separator(tmp_path, profile)
    separator.model.parent.mkdir(parents=True)
    separator.model.touch()
    samples = np.zeros((44100, 2), dtype=np.float32)
    samples[4410:4414] = [[1.5, -1.25], [.00001, -.00001], [.25, -.5], [.9, -.9]]
    source = tmp_path/'source.wav'
    wavfile.write(source, 44100, samples)
    output = tmp_path/separator.spec['directory']
    output.mkdir()
    manifest = output/'separation.json'
    old_report = dict(audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      model_sha256=separator.spec['sha256'])
    manifest.write_text(json.dumps(old_report), encoding='utf-8')
    for voice in separator.spec['voices']:
        np.save(output/f'{voice}.npy', np.zeros(22050, dtype=np.float32))

    def reject_read_bytes(path):
        raise AssertionError(f'whole-file read attempted for {path}')

    monkeypatch.setattr(
        Path,
        'read_bytes',
        reject_read_bytes,
    )
    launches = []
    real_popen = subprocess.Popen

    class FinishedWorker:
        returncode = 0

        def poll(self):
            return self.returncode

    def launch(command, **kwargs):
        if command[0] != sys.executable:
            return real_popen(command, **kwargs)  # Run the real ffmpeg input conversion.
        assert command[command.index("--threads") + 1] == "2"
        assert all(kwargs["env"][name] == "2" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        rate, data = wavfile.read(command[2])
        assert rate == 44100 and data.dtype == np.float32
        np.testing.assert_array_equal(data, samples)
        launches.append(command)
        report = json.loads(manifest.read_text(encoding='utf-8'))
        if '--beat-model' in command:
            report['beat_evidence'] = dict(model_sha256=separation.BEAT_MODEL_SHA256)
        manifest.write_text(json.dumps(report), encoding='utf-8')
        return FinishedWorker()

    monkeypatch.setattr(separation.subprocess, 'Popen', launch)
    _, report = separator.separate(source, tmp_path)
    assert len(launches) == 1 and not report['cached']
    assert report['input_processing'] == separation.INPUT_PROCESSING
    assert not (tmp_path/'separation-input.wav').exists()
    _, report = separator.separate(source, tmp_path)
    assert len(launches) == 1 and report['cached']
    # Adding beat evidence can reuse the new stems, but must keep their identity.
    separator.beat_model.touch()
    _, report = separator.separate(source, tmp_path)
    assert len(launches) == 2 and '--beats-only' in launches[-1]
    assert report['cached'] and report['input_processing'] == separation.INPUT_PROCESSING
    _, report = separator.separate(source, tmp_path)
    assert len(launches) == 2 and report['cached']
    # Another processing revision cannot reuse stems merely because weights match.
    monkeypatch.setattr(separation, 'INPUT_PROCESSING', separation.INPUT_PROCESSING | {'revision': 3})
    _, report = separator.separate(source, tmp_path)
    assert len(launches) == 3 and '--beats-only' not in launches[-1]
    assert not report['cached']
