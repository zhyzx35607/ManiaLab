from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from manialab import audio, audio_io


def test_audio_decode_keeps_legacy_run_hidden_patch_point(tmp_path, monkeypatch):
    samples = np.linspace(-0.25, 0.25, audio.SR // 2, dtype=np.float32)
    calls = []

    def fake_runner(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=samples.astype('<f4').tobytes())

    monkeypatch.setattr(audio, 'run_hidden', fake_runner)
    decoded = audio.decode(Path('synthetic.wav'), tmp_path)

    assert np.array_equal(decoded, samples)
    assert len(calls) == 1
    assert calls[0][1] == {'capture_output': True, 'timeout': 180}
    assert calls[0][0][calls[0][0].index('-ar') + 1] == str(audio.SR)
    rate, preview = wavfile.read(tmp_path / 'preview.wav')
    assert rate == audio.SR and preview.dtype == np.int16


def test_audio_decode_cancellation_terminates_and_reaps_ffmpeg(tmp_path, monkeypatch):
    cancelled = iter((False, False, True))
    calls = []
    processes = []

    class Process:
        returncode = -9

        def __init__(self, args, **kwargs):
            calls.append((args, kwargs))
            self.alive = True
            self.communications = 0
            processes.append(self)

        def communicate(self, timeout=None):
            self.communications += 1
            if self.alive:
                raise subprocess.TimeoutExpired('ffmpeg', timeout)
            return b'', b''

        def poll(self):
            return None if self.alive else self.returncode

        def kill(self):
            self.alive = False

    monkeypatch.setattr(audio_io.subprocess, 'Popen', Process)
    with pytest.raises(InterruptedError, match='已取消'):
        audio_io.decode(Path('synthetic.wav'), tmp_path, cancelled=lambda: next(cancelled))

    assert len(calls) == 1
    assert calls[0][1]['stdin'] is subprocess.DEVNULL
    assert calls[0][1]['stdout'] is subprocess.PIPE
    assert calls[0][1]['stderr'] is subprocess.PIPE
    assert not processes[0].alive
    assert processes[0].communications == 2
    assert not (tmp_path / 'preview.wav').exists()


def test_audio_io_and_import_consumers_do_not_load_audio_runtime():
    code = (
        "import sys; "
        "import manialab.audio_io, manialab.provided_stems, manialab.separation, "
        "manialab.upstream_routes, manialab.arrangement_routes; "
        "assert 'manialab.audio' not in sys.modules; "
        "assert 'onnxruntime' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_audio_module_reexports_shared_process_helper():
    assert audio.run_hidden is audio_io.run_hidden
