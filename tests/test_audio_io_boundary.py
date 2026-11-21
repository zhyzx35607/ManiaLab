import os
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


@pytest.mark.parametrize('existing_preview', [False, True])
def test_audio_decode_does_not_publish_partial_preview_on_write_failure(
        tmp_path, monkeypatch, existing_preview):
    samples = np.zeros(audio.SR // 2, dtype=np.float32)
    preview_path = tmp_path / 'preview.wav'
    original_preview = b'existing-preview'
    if existing_preview:
        preview_path.write_bytes(original_preview)

    def fake_runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=samples.astype('<f4').tobytes())

    def partial_write(path, rate, data):
        Path(path).write_bytes(b'partial-wav')
        raise OSError('synthetic disk-full during preview write')

    monkeypatch.setattr(audio_io, 'run_hidden', fake_runner)
    monkeypatch.setattr(audio_io.wavfile, 'write', partial_write)

    with pytest.raises(OSError, match='synthetic disk-full'):
        audio_io.decode(Path('synthetic.wav'), tmp_path)

    assert preview_path.exists() is existing_preview
    if existing_preview:
        assert preview_path.read_bytes() == original_preview
    assert not list(tmp_path.glob('.preview-*.wav'))


@pytest.mark.skipif(os.name != 'nt', reason='Windows readers can block preview replacement')
@pytest.mark.parametrize('release_reader', [True, False])
def test_audio_decode_retries_windows_preview_reader_without_overwriting_it(
        tmp_path, monkeypatch, release_reader):
    samples = np.full(audio.SR // 2, .25, dtype='<f4')
    preview_path = tmp_path / 'preview.wav'
    wavfile.write(preview_path, audio.SR, np.zeros(len(samples), dtype=np.int16))
    original_preview = preview_path.read_bytes()
    delays = []

    def fake_runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=samples.tobytes())

    with preview_path.open('rb') as reader:
        def sleep(delay):
            delays.append(delay)
            reader.seek(0)
            assert reader.read() == original_preview
            if release_reader:
                reader.close()

        monkeypatch.setattr(audio_io.time, 'sleep', sleep)
        if release_reader:
            assert np.array_equal(audio_io.decode(Path('synthetic.wav'), tmp_path, runner=fake_runner),
                                  samples)
            assert len(delays) == 1
            rate, preview = wavfile.read(preview_path)
            assert rate == audio.SR
            assert np.array_equal(preview, (samples * 32767).astype(np.int16))
        else:
            with pytest.raises(PermissionError) as error:
                audio_io.decode(Path('synthetic.wav'), tmp_path, runner=fake_runner)
            assert error.value.winerror in (5, 32)
            assert delays and sum(delays) <= .5
            assert preview_path.read_bytes() == original_preview
    assert not list(tmp_path.glob('.preview-*.wav'))


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
            assert timeout is not None
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


def test_ffmpeg_cleanup_is_bounded_when_kill_does_not_stop_process(monkeypatch, caplog):
    cancelled = iter((False, False, True))
    processes = []
    clock = [0.0]
    monkeypatch.setattr(audio_io.time, 'monotonic', lambda: clock[0])

    class Pipe:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            self.closed = True

    class ActiveReader:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    class StuckProcess:
        returncode = None

        def __init__(self, args, **kwargs):
            self.alive = True
            self.killed = False
            self.timeouts = []
            self.wait_timeouts = []
            self.stdin = None
            self.stdout = Pipe()
            self.stderr = Pipe()
            self.stdout_thread = ActiveReader()
            self.stderr_thread = ActiveReader()
            processes.append(self)

        def communicate(self, timeout=None):
            self.timeouts.append(timeout)
            clock[0] += timeout
            raise subprocess.TimeoutExpired('ffmpeg', timeout)

        def poll(self):
            return None if self.alive else self.returncode

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            assert not self.stdout.closed and not self.stderr.closed
            clock[0] += timeout
            raise subprocess.TimeoutExpired('ffmpeg', timeout)

    monkeypatch.setattr(audio_io.subprocess, 'Popen', StuckProcess)
    with pytest.raises(ChildProcessError, match='cleanup deadline') as error:
        audio_io.run_hidden_cancellable([], lambda: next(cancelled), timeout=10)

    assert isinstance(error.value.__cause__, InterruptedError)
    assert processes[0].killed
    assert processes[0].timeouts == pytest.approx([.2, 4.7])
    assert processes[0].wait_timeouts == pytest.approx([.2])
    assert processes[0].timeouts[1] + processes[0].wait_timeouts[0] <= 5
    assert processes[0].stdout.close_calls == processes[0].stderr.close_calls == 0
    assert not processes[0].stdout.closed and not processes[0].stderr.closed
    assert "did not exit before its cleanup deadline" in caplog.text


def test_ffmpeg_timeout_cleanup_waits_for_readers_before_closing_pipes(monkeypatch):
    from manialab.cancellation import CancellationContext

    context = CancellationContext()
    cancelled_calls = []
    clock = [0.0]
    monkeypatch.setattr(audio_io.time, 'monotonic', lambda: clock[0])

    def cancelled():
        cancelled_calls.append(True)
        return len(cancelled_calls) >= 3

    cancelled.deadline = None
    cancelled.report_process_cleanup_failure = context.report_process_cleanup_failure

    class Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Reader:
        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

    class ExitsAfterPipeClose:
        def __init__(self, args, **kwargs):
            self.alive = True
            self.returncode = None
            self.communications = 0
            self.kill_calls = 0
            self.wait_timeouts = []
            self.stdin = None
            self.stdout = Pipe()
            self.stderr = Pipe()
            self.stdout_thread = Reader()
            self.stderr_thread = Reader()

        def communicate(self, timeout=None):
            self.communications += 1
            clock[0] += timeout
            raise subprocess.TimeoutExpired('ffmpeg', timeout)

        def poll(self):
            return None if self.alive else self.returncode

        def kill(self):
            self.kill_calls += 1
            raise PermissionError('synthetic access denied')

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            assert not self.stdout.closed and not self.stderr.closed
            self.alive = False
            self.returncode = 0
            self.stdout_thread.alive = False
            self.stderr_thread.alive = False
            return self.returncode

    processes = []

    def create_process(*args, **kwargs):
        process = ExitsAfterPipeClose(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(audio_io.subprocess, 'Popen', create_process)
    with pytest.raises(InterruptedError, match='已取消'):
        audio_io.run_hidden_cancellable([], cancelled, timeout=10)

    process = processes[0]
    assert process.communications == 2
    assert process.kill_calls == 1
    assert process.wait_timeouts == pytest.approx([.2])
    assert process.stdout.closed and process.stderr.closed
    assert not context.process_cleanup_failed
    assert not context.cleanup_failures


@pytest.mark.parametrize('reader_finishes', [True, False])
def test_ffmpeg_timeout_cleanup_joins_slow_readers_after_process_exit(
        monkeypatch, reader_finishes):
    from manialab.cancellation import CancellationContext

    context = CancellationContext()
    cancelled_calls = []
    clock = [0.0]
    monkeypatch.setattr(audio_io.time, 'monotonic', lambda: clock[0])

    def cancelled():
        cancelled_calls.append(True)
        return len(cancelled_calls) >= 3

    cancelled.deadline = None
    cancelled.report_process_cleanup_failure = context.report_process_cleanup_failure

    class Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class SlowReader:
        def __init__(self):
            self.alive = True
            self.join_timeouts = []

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)
            if reader_finishes:
                elapsed = min(.04, timeout)
                self.alive = False
            else:
                elapsed = timeout
            clock[0] += elapsed
            joined_duration[0] += elapsed

    class ExitedWithSlowReaders:
        def __init__(self, args, **kwargs):
            self.alive = True
            self.returncode = None
            self.communications = 0
            self.kill_calls = 0
            self.wait_timeouts = []
            self.stdin = None
            self.stdout = Pipe()
            self.stderr = Pipe()
            self.stdout_thread = SlowReader()
            self.stderr_thread = SlowReader()

        def communicate(self, timeout=None):
            self.communications += 1
            clock[0] += timeout
            if self.communications == 2:
                self.alive = False
                self.returncode = 0
            raise subprocess.TimeoutExpired('ffmpeg', timeout)

        def poll(self):
            return None if self.alive else self.returncode

        def kill(self):
            self.kill_calls += 1

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            raise AssertionError('an exited child should not be waited on')

    processes = []
    joined_duration = [0.0]

    def create_process(*args, **kwargs):
        process = ExitedWithSlowReaders(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(audio_io.subprocess, 'Popen', create_process)
    expected_error = InterruptedError if reader_finishes else ChildProcessError
    with pytest.raises(expected_error, match='已取消' if reader_finishes else 'pipes remain open') as error:
        audio_io.run_hidden_cancellable([], cancelled, timeout=10)

    process = processes[0]
    assert process.communications == 2
    assert process.kill_calls == 1
    assert process.wait_timeouts == []
    assert process.stdout_thread.join_timeouts
    assert process.stderr_thread.join_timeouts
    assert joined_duration[0] <= .1
    if reader_finishes:
        assert process.stdout.closed and process.stderr.closed
        assert not context.process_cleanup_failed
        assert not context.cleanup_failures
    else:
        assert isinstance(error.value.__cause__, InterruptedError)
        assert not process.stdout.closed and not process.stderr.closed
        assert context.process_cleanup_failed
        assert context.cleanup_failures == (error.value.args[0],)


def test_ffmpeg_kill_failure_is_recorded_and_keeps_cancellation_cause(monkeypatch):
    from manialab.cancellation import CancellationContext

    context = CancellationContext()
    calls = []

    def cancelled():
        calls.append(True)
        return len(calls) >= 3

    cancelled.deadline = None
    cancelled.report_process_cleanup_failure = context.report_process_cleanup_failure

    class UnkillableProcess:
        returncode = None

        def __init__(self, args, **kwargs):
            pass

        def poll(self):
            return None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired("ffmpeg", timeout)

        def kill(self):
            raise PermissionError("synthetic access denied")

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("ffmpeg", timeout)

    monkeypatch.setattr(audio_io.subprocess, "Popen", UnkillableProcess)
    with pytest.raises(ChildProcessError, match="could not be killed") as error:
        audio_io.run_hidden_cancellable([], cancelled, timeout=10)

    assert isinstance(error.value.__cause__, InterruptedError)
    assert context.cleanup_failures == (
        "FFmpeg process could not be killed during cleanup",
    )
    assert context.process_cleanup_failed


def test_ffmpeg_kill_failure_is_cleared_when_communicate_later_reaps_process(monkeypatch):
    from manialab.cancellation import CancellationContext

    context = CancellationContext()
    cancelled = iter((False, False, True))
    processes = []

    def is_cancelled():
        return next(cancelled)

    is_cancelled.report_process_cleanup_failure = context.report_process_cleanup_failure

    class ExitsDuringCleanup:
        returncode = 0

        def __init__(self, args, **kwargs):
            self.alive = True
            self.communications = 0
            self.poll_calls = 0
            self.kill_calls = 0
            processes.append(self)

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications == 1:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            self.alive = False
            return b"", b""

        def poll(self):
            self.poll_calls += 1
            return None if self.alive else self.returncode

        def kill(self):
            self.kill_calls += 1
            raise PermissionError("synthetic access denied")

    monkeypatch.setattr(audio_io.subprocess, "Popen", ExitsDuringCleanup)
    with pytest.raises(InterruptedError, match="已取消"):
        audio_io.run_hidden_cancellable([], is_cancelled, timeout=10)

    process = processes[0]
    assert process.kill_calls == 1
    assert process.poll_calls == 2
    assert process.communications == 2
    assert not context.process_cleanup_failed
    assert not context.cleanup_failures


def test_ffmpeg_poll_failure_during_cleanup_recovers_after_communicate_reaps_process(
        monkeypatch):
    from manialab.cancellation import CancellationContext

    context = CancellationContext()
    cancelled = iter((False, False, True))

    def is_cancelled():
        return next(cancelled)

    is_cancelled.report_process_cleanup_failure = context.report_process_cleanup_failure

    class UnqueryableProcess:
        def __init__(self, args, **kwargs):
            self.killed = False
            self.communications = 0
            self.poll_calls = 0
            processes.append(self)

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications == 1:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return b"", b""

        def poll(self):
            self.poll_calls += 1
            if self.poll_calls == 1:
                raise OSError("synthetic process status failure")
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

    monkeypatch.setattr(audio_io.subprocess, "Popen", UnqueryableProcess)
    processes = []
    with pytest.raises(InterruptedError, match="已取消"):
        audio_io.run_hidden_cancellable([], is_cancelled, timeout=10)

    assert processes[0].killed
    assert processes[0].communications == 2
    assert processes[0].poll_calls == 1
    assert not context.process_cleanup_failed
    assert not context.cleanup_failures


def test_ffmpeg_reap_oserror_falls_back_to_bounded_wait_and_closes_pipes(monkeypatch):
    from manialab.cancellation import CancellationContext

    context = CancellationContext()
    cancelled = iter((False, False, True))

    def is_cancelled():
        return next(cancelled)

    is_cancelled.report_process_cleanup_failure = context.report_process_cleanup_failure

    class Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class ReapFailureProcess:
        def __init__(self, args, **kwargs):
            self.alive = True
            self.communications = 0
            self.wait_timeouts = []
            self.stdout = Pipe()
            self.stderr = Pipe()
            processes.append(self)

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications == 1:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            raise OSError("synthetic pipe reaping failure")

        def poll(self):
            return None if self.alive else -9

        def kill(self):
            self.alive = False

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            self.alive = False
            return -9

    monkeypatch.setattr(audio_io.subprocess, "Popen", ReapFailureProcess)
    processes = []
    with pytest.raises(InterruptedError, match="已取消"):
        audio_io.run_hidden_cancellable([], is_cancelled, timeout=10)

    assert processes[0].wait_timeouts == pytest.approx([4.9], abs=.05)
    assert processes[0].wait_timeouts[0] <= 5
    assert processes[0].stdout.closed and processes[0].stderr.closed
    assert not context.process_cleanup_failed
    assert not context.cleanup_failures


@pytest.mark.parametrize(
    ("wait_failure", "cleanup_message"),
    [("timeout", "did not exit"), ("oserror", "could not be reaped")],
)
def test_ffmpeg_cleanup_oserror_keeps_primary_communicate_failure(
        monkeypatch, wait_failure, cleanup_message):
    from manialab.cancellation import CancellationContext

    context = CancellationContext()
    communications = []
    processes = []

    def cancelled():
        return False

    cancelled.report_process_cleanup_failure = context.report_process_cleanup_failure

    class Pipe:
        closed = False

        def close(self):
            self.closed = True

    class DoubleFailureProcess:
        def __init__(self, args, **kwargs):
            self.kill_requested = False
            self.wait_timeouts = []
            self.stdout = Pipe()
            self.stderr = Pipe()
            processes.append(self)

        def communicate(self, timeout=None):
            communications.append(timeout)
            if len(communications) == 1:
                raise OSError("synthetic primary communicate failure")
            raise OSError("synthetic cleanup communicate failure")

        def poll(self):
            return None

        def kill(self):
            self.kill_requested = True

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if wait_failure == "timeout":
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            raise OSError("synthetic cleanup wait failure")

    monkeypatch.setattr(audio_io.subprocess, "Popen", DoubleFailureProcess)
    with pytest.raises(ChildProcessError, match=cleanup_message) as error:
        audio_io.run_hidden_cancellable([], cancelled, timeout=10)

    assert isinstance(error.value.__cause__, OSError)
    assert str(error.value.__cause__) == "synthetic primary communicate failure"
    assert any(
        "synthetic cleanup communicate failure" in note
        for note in getattr(error.value, "__notes__", ())
    )
    assert any("synthetic cleanup wait failure" in note for note in getattr(error.value, "__notes__", ())) \
        is (wait_failure == "oserror")
    assert len(communications) == 2
    assert processes[0].kill_requested
    assert processes[0].wait_timeouts == pytest.approx([4.9], abs=.05)
    assert processes[0].wait_timeouts[0] <= 5
    assert processes[0].stdout.closed and processes[0].stderr.closed
    assert context.process_cleanup_failed
    assert context.cleanup_failures == (error.value.args[0],)


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
