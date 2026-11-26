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
from scripts.separate_six import output_sources_for


def test_htdemucs_profiles_select_named_source_outputs():
    model_sources=('drums','bass','other','vocals','guitar','piano')
    assert output_sources_for('six',model_sources)==('drums','bass','vocals','other','guitar','piano')
    assert output_sources_for('four-piano',model_sources)==('drums','piano','vocals','other')
    with pytest.raises(ValueError,match='inventory'):
        output_sources_for('four-piano',('drums','bass','other','vocals','guitar'))


def test_separator_status_reports_profile_voices_and_manifest_availability(tmp_path):
    worker = tmp_path/'assets/deep/ManiaLabAudio'
    worker.mkdir(parents=True)
    (worker/'ManiaLabAudio.exe').write_bytes(b'worker')
    deep = worker.parent
    manifest = deep/'manifest.json'
    manifest.write_text(json.dumps({'profiles':['four','six','four-piano']}), encoding='utf-8')
    for spec in separation.PROFILES.values():
        (deep/spec['filename']).touch()

    status = separation.Separator(tmp_path).status()
    assert status['profiles']['four-piano'] == {
        'label': separation.PROFILES['four-piano']['label'],
        'voices': ['drums', 'piano', 'vocals', 'other'],
        'available': True,
    }
    assert status['profiles']['six']['voices'] == [
        'drums', 'bass', 'vocals', 'other', 'guitar', 'piano'
    ]

    manifest.write_text(json.dumps({'profiles':['four','six']}), encoding='utf-8')
    assert separation.Separator(tmp_path).status()['profiles']['four-piano']['available'] is False


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
    monkeypatch.setattr(separation, "start_worker", fail_to_start)

    with pytest.raises(OSError, match="worker cannot start"):
        separator.separate(source, tmp_path)

    assert not wave.exists()


@pytest.mark.parametrize('profile', ['four', 'six', 'four-piano'])
@pytest.mark.parametrize('worker_revision', [1, 2])
def test_separator_rebuilds_stale_algorithm_revision(tmp_path, monkeypatch, profile, worker_revision):
    monkeypatch.setenv('MANIALAB_DEEP_PYTHON', sys.executable)
    separator = separation.Separator(tmp_path, profile)
    separator.spec = separator.spec | {'algorithm_revision': 2}
    separator.model.parent.mkdir(parents=True)
    separator.model.write_bytes(b"synthetic model")
    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic audio")
    output = tmp_path / separator.spec['directory']
    output.mkdir()
    manifest = output / "separation.json"
    old_report = dict(
        revision=1,
        algorithm_revision=1,
        audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model_sha256=separator.spec['sha256'],
        profile=profile,
        sources={voice:f'{voice}.npy' for voice in separator.spec['voices']},
        input_processing=dict(separation.INPUT_PROCESSING),
    )
    manifest.write_text(json.dumps(old_report), encoding="utf-8")
    for voice in separator.spec['voices']:
        np.save(output / f"{voice}.npy", np.zeros(8, dtype=np.float32))

    wave = tmp_path / "separation-input.wav"
    launches = []

    def prepare_input(*args, **kwargs):
        wave.write_bytes(b"synthetic converted audio")
        return SimpleNamespace(returncode=0)

    class FinishedWorker:
        returncode = 0

        def poll(self):
            return self.returncode

    def launch(command, **kwargs):
        launches.append(command)
        assert command[command.index("--algorithm-revision") + 1] == "2"
        manifest.write_text(json.dumps(old_report | {'algorithm_revision': worker_revision}), encoding="utf-8")
        return FinishedWorker()

    monkeypatch.setattr(separation, "run_hidden_cancellable", prepare_input)
    monkeypatch.setattr(separation.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(separation, "start_worker", launch)

    if worker_revision != separator.spec['algorithm_revision']:
        with pytest.raises(ValueError):
            separator.separate(source, tmp_path)
        assert len(launches) == 1
        assert not wave.exists()
        return

    _, report = separator.separate(source, tmp_path)

    assert len(launches) == 1
    assert report['algorithm_revision'] == 2
    assert not report['cached']
    assert not wave.exists()


def test_model_separation_identity_includes_algorithm_revision():
    from manialab.stems import separation_identity

    report = dict(source='model', audio_sha256='audio', model_sha256='model',
                  profile='four', input_processing=separation.INPUT_PROCESSING,
                  algorithm_revision=1)
    assert separation_identity(report) != separation_identity(report | {'algorithm_revision': 2})


@pytest.mark.parametrize('profile', ['four', 'six', 'four-piano'])
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
                      model_sha256=separator.spec['sha256'], profile=profile,
                      sources={voice:f'{voice}.npy' for voice in separator.spec['voices']})
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
        assert command[command.index("--algorithm-revision") + 1] == str(separator.spec['algorithm_revision'])
        if profile != 'four':
            assert command[command.index('--profile') + 1] == profile
        assert all(kwargs["env"][name] == "2" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        rate, data = wavfile.read(command[2])
        assert rate == 44100 and data.dtype == np.float32
        np.testing.assert_array_equal(data, samples)
        launches.append(command)
        report = json.loads(manifest.read_text(encoding='utf-8'))
        report['algorithm_revision'] = int(command[command.index("--algorithm-revision") + 1])
        if '--beat-model' in command:
            report['beat_evidence'] = dict(model_sha256=separation.BEAT_MODEL_SHA256)
        manifest.write_text(json.dumps(report), encoding='utf-8')
        return FinishedWorker()

    monkeypatch.setattr(separation, 'start_worker', launch)
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
    monkeypatch.setattr(separation, 'INPUT_PROCESSING', separation.INPUT_PROCESSING | {'revision': 4})
    _, report = separator.separate(source, tmp_path, include_beats=False)
    assert len(launches) == 4
    assert "--beat-model" not in launches[-1]
    assert not report['cached']
