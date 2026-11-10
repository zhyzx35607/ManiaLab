import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from manialab import separation


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
    launches = []
    real_popen = subprocess.Popen

    class FinishedWorker:
        returncode = 0

        def poll(self):
            return self.returncode

    def launch(command, **kwargs):
        if command[0] != sys.executable:
            return real_popen(command, **kwargs)  # Run the real ffmpeg input conversion.
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
