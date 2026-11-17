from pathlib import Path
import subprocess
import sys

from manialab import audio, audio_analysis


def test_audio_module_keeps_legacy_event_analysis_exports():
    for name in ('pitched_notes', 'rhythmic_evidence', 'fuse_attacks', 'refine_releases'):
        assert getattr(audio, name) is getattr(audio_analysis, name)


def test_audio_analysis_import_does_not_load_onnx_runtime():
    code = "import sys; import manialab.audio_analysis; assert 'onnxruntime' not in sys.modules"
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
