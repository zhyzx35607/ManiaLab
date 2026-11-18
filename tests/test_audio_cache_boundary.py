from pathlib import Path
import subprocess
import sys


def test_audio_cache_import_does_not_load_inference_runtime():
    code = (
        "import sys; import manialab.audio_cache; "
        "assert 'manialab.audio' not in sys.modules; "
        "assert 'onnxruntime' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
