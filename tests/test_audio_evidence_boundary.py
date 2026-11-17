from pathlib import Path
import subprocess
import sys

from manialab import audio, audio_config, audio_evidence


def test_audio_facade_preserves_evidence_helpers_and_sample_clock():
    assert audio.SR is audio_config.SR
    assert audio.FRAME is audio_config.FRAME
    assert audio.WINDOW is audio_config.WINDOW
    assert audio.OVERLAP is audio_config.OVERLAP
    for name in (
        "align_stem_samples",
        "source_evidence",
        "timing_evidence_quality",
        "reject_source_bleed",
    ):
        assert getattr(audio, name) is getattr(audio_evidence, name)


def test_audio_evidence_import_does_not_load_inference_runtime():
    code = (
        "import sys; from manialab.audio_evidence import source_evidence; "
        "assert 'manialab.audio' not in sys.modules; "
        "assert 'onnxruntime' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
