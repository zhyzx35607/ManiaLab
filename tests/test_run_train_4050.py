from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_4050_runner_is_portable_and_reuses_local_cuda_runtime():
    runner = (ROOT / "training" / "run_train_4050.ps1").read_text(encoding="utf-8")

    assert '"D:\\' not in runner
    assert 'StartsWith("D:' not in runner
    assert "[switch]$NoMirrors" in runner
    assert "MANIALAB_TORCH_INDEX_URL" in runner
    assert "MANIALAB_PYPI_INDEX_URL" in runner
    assert "mapperatorinator-v32\\venv\\Scripts\\python.exe" in runner
    assert "HF_HUB_OFFLINE = \"1\"" in runner
    assert "TRANSFORMERS_OFFLINE = \"1\"" in runner
    assert "gradient-accumulation-steps" in runner
    assert 'models\\mapperatorinator' in runner
    assert 'mapperatorinator-v32\\source' in runner
    assert 'MANIALAB_WHISPER_CONFIG_CACHE' in runner
    assert 'mapperatorinator-v32\\cache\\huggingface' in runner
    assert '$localAppRoot "engines\\torosu_cli\\torosu-cli.exe"' in runner
