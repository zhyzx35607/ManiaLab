from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile

import pytest

from manialab.task_errors import UserVisibleTimeoutError
from manialab.v32_runtime import (
    PYPI_MIRROR_INDEX,
    TORCH_CUDA_INDEX,
    TORCH_CUDA_MIRROR_INDEX,
    TORCH_CPU_INDEX,
    V32Runtime,
)


def _runtime_fixture(tmp_path):
    resources = tmp_path / "resources"
    assets = resources / "assets"
    (assets / "runtime").mkdir(parents=True)
    source_archive = assets / "mapperatorinator-source.zip"
    with zipfile.ZipFile(source_archive, "w") as archive:
        archive.writestr("inference.py", "# pinned inference entry\n")
        archive.writestr("configs/inference/v32.yaml", "gamemode=3\n")
    source_sha = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    (assets / "mapperatorinator-requirements.txt").write_text(
        "transformers==4.57.3\n", encoding="utf-8",
    )
    (assets / "model-manifest.json").write_text(
        json.dumps({
            "mapperatorinator": {
                "commit": "1" * 40,
                "source_archive_sha256": source_sha,
                "repository": "https://github.com/OliBomby/Mapperatorinator",
            },
        }),
        encoding="utf-8",
    )
    uv = assets / "runtime" / ("uv.exe" if __import__("os").name == "nt" else "uv")
    uv.write_bytes(b"uv fixture")
    runtime = V32Runtime(tmp_path / "app-data", resources)
    return runtime, source_archive, source_sha


def test_runtime_installer_checks_bundle_and_stages_pinned_source(tmp_path, monkeypatch):
    runtime, source_archive, source_sha = _runtime_fixture(tmp_path)
    monkeypatch.setenv("MANIALAB_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("MANIALAB_TEMP_ROOT", str(tmp_path / "temp"))
    monkeypatch.delenv("MANIALAB_V32_DEVICE", raising=False)
    monkeypatch.setattr(runtime, "_cuda_expected", lambda: True)
    commands = []

    def run_command(command, *, message, env, cwd, timeout_seconds):
        commands.append((command, message, env, cwd, timeout_seconds))
        if "venv" in command:
            runtime._default_python.parent.mkdir(parents=True, exist_ok=True)
            runtime._default_python.write_bytes(b"venv interpreter fixture")
        if command[1:2] == ["-c"]:
            return {"log": json.dumps({
                "python": "3.10.20",
                "torch": "2.9.1+cu128",
                "cuda": True,
                "gpu": "RTX 4050",
            })}
        return {"log": ""}

    result = runtime.install(
        progress=lambda *_: None,
        cancelled=lambda: False,
        run_command=run_command,
    )

    assert runtime.source.joinpath("inference.py").is_file()
    identity = json.loads(
        (runtime.source / ".manialab-source.json").read_text(encoding="utf-8")
    )
    assert identity == {"commit": "1" * 40, "sha256": source_sha}
    assert result["device"] == "cuda"
    assert result["gpu"] == "RTX 4050"
    python_install = commands[0][0]
    assert "--no-bin" in python_install
    assert ("--no-registry" in python_install) == (os.name == "nt")
    torch_install = next(row[0] for row in commands if "torch==" + "2.9.1" in " ".join(row[0]))
    assert TORCH_CUDA_INDEX in torch_install
    assert all(row[3] == runtime.source for row in commands[2:])
    assert not source_archive.parent.joinpath("source-stage").exists()


def test_runtime_installer_uses_cpu_wheels_when_no_nvidia_gpu(tmp_path, monkeypatch):
    runtime, _, _ = _runtime_fixture(tmp_path)
    monkeypatch.setenv("MANIALAB_V32_DEVICE", "cpu")
    commands = []

    def run_command(command, **kwargs):
        commands.append(command)
        if "venv" in command:
            runtime._default_python.parent.mkdir(parents=True, exist_ok=True)
            runtime._default_python.write_bytes(b"venv interpreter fixture")
        if command[1:2] == ["-c"]:
            return {"log": json.dumps({
                "python": "3.10.20",
                "torch": "2.9.1+cpu",
                "cuda": False,
                "gpu": None,
            })}
        return {"log": ""}

    runtime.install(progress=lambda *_: None, cancelled=lambda: False, run_command=run_command)

    torch_install = next(command for command in commands if "torch==2.9.1" in " ".join(command))
    assert TORCH_CPU_INDEX in torch_install


def test_runtime_installer_uses_mirrors_and_configurable_long_timeouts(
    tmp_path, monkeypatch,
):
    runtime, _, _ = _runtime_fixture(tmp_path)
    monkeypatch.delenv("MANIALAB_TORCH_INDEX_URL", raising=False)
    monkeypatch.delenv("MANIALAB_PYPI_INDEX_URL", raising=False)
    monkeypatch.setenv("MANIALAB_V32_INSTALL_TIMEOUT_SECONDS", "4321")
    monkeypatch.setattr(runtime, "_cuda_expected", lambda: True)
    commands = []

    def run_command(command, *, message, env, cwd, timeout_seconds):
        commands.append((command, message, env, cwd, timeout_seconds))
        if "venv" in command:
            runtime._default_python.parent.mkdir(parents=True, exist_ok=True)
            runtime._default_python.write_bytes(b"venv interpreter fixture")
        if command[1:2] == ["-c"]:
            return {"log": json.dumps({
                "python": "3.10.20",
                "torch": "2.9.1+cu128",
                "cuda": True,
                "gpu": "RTX 4050",
            })}
        return {"log": ""}

    runtime.install(
        progress=lambda *_: None,
        cancelled=lambda: False,
        run_command=run_command,
        use_mirrors=True,
    )

    torch_row = next(row for row in commands if "torch==2.9.1" in " ".join(row[0]))
    dependency_row = next(row for row in commands if "-r" in row[0])
    assert TORCH_CUDA_MIRROR_INDEX in torch_row[0]
    assert "--extra-index-url" in torch_row[0]
    assert torch_row[0][torch_row[0].index("--extra-index-url") + 1] == PYPI_MIRROR_INDEX
    assert torch_row[4] == 4321
    assert dependency_row[0][dependency_row[0].index("--index-url") + 1] == PYPI_MIRROR_INDEX
    assert dependency_row[4] == 4321
    assert torch_row[2]["MANIALAB_TORCH_INDEX_URL"] == TORCH_CUDA_MIRROR_INDEX
    assert torch_row[2]["MANIALAB_PYPI_INDEX_URL"] == PYPI_MIRROR_INDEX


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number", "86401"])
def test_runtime_installer_timeout_override_is_bounded(tmp_path, monkeypatch, value):
    monkeypatch.setenv("MANIALAB_V32_INSTALL_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="必须为 1–86400 秒"):
        V32Runtime._install_timeout_seconds()


def test_runtime_installer_defaults_to_two_hour_timeout(monkeypatch):
    monkeypatch.delenv("MANIALAB_V32_INSTALL_TIMEOUT_SECONDS", raising=False)
    assert V32Runtime._install_timeout_seconds() == 7200


def test_runtime_installer_failure_suggests_mirror_configuration(tmp_path, monkeypatch):
    runtime, _, _ = _runtime_fixture(tmp_path)
    monkeypatch.delenv("MANIALAB_TORCH_INDEX_URL", raising=False)
    monkeypatch.delenv("MANIALAB_PYPI_INDEX_URL", raising=False)
    monkeypatch.setattr(runtime, "_cuda_expected", lambda: True)

    def run_command(command, **kwargs):
        if "venv" in command:
            runtime._default_python.parent.mkdir(parents=True, exist_ok=True)
            runtime._default_python.write_bytes(b"venv interpreter fixture")
        if "torch==" in " ".join(command):
            raise UserVisibleTimeoutError(
                "安装 CUDA PyTorch 超过 7200 秒，已停止",
                log_path=tmp_path / "install.log",
            )
        if command[1:2] == ["-c"]:
            return {"log": json.dumps({
                "python": "3.10.20",
                "torch": "2.9.1+cu128",
                "cuda": True,
                "gpu": "RTX 4050",
            })}
        return {"log": ""}

    with pytest.raises(UserVisibleTimeoutError) as raised:
        runtime.install(
            progress=lambda *_: None,
            cancelled=lambda: False,
            run_command=run_command,
        )

    assert "MANIALAB_TORCH_INDEX_URL" in raised.value.user_message
    assert "MANIALAB_PYPI_INDEX_URL" in raised.value.user_message
    assert raised.value.log_path == str(tmp_path / "install.log")


def test_runtime_installer_rejects_corrupt_source_archive(tmp_path):
    runtime, archive, _ = _runtime_fixture(tmp_path)
    archive.write_bytes(b"modified archive")
    called = []

    with pytest.raises(ValueError, match="SHA-256"):
        runtime.install(
            progress=lambda *_: None,
            cancelled=lambda: False,
            run_command=lambda *args, **kwargs: called.append(args),
        )

    assert called == []
    assert not runtime.source.exists()


def test_runtime_installer_does_not_claim_unmarked_existing_source(tmp_path):
    runtime, _, _ = _runtime_fixture(tmp_path)
    runtime.source.mkdir(parents=True)
    (runtime.source / "inference.py").write_text("# foreign", encoding="utf-8")
    commands = []

    with pytest.raises(ValueError, match="没有 ManiaLab 固定版本标记"):
        runtime.install(
            progress=lambda *_: None,
            cancelled=lambda: False,
            run_command=lambda *args, **kwargs: commands.append(args),
        )

    assert commands == []
    assert (runtime.source / "inference.py").read_text(encoding="utf-8") == "# foreign"


def test_frozen_app_does_not_probe_itself_as_a_python_runtime(tmp_path, monkeypatch):
    import manialab.v32_runtime as runtime_module

    runtime, _, _ = _runtime_fixture(tmp_path)
    monkeypatch.setattr(runtime_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(runtime_module.sys, "executable", str(tmp_path / "ManiaLab.exe"))

    assert runtime.python == runtime._default_python
    assert runtime.python != Path(runtime_module.sys.executable)
    assert runtime.status()["runtime_python_available"] is False


def test_runtime_root_follows_portable_override(tmp_path, monkeypatch):
    runtime_root = tmp_path / "large-drive" / "ManiaLab-V32"
    monkeypatch.setenv("MANIALAB_V32_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.delenv("MANIALAB_MAPPER_PYTHON", raising=False)
    monkeypatch.setattr("manialab.v32_runtime.sys.frozen", True, raising=False)

    runtime = V32Runtime(tmp_path / "app-data", tmp_path / "resources")

    assert runtime.root == runtime_root.resolve()
    assert runtime.python == (
        runtime_root / "venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
    )


@pytest.mark.parametrize(
    "name",
    ["../outside.txt", "/absolute.txt", "C:/drive.txt", "folder\\..\\outside.txt"],
)
def test_source_archive_extractor_rejects_unsafe_members(tmp_path, name):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(name, "unsafe")

    with pytest.raises(ValueError, match="不安全路径"):
        V32Runtime._safe_extract(archive_path, tmp_path / "destination")
