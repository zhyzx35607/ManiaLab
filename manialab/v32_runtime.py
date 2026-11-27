"""Pinned, per-user Mapperatorinator runtime bootstrap."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from .cancellation import stop_process
from .task_errors import UserVisibleTaskError, UserVisibleTimeoutError
from .worker_process import start_worker


SOURCE_PYTHON = "3.10.20"
TORCH_VERSION = "2.9.1"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_CUDA_MIRROR_INDEX = "https://mirror.nju.edu.cn/pytorch/whl/cu128/"
PYPI_MIRROR_INDEX = "https://mirrors.aliyun.com/pypi/simple/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class V32Runtime:
    def __init__(self, data_root: str | Path, resources: str | Path):
        self.data_root = Path(data_root).resolve()
        self.resources = Path(resources).resolve()
        configured_root = os.environ.get("MANIALAB_V32_RUNTIME_ROOT")
        self.root = (
            Path(configured_root).expanduser().resolve()
            if configured_root else self.data_root / "mapperatorinator-v32"
        )
        self.source = Path(os.environ["MANIALAB_MAPPER_REPO"]).resolve() if os.environ.get(
            "MANIALAB_MAPPER_REPO"
        ) else self.root / "source"
        self._python_override = (
            Path(os.environ["MANIALAB_MAPPER_PYTHON"]).resolve()
            if os.environ.get("MANIALAB_MAPPER_PYTHON") else None
        )
        self._default_python = self.root / "venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        self._nvidia_probe_at = 0.0
        self._nvidia_probe_value = False

    @property
    def python(self) -> Path:
        if self._python_override is not None:
            return self._python_override
        if self._default_python.is_file():
            return self._default_python
        if getattr(sys, "frozen", False):
            return self._default_python
        return Path(sys.executable).resolve()

    @property
    def uv(self) -> Path | None:
        bundled = self.resources / "assets" / "runtime" / (
            "uv.exe" if os.name == "nt" else "uv"
        )
        if bundled.is_file():
            return bundled
        configured = os.environ.get("MANIALAB_UV")
        if configured and Path(configured).is_file():
            return Path(configured).resolve()
        located = shutil.which("uv")
        return Path(located).resolve() if located else None

    def _manifest(self) -> dict:
        path = self.resources / "assets" / "model-manifest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))["mapperatorinator"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError("发行包缺少有效的 Mapperatorinator 固定版本清单") from error

    def status(self) -> dict:
        archive = self.resources / "assets" / "mapperatorinator-source.zip"
        requirements = self.resources / "assets" / "mapperatorinator-requirements.txt"
        bundled_source_available = archive.is_file() and requirements.is_file()
        install_available = bool(bundled_source_available and self.uv)
        source_available = (self.source / "inference.py").is_file()
        try:
            source_commit = self._manifest().get("commit")
        except ValueError:
            source_commit = None
        return {
            "source_available": source_available,
            "runtime_python_available": self.python.is_file(),
            "install_available": install_available and (
                self._python_override is None or self._python_override == self._default_python
            ),
            "source_commit": source_commit,
            "source_archive_available": bundled_source_available,
            "uv_available": self.uv is not None,
            "host_cuda_available": self.host_cuda_available(),
            "runtime_python": str(self.python) if self.python.is_file() else None,
        }

    def host_cuda_available(self, *, refresh: bool = False) -> bool:
        if not refresh and time.monotonic() - self._nvidia_probe_at < 30:
            return self._nvidia_probe_value
        smi = shutil.which("nvidia-smi")
        if not smi:
            self._nvidia_probe_value = False
        else:
            try:
                result = subprocess.run(
                    [smi, "-L"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                self._nvidia_probe_value = result.returncode == 0 and "GPU " in result.stdout
            except (OSError, subprocess.TimeoutExpired):
                self._nvidia_probe_value = False
        self._nvidia_probe_at = time.monotonic()
        return self._nvidia_probe_value

    def _cuda_expected(self) -> bool:
        requested = os.environ.get("MANIALAB_V32_DEVICE", "auto").strip().casefold()
        if requested == "cuda":
            return True
        if requested == "cpu":
            return False
        if self._default_python.is_file():
            probe = subprocess.run(
                [
                    str(self._default_python),
                    "-c",
                    "import torch; print('1' if torch.cuda.is_available() else '0')",
                ],
                env=self._install_environment(create=False),
                capture_output=True,
                text=True,
                timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if probe.returncode == 0:
                if probe.stdout.strip().endswith("1"):
                    return True
        return self.host_cuda_available(refresh=True)

    def _install_environment(self, *, create: bool) -> dict[str, str]:
        env = os.environ.copy()
        cache_root = Path(
            env.get("MANIALAB_CACHE_ROOT")
            or env.get("HF_HOME")
            or self.data_root / "cache"
        ).expanduser().resolve()
        temp_root = Path(env.get("MANIALAB_TEMP_ROOT") or cache_root / "tmp").resolve()
        temp_is_configured = any(
            env.get(name)
            for name in ("MANIALAB_TEMP_ROOT", "MANIALAB_CACHE_ROOT")
        )
        tmp_path = temp_root if temp_is_configured else Path(env.get("TMP") or temp_root)
        temp_path = temp_root if temp_is_configured else Path(env.get("TEMP") or temp_root)
        tmpdir_path = temp_root if temp_is_configured else Path(env.get("TMPDIR") or temp_root)
        python_root = Path(
            env.get("UV_PYTHON_INSTALL_DIR") or cache_root / "python"
        ).expanduser().resolve()
        paths = {
            "HF_HOME": Path(env.get("HF_HOME") or cache_root / "huggingface"),
            "HF_HUB_CACHE": Path(env.get("HF_HUB_CACHE") or cache_root / "huggingface" / "hub"),
            "TRANSFORMERS_CACHE": Path(
                env.get("TRANSFORMERS_CACHE") or cache_root / "huggingface" / "transformers"
            ),
            "TORCH_HOME": Path(env.get("TORCH_HOME") or cache_root / "torch"),
            "TORCHINDUCTOR_CACHE_DIR": Path(env.get("TORCHINDUCTOR_CACHE_DIR") or cache_root / "torch" / "inductor"),
            "XDG_CACHE_HOME": Path(env.get("XDG_CACHE_HOME") or cache_root / "xdg"),
            "PIP_CACHE_DIR": Path(env.get("PIP_CACHE_DIR") or cache_root / "pip"),
            "UV_CACHE_DIR": Path(env.get("UV_CACHE_DIR") or cache_root / "uv"),
            "UV_PYTHON_INSTALL_DIR": python_root,
            "TMP": tmp_path,
            "TEMP": temp_path,
            "TMPDIR": tmpdir_path,
        }
        for key, path in paths.items():
            if create:
                path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path.resolve())
        env.update(
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_DISABLE_SYMLINKS_WARNING="1",
        )
        return env

    @staticmethod
    def _install_timeout_seconds() -> float:
        raw = os.environ.get("MANIALAB_V32_INSTALL_TIMEOUT_SECONDS", "7200")
        try:
            timeout = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "MANIALAB_V32_INSTALL_TIMEOUT_SECONDS 必须为 1–86400 秒"
            ) from error
        if not 1 <= timeout <= 86400:
            raise ValueError(
                "MANIALAB_V32_INSTALL_TIMEOUT_SECONDS 必须为 1–86400 秒"
            )
        return timeout

    @staticmethod
    def _safe_extract(source_zip: Path, destination: Path) -> None:
        root = destination.resolve()
        with zipfile.ZipFile(source_zip) as archive:
            entries = archive.infolist()
            for info in entries:
                member = PurePosixPath(info.filename.replace("\\", "/"))
                if (
                    member.is_absolute()
                    or any(
                        part in ("", ".", "..") or ":" in part
                        for part in member.parts
                    )
                ):
                    raise ValueError(f"固定 Mapperatorinator 源码含不安全路径：{info.filename}")
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError("固定 Mapperatorinator 源码不能包含符号链接")
            for info in entries:
                member = PurePosixPath(info.filename.replace("\\", "/"))
                target = (root / Path(*member.parts)).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"固定 Mapperatorinator 源码路径越界：{info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)

    def _prepare_source(self) -> None:
        if (self.source / "inference.py").is_file():
            marker = self.source / ".manialab-source.json"
            manifest = self._manifest()
            if not marker.is_file():
                raise ValueError(
                    "已有 Mapperatorinator 源码没有 ManiaLab 固定版本标记；"
                    "请通过 MANIALAB_MAPPER_REPO 显式复用该只读源码"
                )
            identity = json.loads(marker.read_text(encoding="utf-8"))
            if (
                identity.get("commit") != manifest["commit"]
                or identity.get("sha256") != manifest["source_archive_sha256"]
            ):
                raise ValueError("已安装的 Mapperatorinator 源码与锁定提交不一致")
            return
        if self.source.exists() and any(self.source.iterdir()):
            raise ValueError(
                f"源码目录非空但不属于 ManiaLab 固定安装，拒绝覆盖：{self.source}"
            )
        archive = self.resources / "assets" / "mapperatorinator-source.zip"
        manifest = self._manifest()
        if sha256_file(archive) != manifest["source_archive_sha256"]:
            raise ValueError("发行包内 Mapperatorinator 源码归档 SHA-256 不匹配")
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".source-stage-", dir=self.root))
        try:
            self._safe_extract(archive, staging)
            if not (staging / "inference.py").is_file():
                raise FileNotFoundError("固定源码归档缺少 inference.py")
            (staging / ".manialab-source.json").write_text(
                json.dumps(
                    {
                        "commit": manifest["commit"],
                        "sha256": manifest["source_archive_sha256"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.source.parent.mkdir(parents=True, exist_ok=True)
            if self.source.exists():
                if any(self.source.iterdir()):
                    raise FileExistsError(f"源码目标在安装期间出现且非空：{self.source}")
                self.source.rmdir()
            staging.replace(self.source)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def install(self, *, progress, cancelled, run_command, use_mirrors=False) -> dict:
        uv = self.uv
        if uv is None:
            raise ValueError("发行包缺少 uv；请安装支持 V32 的完整 Windows 包")
        if type(use_mirrors) is not bool:
            raise ValueError("国内镜像选项必须为布尔值")
        if self._python_override is not None and self._python_override != self._default_python:
            raise ValueError("已显式配置外部 Mapperatorinator Python，拒绝自动替换该环境")
        self._prepare_source()
        env = self._install_environment(create=True)
        if use_mirrors:
            if not env.get("MANIALAB_TORCH_INDEX_URL") and self._cuda_expected():
                env["MANIALAB_TORCH_INDEX_URL"] = TORCH_CUDA_MIRROR_INDEX
            if not env.get("MANIALAB_PYPI_INDEX_URL"):
                env["MANIALAB_PYPI_INDEX_URL"] = PYPI_MIRROR_INDEX

        def run_install_step(command, *, message, timeout_seconds, **kwargs):
            try:
                return run_command(
                    command,
                    message=message,
                    timeout_seconds=timeout_seconds,
                    **kwargs,
                )
            except (UserVisibleTaskError, UserVisibleTimeoutError) as error:
                mirror_hint = (
                    "；可勾选“使用国内镜像加速”，或设置 "
                    "MANIALAB_TORCH_INDEX_URL / MANIALAB_PYPI_INDEX_URL"
                    "（南大 CUDA 与阿里云 PyPI）"
                )
                error_type = (
                    UserVisibleTimeoutError
                    if isinstance(error, UserVisibleTimeoutError)
                    else UserVisibleTaskError
                )
                raise error_type(
                    f"{error.user_message}{mirror_hint}",
                    log_path=error.log_path,
                    global_failure=error.global_failure,
                ) from error

        if not self._default_python.is_file():
            run_install_step(
                [
                    str(uv), "python", "install", SOURCE_PYTHON,
                    "--install-dir", env["UV_PYTHON_INSTALL_DIR"],
                    "--no-bin",
                    *(["--no-registry"] if os.name == "nt" else []),
                    "--cache-dir", env["UV_CACHE_DIR"],
                ],
                message=f"准备 Mapperatorinator Python {SOURCE_PYTHON}",
                env=env,
                cwd=self.root,
                timeout_seconds=900,
            )
            run_install_step(
                [
                    str(uv), "venv", "--managed-python", "--python", SOURCE_PYTHON,
                    str(self._default_python.parent.parent),
                ],
                message="创建隔离的 V32 推理环境",
                env=env,
                cwd=self.root,
                timeout_seconds=300,
            )
        python = self._default_python
        if not python.is_file():
            raise FileNotFoundError(f"uv 未创建 V32 Python 环境：{python}")
        cuda = self._cuda_expected()
        index = env.get("MANIALAB_TORCH_INDEX_URL") or (
            TORCH_CUDA_INDEX if cuda else TORCH_CPU_INDEX
        )
        install_timeout = self._install_timeout_seconds()
        torch_command = [
            str(uv), "pip", "install", "--python", str(python),
            "--index-url", index,
        ]
        pypi_index = env.get("MANIALAB_PYPI_INDEX_URL")
        if pypi_index and pypi_index != index:
            torch_command.extend(("--extra-index-url", pypi_index))
        torch_command.extend((
            f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}",
        ))
        run_install_step(
            torch_command,
            message=(
                "安装 CUDA PyTorch（RTX/NVIDIA）"
                if cuda else "安装 CPU PyTorch"
            ),
            env=env,
            cwd=self.source,
            timeout_seconds=install_timeout,
        )
        dependency_command = [
            str(uv), "pip", "install", "--python", str(python),
        ]
        if pypi_index:
            dependency_command.extend(("--index-url", pypi_index))
        dependency_command.extend((
            "-r", str(self.resources / "assets" / "mapperatorinator-requirements.txt"),
        ))
        run_install_step(
            dependency_command,
            message="安装 Mapperatorinator 固定依赖",
            env=env,
            cwd=self.source,
            timeout_seconds=install_timeout,
        )
        verification = run_install_step(
            [
                str(python), "-c",
                "import json,torch; from pathlib import Path; "
                "print(json.dumps({'python':__import__('sys').version.split()[0],"
                "'torch':torch.__version__,'cuda':torch.cuda.is_available(),"
                "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
                "'source':Path.cwd().as_posix()}))",
            ],
            message="验证 V32 GPU/CPU 运行环境",
            env=env,
            cwd=self.source,
            timeout_seconds=30,
        )
        try:
            result = json.loads(verification["log"].splitlines()[-1])
        except (KeyError, IndexError, json.JSONDecodeError) as error:
            raise UserVisibleTaskError(
                "V32 环境验证没有返回有效设备信息",
                log_path=verification.get("log_path"),
            ) from error
        progress(100, "V32 推理环境已就绪")
        return {
            "source_commit": self._manifest()["commit"],
            "python": str(python),
            "torch": result["torch"],
            "device": "cuda" if result["cuda"] else "cpu",
            "gpu": result["gpu"],
            "install_target": "cuda" if cuda else "cpu",
        }
