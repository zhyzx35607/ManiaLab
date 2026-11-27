"""Portable Mapperatorinator V32 inference adapter."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .v32_device import DevicePlan, select_device_plan
from .v32_process import run_owned_command
from .v32_runtime import V32Runtime


MODEL_SPECS = {
    "v32": {
        "repo": "OliBomby/Mapperatorinator-v32",
        "revision": "74f22583400d259bf424819e11027c17933efe54",
        "config": "v32",
    },
    "v32-mini": {
        "repo": "OliBomby/Mapperatorinator-v32-mini",
        "revision": "7807f0dc70cab671be012e1f5ddf945b0b8b7278",
        "config": "v32-mini",
    },
}
_REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
)


def _path_from_env(name: str, fallback: Path) -> Path:
    configured = os.environ.get(name)
    return Path(configured).expanduser().resolve() if configured else fallback.resolve()


class V32Engine:
    """Runs the upstream CLI in an owned worker process for cancellable jobs."""

    def __init__(
        self,
        data_root: str | Path,
        resources: str | Path,
        *,
        mapper_repo: str | Path | None = None,
        python: str | Path | None = None,
        model_cache: str | Path | None = None,
        probe_ttl_seconds: float = 15,
    ):
        self.data_root = Path(data_root).resolve()
        self.resources = Path(resources).resolve()
        self.runtime = V32Runtime(self.data_root, self.resources)
        self.mapper_repo = Path(mapper_repo).resolve() if mapper_repo else (
            Path(os.environ["MANIALAB_MAPPER_REPO"]).resolve()
            if os.environ.get("MANIALAB_MAPPER_REPO")
            else self.runtime.source
        )
        self._python_override = Path(python).resolve() if python else None
        default_model_cache = (
            Path(os.environ["MANIALAB_CACHE_ROOT"]).expanduser().resolve() / "huggingface" / "manialab"
            if os.environ.get("MANIALAB_CACHE_ROOT")
            else Path(os.environ["HF_HOME"]).expanduser().resolve() / "manialab"
            if os.environ.get("HF_HOME")
            else self.data_root / "models" / "mapperatorinator"
        )
        self.model_cache = Path(model_cache).resolve() if model_cache else _path_from_env(
            "MANIALAB_V32_MODEL_CACHE",
            default_model_cache,
        )
        self.probe_ttl_seconds = probe_ttl_seconds
        self._probe_lock = threading.Lock()
        self._last_probe_at = 0.0
        self._runtime_probe: dict | None = None
        self._last_lora_path: str | None = None
        self._last_lora_loaded = False

    @property
    def python(self) -> Path:
        return self._python_override or self.runtime.python

    def model_directory(self, variant: str) -> Path:
        spec = self._model_spec(variant)
        return self.model_cache / f"Mapperatorinator-{variant}-{spec['revision'][:7]}" / "gamemode=3"

    @staticmethod
    def _model_spec(variant: str) -> dict:
        try:
            return MODEL_SPECS[variant]
        except KeyError as error:
            raise ValueError("模型版本必须为 v32 或 v32-mini") from error

    def _runtime_environment(self, *, create: bool = False) -> dict[str, str]:
        env = os.environ.copy()
        cache_root_value = env.get("MANIALAB_CACHE_ROOT")
        if cache_root_value:
            cache_root = Path(cache_root_value)
        elif env.get("HF_HOME"):
            cache_root = Path(env["HF_HOME"]).expanduser().parent
        else:
            cache_root = self.data_root / "cache"
        cache_root = cache_root.expanduser().resolve()
        temp_root = Path(
            env.get("MANIALAB_TEMP_ROOT")
            or cache_root / "tmp"
        ).expanduser().resolve()
        temp_is_configured = any(
            env.get(name)
            for name in ("MANIALAB_TEMP_ROOT", "MANIALAB_CACHE_ROOT", "HF_HOME")
        )
        tmp_path = temp_root if temp_is_configured else Path(env.get("TMP") or temp_root)
        temp_path = temp_root if temp_is_configured else Path(env.get("TEMP") or temp_root)
        tmpdir_path = temp_root if temp_is_configured else Path(env.get("TMPDIR") or temp_root)
        path_values = {
            "HF_HOME": Path(env.get("HF_HOME") or cache_root / "huggingface"),
            "HF_HUB_CACHE": Path(env.get("HF_HUB_CACHE") or cache_root / "huggingface" / "hub"),
            "TRANSFORMERS_CACHE": Path(
                env.get("TRANSFORMERS_CACHE") or cache_root / "huggingface" / "transformers"
            ),
            "TORCH_HOME": Path(env.get("TORCH_HOME") or cache_root / "torch"),
            "TORCHINDUCTOR_CACHE_DIR": Path(
                env.get("TORCHINDUCTOR_CACHE_DIR") or cache_root / "torch" / "inductor"
            ),
            "XDG_CACHE_HOME": Path(env.get("XDG_CACHE_HOME") or cache_root / "xdg"),
            "PIP_CACHE_DIR": Path(env.get("PIP_CACHE_DIR") or cache_root / "pip"),
            "UV_CACHE_DIR": Path(env.get("UV_CACHE_DIR") or cache_root / "uv"),
            "TMP": tmp_path,
            "TEMP": temp_path,
            "TMPDIR": tmpdir_path,
        }
        for key, value in path_values.items():
            if create:
                value.mkdir(parents=True, exist_ok=True)
            env[key] = str(value.resolve())
        env.update(
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_DISABLE_SYMLINKS_WARNING="1",
        )
        return env

    def _probe(self, *, refresh: bool = False) -> dict:
        now = time.monotonic()
        if (
            not refresh
            and self._runtime_probe is not None
            and now - self._last_probe_at < self.probe_ttl_seconds
        ):
            return dict(self._runtime_probe)
        with self._probe_lock:
            now = time.monotonic()
            if (
                not refresh
                and self._runtime_probe is not None
                and now - self._last_probe_at < self.probe_ttl_seconds
            ):
                return dict(self._runtime_probe)
            if not self.python.is_file():
                result = {"available": False, "error": "推理 Python 不存在"}
            else:
                script = (
                    "import json\n"
                    "try:\n"
                    " import torch\n"
                    " cuda=torch.cuda.is_available()\n"
                    " if cuda:\n"
                    "  p=torch.cuda.get_device_properties(torch.cuda.current_device())\n"
                    "  try: bf16=bool(torch.cuda.is_bf16_supported(including_emulation=False))\n"
                    "  except TypeError: bf16=bool(torch.cuda.is_bf16_supported())\n"
                    "  result={'available':True,'cuda':True,'name':str(p.name),"
                    "'memory_gib':round(p.total_memory/2**30,2),"
                    "'bf16':bf16}\n"
                    " else:\n"
                    "  result={'available':True,'cuda':False}\n"
                    " print(json.dumps(result))\n"
                    "except Exception as error:\n"
                    " print(json.dumps({'available':False,'error':str(error)}))\n"
                )
                try:
                    completed = subprocess.run(
                        [str(self.python), "-c", script],
                        cwd=self.mapper_repo if self.mapper_repo.is_dir() else self.data_root,
                        env=self._runtime_environment(),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=20,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    payload = next(
                        line for line in reversed(completed.stdout.splitlines())
                        if line.startswith("{")
                    )
                    result = json.loads(payload)
                    if completed.returncode and result.get("available"):
                        result = {"available": False, "error": "推理环境检查失败"}
                except (OSError, subprocess.TimeoutExpired, StopIteration, json.JSONDecodeError) as error:
                    result = {"available": False, "error": str(error)[:300]}
            self._runtime_probe = result
            self._last_probe_at = time.monotonic()
            return dict(result)

    def _device_plan(self, requested: str = "auto", *, runtime: dict | None = None) -> DevicePlan:
        if requested == "auto":
            requested = os.environ.get("MANIALAB_V32_DEVICE", "auto").strip().casefold()
        runtime = runtime or self._probe(refresh=True)
        if not runtime.get("available"):
            raise ValueError(runtime.get("error") or "Mapperatorinator 推理环境未就绪")
        if requested not in ("auto", "cpu", "cuda"):
            raise ValueError("device 必须为 auto、cpu 或 cuda")
        raw_threads = os.environ.get("MANIALAB_V32_CPU_THREADS", "")
        try:
            threads = None if not raw_threads else int(raw_threads)
        except ValueError as error:
            raise ValueError("MANIALAB_V32_CPU_THREADS 必须为 1–6") from error
        if requested == "cuda" and not runtime.get("cuda"):
            raise ValueError("已显式选择 CUDA，但当前推理 Python 环境没有可用 CUDA")
        if requested == "cpu" or not runtime.get("cuda"):
            return select_device_plan(cpu_threads=threads)
        memory_gib = float(runtime["memory_gib"])
        batch = 1 if memory_gib <= 8 else 2 if memory_gib <= 12 else 4
        return DevicePlan(
            "cuda",
            str(runtime.get("name") or "CUDA"),
            memory_gib,
            "bf16" if runtime.get("bf16") else "fp16",
            True,
            batch,
            min(6, max(1, threads or os.cpu_count() or 1)),
        )

    def status(self) -> dict:
        source_available = (self.mapper_repo / "inference.py").is_file()
        runtime = self._probe()
        runtime_status = self.runtime.status()
        device_error = None
        plan = None
        if runtime.get("available"):
            try:
                plan = self._device_plan(runtime=runtime)
            except ValueError as error:
                device_error = str(error)
        device = {
            "type": plan.device if plan else ("cuda" if runtime.get("cuda") else "cpu"),
            "name": (
                plan.name if plan else str(runtime.get("name") or
                                           ("CUDA" if runtime.get("cuda") else
                                            "CPU" if runtime.get("available") else
                                            "CPU (runtime unavailable)"))
            ),
            "memory_gib": plan.memory_gib if plan else runtime.get("memory_gib"),
            "precision": plan.precision if plan else (
                "bf16" if runtime.get("bf16") else "fp16"
                if runtime.get("cuda") else "fp32"
            ),
            "fast_decoder_loop": plan.fast_decoder_loop if plan else False,
        }
        models = {}
        for variant, spec in MODEL_SPECS.items():
            directory = self.model_directory(variant)
            models[variant] = {
                "ready": all((directory / name).is_file() for name in _REQUIRED_MODEL_FILES),
                "revision": spec["revision"],
            }
        return {
            "available": source_available and bool(runtime.get("available")) and device_error is None,
            "source_available": source_available,
            "runtime_available": bool(runtime.get("available")),
            "runtime_error": runtime.get("error"),
            "install_available": bool(runtime_status["install_available"])
            and self._python_override is None
            and self.mapper_repo.resolve() == self.runtime.source.resolve(),
            "host_cuda_available": runtime_status["host_cuda_available"],
            "runtime_needs_cuda": bool(runtime_status["host_cuda_available"])
            and bool(runtime.get("available"))
            and not bool(runtime.get("cuda")),
            "runtime_python": runtime_status["runtime_python"],
            "device_override_error": device_error,
            "device": device,
            "models": models,
            "lora": {
                "loaded": self._last_lora_loaded,
                "path": self._last_lora_path,
            },
        }

    @staticmethod
    def validate_options(
        *,
        model_variant: str,
        generation_strategy: str,
        torosu_density: str,
        difficulty: float,
        hold_note_ratio: float,
        mania_column_temperature: float,
        lora_path: str | Path | None,
        device: str = "auto",
        keys: int = 10,
        torosu_keep_jacks: bool = True,
        torosu_ln_mode: str = "keep",
        preset: str | None = None,
    ) -> dict:
        V32Engine._model_spec(model_variant)
        if type(keys) is not int or keys not in range(6, 11):
            raise ValueError("目标键数必须为 6–10")
        if generation_strategy not in (
            "7k_to_torosu_10k",
            "native_10k",
            "7k_to_torosu",
            "native",
        ):
            raise ValueError(
                "生成策略必须为 7k_to_torosu、native 或兼容旧值 "
                "7k_to_torosu_10k、native_10k"
            )
        if generation_strategy == "7k_to_torosu_10k":
            if keys != 10:
                raise ValueError("旧策略 7k_to_torosu_10k 的目标键数固定为 10K")
            model_keycount, target_keys = 7, 10
        elif generation_strategy == "native_10k":
            if keys != 10:
                raise ValueError("旧策略 native_10k 的键数固定为 10K")
            model_keycount, target_keys = 10, 10
        elif generation_strategy == "7k_to_torosu":
            model_keycount, target_keys = 7, keys
        else:
            model_keycount, target_keys = keys, keys
        if torosu_density not in ("none", "dense", "more-dense", "full"):
            raise ValueError("Torosu 密度必须为 none、dense、more-dense 或 full")
        if type(torosu_keep_jacks) is not bool:
            raise ValueError("Torosu keep_jacks 必须为布尔值")
        if torosu_ln_mode not in ("keep", "fln", "short_ln"):
            raise ValueError("Torosu LN 模式必须为 keep、fln 或 short_ln")
        if preset is not None:
            if not isinstance(preset, str):
                raise ValueError("预设 ID 必须为文本")
            from .presets import canonical_preset_id

            preset = canonical_preset_id(preset)
        for name, value, low, high in (
            ("difficulty", difficulty, 0, 10),
            ("hold_note_ratio", hold_note_ratio, 0, 0.85),
            ("mania_column_temperature", mania_column_temperature, 0.1, 2),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} 必须在 {low}–{high} 范围内")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 必须为 auto、cpu 或 cuda")
        normalized_lora = None
        if lora_path:
            normalized_lora = Path(lora_path).expanduser().resolve()
            if not all(
                (normalized_lora / name).is_file()
                for name in ("adapter_config.json", "adapter_model.safetensors")
            ):
                raise ValueError("LoRA 目录缺少 adapter_config.json 或 adapter_model.safetensors")
        return {
            "model_variant": model_variant,
            "generation_strategy": generation_strategy,
            "model_keycount": model_keycount,
            "keys": target_keys,
            "torosu_density": torosu_density,
            "torosu_keep_jacks": torosu_keep_jacks,
            "torosu_ln_mode": torosu_ln_mode,
            "preset": preset,
            "difficulty": float(difficulty),
            "hold_note_ratio": float(hold_note_ratio),
            "mania_column_temperature": float(mania_column_temperature),
            "lora_path": normalized_lora,
            "device": device,
        }

    def build_command(
        self,
        *,
        model_variant: str,
        audio_path: str | Path,
        reference_chart: str | Path,
        output_dir: str | Path,
        model_directory: str | Path | None = None,
        keycount: int,
        difficulty: float,
        hold_note_ratio: float,
        mania_column_temperature: float = 0.8,
        lora_path: str | Path | None = None,
        device_plan: DevicePlan,
        start_ms: int = 0,
        end_ms: int | None = None,
        seed: int = 20261126,
    ) -> list[str]:
        spec = self._model_spec(model_variant)
        if type(keycount) is not int or keycount not in range(6, 11):
            raise ValueError("Mapperatorinator V32 键数必须为 6–10")
        if type(start_ms) is not int or start_ms < 0:
            raise ValueError("推理起点必须为非负毫秒整数")
        if end_ms is not None and (type(end_ms) is not int or end_ms <= start_ms):
            raise ValueError("推理终点必须晚于起点")
        if not math.isfinite(difficulty) or not 0 <= difficulty <= 10:
            raise ValueError("difficulty 必须在 0–10 之间")
        if not math.isfinite(hold_note_ratio) or not 0 <= hold_note_ratio <= 0.85:
            raise ValueError("hold_note_ratio 必须在 0–0.85 之间")
        if not math.isfinite(mania_column_temperature) or not 0.1 <= mania_column_temperature <= 2:
            raise ValueError("mania_column_temperature 必须在 0.1–2 之间")

        def hydra_path(value: str | Path) -> str:
            return json.dumps(Path(value).resolve().as_posix())

        model = Path(model_directory).resolve() if model_directory else self.model_directory(model_variant)
        command = [
            str(self.python),
            str(self.mapper_repo / "inference.py"),
            "--config-name",
            spec["config"],
            f"model_path={hydra_path(model)}",
            "auto_select_gamemode_model=false",
            "gamemode=3",
            f"keycount={keycount}",
            f"circle_size={keycount}",
            f"hold_note_ratio={hold_note_ratio:.4f}",
            f"difficulty={difficulty:.3f}",
            f"mania_column_temperature={mania_column_temperature:.3f}",
            f"device={device_plan.device}",
            f"precision={device_plan.precision}",
            f"fast_decoder_loop={str(device_plan.fast_decoder_loop).lower()}",
            "super_timing_fast_loop=false",
            "use_server=false",
            "compile=false",
            "generate_positions=false",
            f"max_batch_size={device_plan.max_batch_size}",
            f"audio_path={hydra_path(audio_path)}",
            f"beatmap_path={hydra_path(reference_chart)}",
            f"output_path={hydra_path(output_dir)}",
            f"hydra.run.dir={hydra_path(Path(output_dir) / 'hydra-logs')}",
            "in_context=[TIMING]",
            "output_type=[MAP]",
            f"seed={seed}",
            "hitsounded=false",
        ]
        if end_ms is not None:
            command.extend((f"start_time={start_ms}", f"end_time={end_ms}"))
        if lora_path:
            command.append(f"lora_path={hydra_path(lora_path)}")
        return command

    def _run(
        self,
        command: list[str],
        *,
        log_path: Path,
        progress,
        cancelled,
        message: str,
        cwd: Path | None = None,
        timeout_seconds: float = 3600,
        env: dict[str, str] | None = None,
        measure_memory: bool = True,
    ) -> dict:
        env = env or self._runtime_environment(create=True)
        return run_owned_command(
            [str(part) for part in command],
            log_path=log_path,
            cwd=cwd or self.mapper_repo,
            env=env,
            progress=progress,
            cancelled=cancelled,
            message=message,
            timeout_seconds=timeout_seconds,
            monitor_python=self.python if measure_memory else None,
        )

    def install_runtime(self, *, progress, cancelled) -> dict:
        if self._python_override is not None or self.mapper_repo.resolve() != self.runtime.source.resolve():
            raise ValueError("当前已显式指定外部 V32 运行环境，无需自动安装")
        install_id = f"{os.getpid()}-{time.time_ns()}"

        def run_command(
            command,
            *,
            message,
            env,
            cwd,
            timeout_seconds,
        ):
            log_path = (
                self.runtime.root / "logs" / f"{install_id}-{len(command)}-{abs(hash(message))}.log"
            )
            return self._run(
                [str(part) for part in command],
                log_path=log_path,
                progress=progress,
                cancelled=cancelled,
                message=message,
                cwd=Path(cwd),
                timeout_seconds=timeout_seconds,
                env=env,
                measure_memory=False,
            )

        return self.runtime.install(
            progress=progress,
            cancelled=cancelled,
            run_command=run_command,
        )

    def ensure_model(
        self,
        variant: str,
        *,
        progress,
        cancelled,
        model_directory: str | Path | None = None,
    ) -> Path:
        spec = self._model_spec(variant)
        destination = Path(model_directory).resolve() if model_directory else (
            self.model_cache / f"Mapperatorinator-{variant}-{spec['revision'][:7]}"
        )
        checkpoint = destination / "gamemode=3"
        if all((checkpoint / name).is_file() for name in _REQUIRED_MODEL_FILES):
            manifest_path = destination / "model-revision.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("repo") != spec["repo"] or manifest.get("revision") != spec["revision"]:
                    raise ValueError(f"{variant} 本地缓存 revision 与固定版本不一致")
                expected_hashes = manifest.get("sha256", {})
                for name in _REQUIRED_MODEL_FILES:
                    expected = expected_hashes.get(name)
                    if expected and self._sha256_file(checkpoint / name) != expected:
                        raise ValueError(f"{variant} 本地缓存校验失败：{name}")
            else:
                self._verify_hf_local_snapshot(destination, spec)
                manifest_path.write_text(
                    json.dumps(
                        {
                            "repo": spec["repo"],
                            "revision": spec["revision"],
                            "sha256": {
                                name: self._sha256_file(checkpoint / name)
                                for name in _REQUIRED_MODEL_FILES
                            },
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            return checkpoint
        if not self.python.is_file():
            raise FileNotFoundError(f"推理 Python 不存在：{self.python}")
        loader = (
            "import hashlib,json,sys\n"
            "from pathlib import Path\n"
            "from huggingface_hub import HfApi,snapshot_download\n"
            "repo,revision,destination,cache=sys.argv[1:]\n"
            "info=HfApi().model_info(repo,revision=revision)\n"
            "if info.sha != revision: raise RuntimeError('model revision mismatch')\n"
            "snapshot_download(repo_id=repo,revision=revision,local_dir=destination,"
            "cache_dir=cache,allow_patterns=('gamemode=3/*',))\n"
            "model=Path(destination)/'gamemode=3'\n"
            "required=('config.json','generation_config.json','model.safetensors','tokenizer.json')\n"
            "if not all((model/name).is_file() for name in required): "
            "raise FileNotFoundError('missing gamemode=3 checkpoint files')\n"
            "def sha(path):\n"
            " h=hashlib.sha256()\n"
            " with path.open('rb') as f:\n"
            "  for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)\n"
            " return h.hexdigest()\n"
            "Path(destination,'model-revision.json').write_text(json.dumps("
            "{'repo':repo,'revision':revision,'sha256':{n:sha(model/n) for n in required}},"
            "indent=2),encoding='utf-8')\n"
        )
        env = self._runtime_environment(create=True)
        cache = env["HF_HOME"]
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python),
            "-c",
            loader,
            spec["repo"],
            spec["revision"],
            str(destination),
            cache,
        ]
        self._run(
            command,
            log_path=destination / "download.log",
            progress=progress,
            cancelled=cancelled,
            message=f"下载 {variant} mania 权重",
            cwd=self.mapper_repo if self.mapper_repo.is_dir() else self.data_root,
            timeout_seconds=3600,
        )
        return checkpoint

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _verify_hf_local_snapshot(cls, destination: Path, spec: dict) -> None:
        metadata_root = destination / ".cache" / "huggingface" / "download" / "gamemode=3"
        for name in _REQUIRED_MODEL_FILES:
            metadata_path = metadata_root / f"{name}.metadata"
            if not metadata_path.is_file():
                raise ValueError(
                    f"拒绝使用缺少固定 revision 元数据的模型文件：{name}"
                )
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
            if len(lines) < 2 or lines[0] != spec["revision"]:
                raise ValueError(f"模型文件不属于固定 revision：{name}")
            file_path = destination / "gamemode=3" / name
            if len(lines[1]) == 64:
                actual_digest = cls._sha256_file(file_path)
            elif len(lines[1]) == 40:
                digest = hashlib.sha1()
                digest.update(f"blob {file_path.stat().st_size}\0".encode("ascii"))
                with file_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                actual_digest = digest.hexdigest()
            else:
                raise ValueError(f"模型文件缓存摘要格式无效：{name}")
            if actual_digest != lines[1]:
                raise ValueError(f"模型文件 SHA-256 校验失败：{name}")

    def generate(
        self,
        *,
        model_variant: str,
        audio_path: str | Path,
        reference_chart: str | Path,
        output_dir: str | Path,
        keycount: int,
        difficulty: float,
        hold_note_ratio: float,
        mania_column_temperature: float = 0.8,
        lora_path: str | Path | None = None,
        device: str = "auto",
        progress,
        cancelled,
        start_ms: int = 0,
        end_ms: int | None = None,
        model_directory: str | Path | None = None,
        seed: int = 20261126,
        target_keycount: int | None = None,
        generation_strategy: str | None = None,
        torosu_density: str = "none",
        torosu_keep_jacks: bool = True,
        torosu_ln_mode: str = "keep",
        preset: str | None = None,
    ) -> dict:
        if generation_strategy is None:
            generation_strategy = "native"
            target_keycount = keycount if target_keycount is None else target_keycount
        options = self.validate_options(
            model_variant=model_variant,
            generation_strategy=generation_strategy,
            torosu_density=torosu_density,
            difficulty=difficulty,
            hold_note_ratio=hold_note_ratio,
            mania_column_temperature=mania_column_temperature,
            lora_path=lora_path,
            device=device,
            keys=keycount if target_keycount is None else target_keycount,
            torosu_keep_jacks=torosu_keep_jacks,
            torosu_ln_mode=torosu_ln_mode,
            preset=preset,
        )
        if not (self.mapper_repo / "inference.py").is_file():
            raise FileNotFoundError(f"Mapperatorinator inference.py 不存在：{self.mapper_repo}")
        device_plan = self._device_plan(requested=device)
        checkpoint = self.ensure_model(
            model_variant,
            progress=progress,
            cancelled=cancelled,
            model_directory=model_directory,
        )
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        command = self.build_command(
            model_variant=model_variant,
            audio_path=audio_path,
            reference_chart=reference_chart,
            output_dir=output,
            model_directory=checkpoint,
            keycount=keycount,
            difficulty=options["difficulty"],
            hold_note_ratio=options["hold_note_ratio"],
            mania_column_temperature=options["mania_column_temperature"],
            lora_path=options["lora_path"],
            device_plan=device_plan,
            start_ms=start_ms,
            end_ms=end_ms,
            seed=seed,
        )
        env = self._runtime_environment(create=True)
        env.update(
            OMP_NUM_THREADS=str(device_plan.cpu_threads),
            MKL_NUM_THREADS=str(device_plan.cpu_threads),
            OPENBLAS_NUM_THREADS=str(device_plan.cpu_threads),
            NUMEXPR_NUM_THREADS=str(device_plan.cpu_threads),
        )
        log_path = output / "inference.log"
        measurement = self._run(
            command,
            log_path=log_path,
            progress=progress,
            cancelled=cancelled,
            message=f"{model_variant} · {device_plan.name} 推理",
            cwd=self.mapper_repo,
            env=env,
        )
        log = measurement["log"]
        maps = sorted(output.glob("*.osu"))
        if len(maps) != 1:
            raise RuntimeError(f"Mapperatorinator 应生成 1 张谱面，实际找到 {len(maps)} 张")
        lora_loaded = bool(
            options["lora_path"]
            and f"Loaded LoRA weights from {options['lora_path'].as_posix()}" in log.replace("\\", "/")
        )
        if options["lora_path"] and not lora_loaded:
            raise RuntimeError("请求的 LoRA 未在上游日志中确认加载")
        self._last_lora_path = str(options["lora_path"]) if options["lora_path"] else None
        self._last_lora_loaded = lora_loaded
        progress(92, "V32 推理完成，校验谱面")
        report = {
            "map": maps[0],
            "log": log_path,
            "model_variant": model_variant,
            "model_revision": MODEL_SPECS[model_variant]["revision"],
            "device": device_plan.data(),
            "elapsed_seconds": measurement["elapsed_seconds"],
            "peak_rss_bytes": measurement["peak_rss_bytes"],
            "peak_rss_mib": measurement["peak_rss_mib"],
            "fast_decoder_loop_active": (
                device_plan.fast_decoder_loop
                and "disabling on 'cpu'" not in log
                and "cuda graph capture is not supported" not in log.casefold()
            ),
            "lora_loaded": lora_loaded,
            "lora_path": str(options["lora_path"]) if options["lora_path"] else None,
            "generation_strategy": options["generation_strategy"],
            "model_keycount": options["model_keycount"],
            "keys": options["keys"],
            "torosu_density": options["torosu_density"],
            "torosu_keep_jacks": options["torosu_keep_jacks"],
            "torosu_ln_mode": options["torosu_ln_mode"],
            "preset": options["preset"],
        }
        (output / "run-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return report
