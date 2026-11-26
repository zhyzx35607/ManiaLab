from __future__ import annotations

from types import SimpleNamespace

import pytest

from manialab.v32_engine import DevicePlan, MODEL_SPECS, V32Engine, select_device_plan


class FakeCuda:
    def __init__(self, *, available=True, memory_gib=6, bf16=True, name="RTX 4050"):
        self.available = available
        self.memory_gib = memory_gib
        self.bf16 = bf16
        self.name = name

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def get_device_properties(self, _index):
        return SimpleNamespace(
            name=self.name,
            total_memory=self.memory_gib * 2**30,
        )

    def is_bf16_supported(self):
        return self.bf16


def _torch(cuda):
    return SimpleNamespace(cuda=cuda)


def test_cpu_device_plan_is_fp32_without_fast_decode(monkeypatch):
    monkeypatch.delenv("MANIALAB_V32_CPU_THREADS", raising=False)

    plan = select_device_plan(_torch(FakeCuda(available=False)), cpu_count=6)

    assert plan == DevicePlan("cpu", "CPU", None, "fp32", False, 1, 6)


def test_explicit_cpu_overrides_available_cuda():
    plan = select_device_plan(_torch(FakeCuda()), requested="cpu", cpu_threads=4)

    assert (plan.device, plan.precision, plan.fast_decoder_loop, plan.cpu_threads) == (
        "cpu", "fp32", False, 4,
    )


def test_six_gib_cuda_uses_bf16_fast_loop_and_single_safe_batch():
    plan = select_device_plan(_torch(FakeCuda(memory_gib=6)), cpu_threads=6)

    assert plan.device == "cuda"
    assert plan.name == "RTX 4050"
    assert plan.memory_gib == pytest.approx(6)
    assert plan.precision == "bf16"
    assert plan.fast_decoder_loop is True
    assert plan.max_batch_size == 1


def test_engine_production_probe_applies_six_gib_cuda_policy(tmp_path, monkeypatch):
    monkeypatch.delenv("MANIALAB_V32_CPU_THREADS", raising=False)
    engine = V32Engine(tmp_path / "data", tmp_path / "resources")
    engine._probe = lambda refresh=False: {
        "available": True,
        "cuda": True,
        "name": "RTX 4050",
        "memory_gib": 6.0,
        "bf16": True,
    }

    plan = engine._device_plan()

    assert plan.device == "cuda"
    assert plan.name == "RTX 4050"
    assert plan.precision == "bf16"
    assert plan.fast_decoder_loop is True
    assert plan.max_batch_size == 1


def test_engine_device_environment_override_can_force_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIALAB_V32_DEVICE", "cpu")
    engine = V32Engine(tmp_path / "data", tmp_path / "resources")
    engine._probe = lambda refresh=False: {
        "available": True,
        "cuda": True,
        "name": "RTX 4050",
        "memory_gib": 6.0,
        "bf16": True,
    }

    plan = engine._device_plan()

    assert plan.device == "cpu"
    assert plan.precision == "fp32"
    assert plan.fast_decoder_loop is False


def test_status_reports_effective_device_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIALAB_V32_DEVICE", "cpu")
    mapper = tmp_path / "mapper"
    mapper.mkdir()
    (mapper / "inference.py").write_text("# fixture", encoding="utf-8")
    engine = V32Engine(
        tmp_path / "data",
        tmp_path / "resources",
        mapper_repo=mapper,
    )
    engine._probe = lambda refresh=False: {
        "available": True,
        "cuda": True,
        "name": "RTX 4050",
        "memory_gib": 6.0,
        "bf16": True,
    }

    status = engine.status()

    assert status["available"] is True
    assert status["device"]["type"] == "cpu"
    assert status["device"]["precision"] == "fp32"
    assert status["device"]["fast_decoder_loop"] is False


def test_status_marks_unavailable_explicit_cuda_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIALAB_V32_DEVICE", "cuda")
    mapper = tmp_path / "mapper"
    mapper.mkdir()
    (mapper / "inference.py").write_text("# fixture", encoding="utf-8")
    engine = V32Engine(
        tmp_path / "data",
        tmp_path / "resources",
        mapper_repo=mapper,
    )
    engine._probe = lambda refresh=False: {
        "available": True,
        "cuda": False,
    }

    status = engine.status()

    assert status["available"] is False
    assert "没有可用 CUDA" in status["device_override_error"]


def test_small_cuda_without_bf16_uses_fp16():
    plan = select_device_plan(
        _torch(FakeCuda(memory_gib=6, bf16=False)),
        cpu_threads=2,
    )

    assert plan.precision == "fp16"
    assert plan.fast_decoder_loop is True
    assert plan.max_batch_size == 1


def test_cuda_memory_policy_scales_batch_conservatively():
    assert select_device_plan(
        _torch(FakeCuda(memory_gib=10)),
    ).max_batch_size == 2
    assert select_device_plan(
        _torch(FakeCuda(memory_gib=16)),
    ).max_batch_size == 4


def test_explicit_cuda_requires_available_device():
    with pytest.raises(ValueError, match="没有可用 CUDA"):
        select_device_plan(
            _torch(FakeCuda(available=False)),
            requested="cuda",
        )


def test_command_selects_both_model_variants_and_forwards_controls(tmp_path):
    engine = V32Engine(
        tmp_path / "data",
        tmp_path / "resources",
        mapper_repo=tmp_path / "mapper",
        python=tmp_path / "python.exe",
    )
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
    (lora / "adapter_model.safetensors").write_bytes(b"adapter")
    plan = DevicePlan("cuda", "RTX 4050", 6.0, "bf16", True, 1, 6)

    for variant, config in (("v32", "v32"), ("v32-mini", "v32-mini")):
        command = engine.build_command(
            model_variant=variant,
            audio_path=tmp_path / "song.wav",
            reference_chart=tmp_path / "timing.osu",
            output_dir=tmp_path / f"out-{variant}",
            keycount=7,
            difficulty=5.5,
            hold_note_ratio=0.55,
            mania_column_temperature=0.7,
            lora_path=lora,
            device_plan=plan,
            start_ms=1000,
            end_ms=31000,
        )
        joined = " ".join(command)

        assert f"--config-name {config}" in joined
        assert engine.model_directory(variant).as_posix() in joined
        assert "device=cuda" in command
        assert "precision=bf16" in command
        assert "fast_decoder_loop=true" in command
        assert "max_batch_size=1" in command
        assert "gamemode=3" in command
        assert "keycount=7" in command
        assert "difficulty=5.500" in command
        assert "hold_note_ratio=0.5500" in command
        assert "mania_column_temperature=0.700" in command
        assert "start_time=1000" in command
        assert "end_time=31000" in command
        assert "hydra.run.dir=" in " ".join(command)
        assert any(arg.startswith("lora_path=") for arg in command)
        assert MODEL_SPECS[variant]["revision"][:7] in str(engine.model_directory(variant))


def test_command_disables_cuda_fast_loop_for_cpu(tmp_path):
    engine = V32Engine(
        tmp_path / "data",
        tmp_path / "resources",
        mapper_repo=tmp_path / "mapper",
        python=tmp_path / "python.exe",
    )
    command = engine.build_command(
        model_variant="v32",
        audio_path=tmp_path / "song.wav",
        reference_chart=tmp_path / "timing.osu",
        output_dir=tmp_path / "out",
        keycount=10,
        difficulty=5.5,
        hold_note_ratio=0.55,
        device_plan=DevicePlan("cpu", "CPU", None, "fp32", False, 1, 6),
    )

    assert "device=cpu" in command
    assert "precision=fp32" in command
    assert "fast_decoder_loop=false" in command
    assert "max_batch_size=1" in command


@pytest.mark.parametrize(
    "options, message",
    [
        ({"model_variant": "v31"}, "模型版本"),
        ({"generation_strategy": "native"}, "生成策略"),
        ({"torosu_density": "sparse"}, "Torosu 密度"),
        ({"difficulty": 10.1}, "difficulty"),
        ({"hold_note_ratio": 0.86}, "hold_note_ratio"),
        ({"mania_column_temperature": 2.1}, "mania_column_temperature"),
        ({"device": "mps"}, "device"),
    ],
)
def test_generation_options_are_validated(tmp_path, options, message):
    values = {
        "model_variant": "v32",
        "generation_strategy": "7k_to_torosu_10k",
        "torosu_density": "dense",
        "difficulty": 5.5,
        "hold_note_ratio": 0.55,
        "mania_column_temperature": 0.8,
        "lora_path": None,
        "device": "auto",
    }
    values.update(options)

    with pytest.raises(ValueError, match=message):
        V32Engine.validate_options(**values)


def test_lora_path_requires_native_adapter_files(tmp_path):
    lora = tmp_path / "lora"
    lora.mkdir()
    options = {
        "model_variant": "v32",
        "generation_strategy": "native_10k",
        "torosu_density": "none",
        "difficulty": 5.5,
        "hold_note_ratio": 0.55,
        "mania_column_temperature": 0.8,
        "lora_path": lora,
    }

    with pytest.raises(ValueError, match="LoRA 目录缺少"):
        V32Engine.validate_options(**options)

    (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
    (lora / "adapter_model.safetensors").write_bytes(b"adapter")
    assert V32Engine.validate_options(**options)["lora_path"] == lora.resolve()


def test_runtime_environment_uses_configured_roots_without_creating_them(tmp_path, monkeypatch):
    root = tmp_path / "data"
    cache = tmp_path / "cache"
    temporary = tmp_path / "temp"
    engine = V32Engine(root, tmp_path / "resources", python=tmp_path / "missing.exe")

    for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME",
                 "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME", "PIP_CACHE_DIR",
                 "UV_CACHE_DIR", "TMP", "TEMP", "TMPDIR",
                 "MANIALAB_CACHE_ROOT", "MANIALAB_TEMP_ROOT"):
        monkeypatch.delenv(name, raising=False)
    env = engine._runtime_environment()

    assert env["HF_HOME"] == str((root / "cache" / "huggingface").resolve())
    assert env["TORCH_HOME"] == str((root / "cache" / "torch").resolve())
    assert env["TEMP"] == str((root / "cache" / "tmp").resolve())
    assert not (root / "cache").exists()

    monkeypatch.setenv("MANIALAB_CACHE_ROOT", str(cache))
    monkeypatch.setenv("MANIALAB_TEMP_ROOT", str(temporary))
    configured = engine._runtime_environment()
    assert configured["HF_HOME"] == str((cache / "huggingface").resolve())
    assert configured["TEMP"] == str(temporary.resolve())


def test_model_cache_follows_huggingface_environment_root(tmp_path, monkeypatch):
    configured_cache = tmp_path / "large-cache"
    monkeypatch.delenv("MANIALAB_V32_MODEL_CACHE", raising=False)
    monkeypatch.delenv("MANIALAB_CACHE_ROOT", raising=False)
    monkeypatch.setenv("HF_HOME", str(configured_cache / "huggingface"))

    engine = V32Engine(tmp_path / "models", tmp_path / "resources")

    assert engine.model_cache == (configured_cache / "huggingface" / "manialab").resolve()


def test_model_cache_prefers_configured_application_cache_root(tmp_path, monkeypatch):
    configured_cache = tmp_path / "large-cache"
    monkeypatch.delenv("MANIALAB_V32_MODEL_CACHE", raising=False)
    monkeypatch.setenv("MANIALAB_CACHE_ROOT", str(configured_cache))

    engine = V32Engine(tmp_path / "models", tmp_path / "resources")

    assert engine.model_cache == (configured_cache / "huggingface" / "manialab").resolve()


def test_torch_cache_follows_huggingface_cache_when_only_hf_home_is_set(tmp_path, monkeypatch):
    cache_root = tmp_path / "large-cache"
    temp_root = cache_root / "tmp"
    monkeypatch.delenv("MANIALAB_CACHE_ROOT", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.delenv("PIP_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", str(cache_root / "huggingface"))
    monkeypatch.setenv("MANIALAB_TEMP_ROOT", str(temp_root))
    engine = V32Engine(
        tmp_path / "app-data",
        tmp_path / "resources",
        python=tmp_path / "missing.exe",
    )

    env = engine._runtime_environment()

    assert env["HF_HOME"] == str((cache_root / "huggingface").resolve())
    assert env["TORCH_HOME"] == str((cache_root / "torch").resolve())
    assert env["PIP_CACHE_DIR"] == str((cache_root / "pip").resolve())
    assert env["TEMP"] == str(temp_root.resolve())


def test_external_mapperatorinator_source_disables_managed_runtime_install(tmp_path, monkeypatch):
    monkeypatch.delenv("MANIALAB_MAPPER_PYTHON", raising=False)
    engine = V32Engine(
        tmp_path / "data",
        tmp_path / "resources",
        mapper_repo=tmp_path / "external-mapperatorinator",
    )

    with pytest.raises(ValueError, match="外部 V32 运行环境"):
        engine.install_runtime(progress=lambda *_: None, cancelled=lambda: False)
