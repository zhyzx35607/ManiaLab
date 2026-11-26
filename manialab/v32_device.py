"""Runtime device selection for portable V32 inference."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DevicePlan:
    device: str
    name: str
    memory_gib: float | None
    precision: str
    fast_decoder_loop: bool
    max_batch_size: int
    cpu_threads: int

    def data(self) -> dict:
        return {
            "device": self.device,
            "name": self.name,
            "memory_gib": self.memory_gib,
            "precision": self.precision,
            "fast_decoder_loop": self.fast_decoder_loop,
            "max_batch_size": self.max_batch_size,
            "cpu_threads": self.cpu_threads,
        }


def select_device_plan(
    torch_module=None,
    *,
    requested: str = "auto",
    cpu_threads: int | None = None,
    cpu_count: int | None = None,
) -> DevicePlan:
    """Choose conservative inference settings without importing Torch eagerly."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device 必须为 auto、cpu 或 cuda")
    if cpu_threads is None:
        raw_threads = os.environ.get("MANIALAB_V32_CPU_THREADS", "")
        if raw_threads:
            try:
                cpu_threads = int(raw_threads)
            except ValueError as error:
                raise ValueError("MANIALAB_V32_CPU_THREADS 必须为 1–6") from error
        else:
            cpu_threads = min(6, max(1, cpu_count or os.cpu_count() or 1))
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 6:
        raise ValueError("CPU 推理线程数必须为 1–6")

    cuda_available = bool(
        torch_module is not None
        and getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)()
    )
    if requested == "cuda" and not cuda_available:
        raise ValueError("已显式选择 CUDA，但当前推理 Python 环境没有可用 CUDA")
    use_cuda = cuda_available and requested != "cpu"
    if not use_cuda:
        return DevicePlan("cpu", "CPU", None, "fp32", False, 1, cpu_threads)

    cuda = torch_module.cuda
    try:
        index = int(cuda.current_device())
        properties = cuda.get_device_properties(index)
        memory_gib = float(properties.total_memory) / 2**30
        name = str(properties.name)
        bf16 = bool(cuda.is_bf16_supported())
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        if requested == "cuda":
            raise ValueError(f"无法读取 CUDA 设备能力：{error}") from error
        return DevicePlan("cpu", "CPU (CUDA probe failed)", None, "fp32", False, 1, cpu_threads)

    max_batch_size = 1 if memory_gib <= 8 else 2 if memory_gib <= 12 else 4
    return DevicePlan(
        "cuda",
        name,
        round(memory_gib, 2),
        "bf16" if bf16 else "fp16",
        True,
        max_batch_size,
        cpu_threads,
    )
