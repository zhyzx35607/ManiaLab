"""Launch the isolated, portable CPU music separation worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg

from .audio import run_hidden

MODEL_SHA256 = "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"
BEAT_MODEL_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"


class Separator:
    def __init__(self, resources):
        self.resources = Path(resources)
        self.executable = self.resources / "assets/deep/ManiaLabAudio/ManiaLabAudio.exe"
        self.model = self.resources / "assets/deep/hdemucs_high_trained.pt"
        self.beat_model = self.resources / "assets/deep/beat-this-final0.ckpt"
        self.python = os.getenv("MANIALAB_DEEP_PYTHON")
        if os.getenv("MANIALAB_DEEP_MODEL"):
            self.model = Path(os.environ["MANIALAB_DEEP_MODEL"])

    def status(self):
        return dict(available=bool((self.executable.is_file() or (self.python and Path(self.python).is_file())) and self.model.is_file()),
                    model="Hybrid Demucs · drums / bass / other / vocals", cpu=True,
                    beat_model=self.beat_model.is_file())

    def separate(self, path, work, progress=lambda p, s: None, cancelled=lambda: False):
        if not self.status()["available"]:
            raise ValueError("深度采音组件未就绪，请使用包含四声部模型的完整 Windows 包，或选择快速采音")
        work = Path(work)
        output = work / "stems"
        manifest = output / "separation.json"
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        stems_cached = False
        if manifest.is_file():
            try:
                cached = json.loads(manifest.read_text(encoding="utf-8"))
                if cached.get("audio_sha256") == digest and cached.get("model_sha256") == MODEL_SHA256 and all((output / f"{s}.npy").is_file() for s in ("drums", "bass", "other", "vocals")):
                    stems_cached = True
                    if not self.beat_model.is_file() or cached.get("beat_evidence", {}).get("model_sha256") == BEAT_MODEL_SHA256:
                        progress(30, "复用四声部与节拍分析结果")
                        return output, cached | {"cached": True}
            except (ValueError, OSError):
                pass
        output.mkdir(exist_ok=True)
        wave = work / "separation-input.wav"
        progress(2, "准备四声部分析音频")
        result = run_hidden([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-i", str(path),
            "-vn", "-ac", "2", "-ar", "44100", "-t", "1201", "-c:a", "pcm_s16le", str(wave)],
            capture_output=True, timeout=180)
        if result.returncode:
            raise ValueError("四声部分析无法解码音频")
        command = [str(self.executable)] if self.executable.is_file() else [str(self.python), str(self.resources / "scripts/separate_audio.py")]
        command += [str(wave), str(output), "--model", str(self.model)]
        if self.beat_model.is_file():
            command += ["--beat-model", str(self.beat_model)]
            if stems_cached:
                command += ["--beats-only"]
        log = work / "separation.log"
        started = time.monotonic()
        last_message = "正在分离鼓、贝斯、人声与伴奏"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=stream,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env=os.environ | {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            try:
                position = 0
                with log.open("r", encoding="utf-8", errors="replace") as reader:
                    while process.poll() is None:
                        if cancelled():
                            raise InterruptedError("已取消四声部分析")
                        if time.monotonic() - started > 7200:
                            raise ValueError("四声部分离超过两小时，请尝试较短片段")
                        for line in reader:
                            try:
                                row = json.loads(line)
                                position = int(row.get("progress", position))
                                if "chunk" in row:
                                    last_message = f"四声部分离 {row['chunk']}/{row['chunks']} · 已运行 {int(time.monotonic() - started)} 秒"
                                elif position >= 94:
                                    last_message = "检查节拍结构与半速 / 倍速"
                            except (ValueError, TypeError):
                                continue
                        progress(2 + round(position * .28), last_message)
                        time.sleep(.25)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=10)
                wave.unlink(missing_ok=True)
        if process.returncode or not manifest.is_file():
            raise ValueError(f"四声部分离失败，诊断日志：{log}")
        report = json.loads(manifest.read_text(encoding="utf-8"))
        if self.beat_model.is_file() and report.get("beat_evidence", {}).get("model_sha256") != BEAT_MODEL_SHA256:
            raise ValueError("节拍模型没有完成分析，请重新分析或检查本地日志")
        report["audio_sha256"] = digest
        manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return output, report | {"cached": stems_cached}
