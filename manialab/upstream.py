"""Optional, isolated Mapperatorinator v32 CPU backend. Baseline needs no setup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

SOURCE_COMMIT = "5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa"
MODEL_REVISION = "74f22583400d259bf424819e11027c17933efe54"


class MapperEngine:
    def __init__(self, root, resources):
        self.root, self.resources = Path(root) / "mapperatorinator", Path(resources)
        self.source = self.root / "source"
        self.python = self.root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def status(self):
        return {"installed": (self.root / "ready.json").is_file() and self.python.is_file(),
                "model_downloaded": (self.root / "model/gamemode=3/model.safetensors").is_file(),
                "source_commit": SOURCE_COMMIT, "model_revision": MODEL_REVISION}

    def run(self, argv, message, progress, cancelled, percent=10, cwd=None):
        self.root.mkdir(parents=True, exist_ok=True)
        log = self.root / "engine.log"
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1",
                   TOKENIZERS_PARALLELISM="false", UV_LINK_MODE="copy", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        with log.open("a", encoding="utf-8") as out:
            out.write("\n" + message + "\n");out.flush()
            log_start = log.stat().st_size
            process = subprocess.Popen([str(x) for x in argv], cwd=cwd, env=env, stdout=out, stderr=out,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if cancelled():
                        # uv and Python may have child processes; stop this process tree only.
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                        else:
                            process.terminate()
                        process.wait(timeout=15)
                        raise InterruptedError("已取消原模型任务")
                    if time.monotonic() - started > 3600:
                        process.kill()
                        raise ValueError("原模型步骤超过一小时，已停止；请查看 engine.log")
                    progress(percent, f"{message} · 已运行 {int(time.monotonic()-started)} 秒")
                    time.sleep(.6)
            finally:
                if process.poll() is None:
                    process.kill()
            if process.returncode:
                raise ValueError(f"{message}失败（退出码 {process.returncode}），日志：{log}")
        return log.read_bytes()[log_start:].decode("utf-8", errors="replace")

    def install(self, progress, cancelled):
        uv = self.resources / "assets/runtime/uv.exe"
        if not uv.is_file():
            found = shutil.which("uv")
            if not found:
                raise ValueError("缺少 uv 运行时，请使用完整的 Windows Release 包")
            uv = Path(found)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source.mkdir(exist_ok=True)
        with zipfile.ZipFile(self.resources / "assets/mapperatorinator-source.zip") as archive:
            archive.extractall(self.source)  # trusted, pinned asset shipped with this program
        if not self.python.is_file():
            self.run([uv, "venv", str(self.python.parent.parent), "--python", "3.12"], "准备独立 Python 环境", progress, cancelled, 5)
        self.run([uv, "pip", "install", "--python", self.python, "torch==2.9.1", "torchaudio==2.9.1", "--index-url", "https://download.pytorch.org/whl/cpu"],
                 "下载并安装 CPU PyTorch", progress, cancelled, 25)
        self.run([uv, "pip", "install", "--python", self.python, "-r", self.resources / "assets/mapperatorinator-requirements.txt"],
                 "安装原模型依赖", progress, cancelled, 55)
        self.run([self.python, "inference.py", "--help"], "检查原模型入口", progress, cancelled, 90, self.source)
        (self.root / "ready.json").write_text(json.dumps(self.status() | {"source_commit": SOURCE_COMMIT}), encoding="utf-8")

    def generate(self, audio, output, settings, progress, cancelled, seconds=30):
        if not self.status()["installed"]:
            raise ValueError("请先安装可选原模型引擎")
        if not self.status()["model_downloaded"]:
            self.run([self.python, self.resources / "assets/download_mapper.py", self.root / "model", MODEL_REVISION],
                     "下载固定版本的 mania 模型权重", progress, cancelled, 10)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        # pydub's compressed-file loader needs an external ffprobe, which is
        # absent on clean Windows installs. Decode with our bundled FFmpeg;
        # pydub reads PCM WAV directly, preserving the analysis time base.
        from .audio import run_hidden
        import imageio_ffmpeg
        decoded = output / "ManiaLab-source.wav"
        decode_result = run_hidden([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-i", str(Path(audio).resolve()),
            "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(decoded.resolve())], capture_output=True, timeout=180)
        if decode_result.returncode:
            raise ValueError("原模型输入音频无法解码")
        argv = [self.python, "inference.py", "-cn", "v32", "gamemode=3", f'keycount={settings["keys"]}', f'circle_size={settings["keys"]}',
                "device=cpu", "precision=fp32", "use_server=false", "compile=false", "generate_positions=false",
                "fast_decoder_loop=false", "super_timing_fast_loop=false", "export_osz=false", "max_batch_size=1",
                "auto_select_gamemode_model=false",
                f'hold_note_ratio={settings["ln_ratio"]}', f'seed={settings["seed"]}',
                "audio_path=" + json.dumps(str(decoded.resolve()).replace("\\", "/")),
                "output_path=" + json.dumps(str(output.resolve()).replace("\\", "/")),
                "model_path=" + json.dumps(str((self.root / "model/gamemode=3").resolve()).replace("\\", "/"))]
        if seconds:
            argv += [f"end_time={int(seconds*1000)}"]
        log_text = self.run(argv, "原模型生成（CPU 推理可能较慢）", progress, cancelled, 40, self.source)
        self.last_warnings = [line.strip() for line in log_text.splitlines() if "Warning:" in line][:30]
        files = sorted(output.rglob("*.osu"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise ValueError("原模型没有导出谱面，请查看 engine.log")
        return files[0]
