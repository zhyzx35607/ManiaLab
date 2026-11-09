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
from .stems import PROFILES, profile_for

MODEL_SHA256 = "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"
BEAT_MODEL_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
# This is part of the cache identity: older int16 inputs can clip decoded MP3.
INPUT_PROCESSING = dict(revision=2, codec='pcm_f32le', sample_rate=44100,
                        channels=2, max_seconds=1201)


class Separator:
    def __init__(self, resources, profile='four'):
        self.resources = Path(resources)
        self.profile = profile
        self.spec = profile_for(profile)
        self.executable = self.resources / "assets/deep/ManiaLabAudio/ManiaLabAudio.exe"
        self.model = self.resources / 'assets/deep' / self.spec['filename']
        self.beat_model = self.resources / "assets/deep/beat-this-final0.ckpt"
        self.python = os.getenv("MANIALAB_DEEP_PYTHON")
        if profile == 'four' and os.getenv("MANIALAB_DEEP_MODEL"):
            self.model = Path(os.environ["MANIALAB_DEEP_MODEL"])

    def status(self):
        runtime=bool(self.executable.is_file() or (self.python and Path(self.python).is_file()))
        supported={'four'}
        try:
            supported.update(json.loads((self.resources/'assets/deep/manifest.json').read_text(encoding='utf-8')).get('profiles',[]))
        except (OSError,ValueError):
            pass
        if self.python and Path(self.python).is_file():supported.update(PROFILES)
        return dict(available=bool(runtime and self.profile in supported and self.model.is_file()),
                    model=self.spec['label'], profile=self.profile, voices=list(self.spec['voices']), cpu=True,
                    profiles={key:{'label':spec['label'],'available':runtime and key in supported and (self.resources/'assets/deep'/spec['filename']).is_file()}
                              for key,spec in PROFILES.items()},
                    beat_model=self.beat_model.is_file())

    def separate(self, path, work, progress=lambda p, s: None, cancelled=lambda: False):
        if not self.status()["available"]:
            raise ValueError("所选声部分离组件未就绪，请使用完整 Windows 包，或选择其他分离方案")
        work = Path(work)
        output = work / self.spec['directory']
        manifest = output / "separation.json"
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        stems_cached = False
        if manifest.is_file():
            try:
                cached = json.loads(manifest.read_text(encoding="utf-8"))
                if (cached.get("audio_sha256") == digest
                        and cached.get("model_sha256") == self.spec['sha256']
                        and cached.get('input_processing') == INPUT_PROCESSING
                        and all((output / f"{s}.npy").is_file() for s in self.spec['voices'])):
                    stems_cached = True
                    if not self.beat_model.is_file() or cached.get("beat_evidence", {}).get("model_sha256") == BEAT_MODEL_SHA256:
                        progress(30, "复用所选声部与节拍分析结果")
                        return output, cached | {"cached": True, 'profile':self.profile}
            except (ValueError, OSError):
                pass
        output.mkdir(exist_ok=True)
        wave = work / "separation-input.wav"
        progress(2, "准备"+self.spec['label']+"分析音频")
        result = run_hidden([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-i", str(path),
            "-vn", "-ac", str(INPUT_PROCESSING['channels']), "-ar", str(INPUT_PROCESSING['sample_rate']),
            "-t", str(INPUT_PROCESSING['max_seconds']), "-c:a", INPUT_PROCESSING['codec'], str(wave)],
            capture_output=True, timeout=180)
        if result.returncode:
            raise ValueError("声部分离无法解码音频")
        command = [str(self.python), str(self.resources/'scripts/separate_audio.py')] if self.python else [str(self.executable)]
        command += [str(wave), str(output), "--model", str(self.model)]
        if self.profile != 'four':
            command += ['--profile', self.profile]
        if self.beat_model.is_file():
            command += ["--beat-model", str(self.beat_model)]
            if stems_cached:
                command += ["--beats-only"]
        log = work / "separation.log"
        started = time.monotonic()
        last_message = "正在分离 "+' / '.join(self.spec['voices'])
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=stream,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env=os.environ | {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            try:
                position = 0
                with log.open("r", encoding="utf-8", errors="replace") as reader:
                    while process.poll() is None:
                        if cancelled():
                            raise InterruptedError("已取消声部分离")
                        if time.monotonic() - started > 7200:
                            raise ValueError("声部分离超过两小时，请尝试较短片段")
                        for line in reader:
                            try:
                                row = json.loads(line)
                                position = int(row.get("progress", position))
                                if "chunk" in row:
                                    last_message = f"声部分离 {row['chunk']}/{row['chunks']}"
                                elif position >= 94:
                                    last_message = "检查节拍结构与半速 / 倍速"
                            except (ValueError, TypeError):
                                continue
                        progress(2 + round(position * .28), last_message+f' · 已运行 {int(time.monotonic()-started)} 秒')
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
            raise ValueError(f"声部分离失败，诊断日志：{log}")
        report = json.loads(manifest.read_text(encoding="utf-8"))
        if self.beat_model.is_file() and report.get("beat_evidence", {}).get("model_sha256") != BEAT_MODEL_SHA256:
            raise ValueError("节拍模型没有完成分析，请重新分析或检查本地日志")
        report["audio_sha256"] = digest
        report['profile'] = self.profile
        report['input_processing'] = dict(INPUT_PROCESSING)
        manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return output, report | {"cached": stems_cached}
