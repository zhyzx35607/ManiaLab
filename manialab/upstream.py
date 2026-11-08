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


def fixed_timing_output(text, tempo):
    """Conform newly generated model events to the confirmed chart clock.

    TIMING context conditions the model; it is not a decoder constraint. Keep
    the untouched model file in its run directory, and record every movement.
    Model timestamps already use chart time: never subtract audio_shift again.
    """
    from dataclasses import replace
    from .domain import parse_osu, sections
    from .timing import align_candidates, grid_report
    parsed = parse_osu(text)
    aligned, report = align_candidates(parsed['notes'], tempo, confirmed=True)
    notes = []
    for old, new in zip(parsed['notes'], aligned):
        fields = old.raw.split(',')
        fields[2] = str(new.start)
        if old.hold:
            sample = fields[5].split(':', 1)[1] if ':' in fields[5] else '0:0:0:0:'
            if new.hold:
                fields[5] = f'{new.end}:{sample}'
            else:
                fields[3] = str((int(fields[3]) & ~128) | 1)
                fields[5] = sample
        notes.append(replace(new, raw=','.join(fields)))
    parts = sections(text)
    parts['TimingPoints'] = [f"{tempo['offset']:.3f},{60000 / tempo['bpm']:.8f},4,2,0,70,1,0"]
    parts['Editor'] = [line for line in parts.get('Editor', []) if not line.startswith('BeatDivisor:')]
    parts['Editor'].append(f"BeatDivisor:{tempo.get('division', 4)}")
    parts['HitObjects'] = [n.raw for n in notes]
    content = '\n'.join(['\n'.join(parts.get('', ['osu file format v14']))] +
                        [f'[{name}]\n' + '\n'.join(body) for name, body in parts.items() if name]) + '\n'
    report['before'] = grid_report(parsed['notes'], tempo)
    report['after'] = verify_fixed_timing(content, tempo)
    return content, notes, report


def verify_fixed_timing(content, tempo):
    """Check serialized red line and object times, including raw hitobject rows."""
    from .domain import parse_osu
    from .timing import grid_report
    parsed = parse_osu(content)
    red = [line.split(',') for line in parsed['timing'] if line.strip() and not line.startswith('//')
           and float(line.split(',')[1]) > 0]
    if (len(red) != 1 or abs(float(red[0][0]) - tempo['offset']) > .001 or
            abs(float(red[0][1]) - 60000 / tempo['bpm']) > .00001):
        raise ValueError('原模型输出的红线与确认的 BPM / Offset 不一致，请重新生成')
    report = grid_report(parsed['notes'], dict(bpm=60000 / float(red[0][1]),
                         offset=float(red[0][0]), division=tempo.get('division', 4)))
    if report['off_grid']:
        raise ValueError('原模型输出未通过完整落线检查，请重新生成；原始结果保留在模型运行目录')
    return report


class MapperEngine:
    def __init__(self, root, resources):
        self.root, self.resources = Path(root) / "mapperatorinator", Path(resources)
        self.source = self.root / "source"
        self.python = self.root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def status(self):
        return {"installed": (self.root / "ready.json").is_file() and self.python.is_file(),
                "model_downloaded": (self.root / "model/gamemode=3/model.safetensors").is_file(),
                "source_commit": SOURCE_COMMIT, "model_revision": MODEL_REVISION}

    @staticmethod
    def stop_process(process):
        if process.poll() is not None:
            return
        # The Windows venv launcher and uv create children. Killing only the
        # launcher leaves model inference running after a timeout/cancellation.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=30)
        else:
            process.terminate()
        process.wait(timeout=15)

    def run(self, argv, message, progress, cancelled, percent=10, cwd=None, log_path=None):
        self.root.mkdir(parents=True, exist_ok=True)
        log = Path(log_path) if log_path is not None else self.root / "engine.log"
        log.parent.mkdir(parents=True, exist_ok=True)
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
                        self.stop_process(process)
                        raise InterruptedError("已取消原模型任务")
                    if time.monotonic() - started > 3600:
                        self.stop_process(process)
                        raise ValueError("原模型步骤超过一小时，已停止；请查看 engine.log")
                    progress(percent, f"{message} · 已运行 {int(time.monotonic()-started)} 秒")
                    time.sleep(.6)
            finally:
                self.stop_process(process)
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

    def generate(self, audio, output, settings, progress, cancelled, seconds=30, tempo=None, start_ms=0,
                 timing_reference=None, lora_path=None, difficulty=5.0, conditioning=None,
                 arrangement_reference=None, sketch_codec=None):
        if type(start_ms) is not int or start_ms < 0:
            raise ValueError('原模型片段起点必须是非负整数毫秒')
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not 0 <= seconds <= 1200:
            raise ValueError('原模型片段时长必须在 0–1200 秒之间')
        if tempo is not None and timing_reference is not None:
            raise ValueError('固定节拍与参考谱节拍不能同时指定')
        if arrangement_reference is not None and (tempo is not None or timing_reference is not None):
            raise ValueError('编排参考与独立节拍输入不能同时指定')
        if arrangement_reference is not None and sketch_codec!='map-delimited-sketch-v1':
            raise ValueError('当前 mania 权重不原生支持草稿输入，必须明确选择实验草稿编码')
        if sketch_codec is not None and arrangement_reference is None:
            raise ValueError('实验草稿编码需要对应的草稿输入')
        if not isinstance(difficulty, (int, float)) or not 0 <= difficulty <= 100:
            raise ValueError('原模型难度必须在 0–100 之间')
        if conditioning is not None:
            allowed={'add_mapper_token','add_year_token','add_descriptors','add_song_position_token'}
            if not isinstance(conditioning,dict) or set(conditioning)-allowed or any(type(v) is not bool for v in conditioning.values()):
                raise ValueError('实验训练条件配置无效')
        if lora_path is not None:
            lora_path = Path(lora_path).resolve()
            if not all((lora_path / name).is_file() for name in ('adapter_config.json', 'adapter_model.safetensors')):
                raise ValueError('实验适配器文件不完整')
            if arrangement_reference is not None:
                codec_path=lora_path/'context-codec.json'
                if not codec_path.exists() or json.loads(codec_path.read_text())['codec']!=sketch_codec:
                    raise ValueError('实验适配器没有匹配的草稿输入编码')
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
        argv = [self.python, self.resources/'assets/run_mapper.py', str(os.getpid()), "inference.py", "-cn", "v32", "gamemode=3", f'keycount={settings["keys"]}', f'circle_size={settings["keys"]}',
                "device=cpu", "precision=fp32", "use_server=false", "compile=false", "generate_positions=false",
                "fast_decoder_loop=false", "super_timing_fast_loop=false", "export_osz=false", "max_batch_size=1",
                "auto_select_gamemode_model=false",
                f'hold_note_ratio={settings["ln_ratio"]}', f'seed={settings["seed"]}',
                "audio_path=" + json.dumps(str(decoded.resolve()).replace("\\", "/")),
                "output_path=" + json.dumps(str(output.resolve()).replace("\\", "/")),
                "model_path=" + json.dumps(str((self.root / "model/gamemode=3").resolve()).replace("\\", "/"))]
        if lora_path is not None:
            argv += ['lora_path=' + json.dumps(lora_path.as_posix())]
        if conditioning is not None:
            argv += [f'train.data.{key}={str(value).lower()}' for key,value in sorted(conditioning.items())]
        if arrangement_reference is not None:
            argv += ['--mania-sketch-context='+sketch_codec,
                     'train.data.context_types=[{in:["${context_type:timing}","${context_type:gd}"],out:["${context_type:map}"]}]',
                     'train.data.context_weights=[1]']
        if seconds:
            argv += [f"end_time={start_ms+int(seconds*1000)}"]
        if start_ms:
            # Retain the complete source and global clock: clipping the audio
            # without shifting its reference timing would silently misalign it.
            argv += [f"start_time={start_ms}"]
        if tempo is not None or timing_reference is not None or arrangement_reference is not None:
            # Give the model the already checked chart clock. The upstream
            # generator consumes TIMING context instead of predicting it again.
            from .domain import export_osu, sections
            reference=output/'timing-context.osu'
            content=export_osu([],settings['keys'],self.resources/'vendor/mapperatorinator/template.osu',
                audio=decoded.name,bpm=tempo['bpm'] if tempo else 120,offset=tempo['offset'] if tempo else 0)
            if timing_reference is not None:
                timing=sections(Path(timing_reference).read_text(encoding='utf-8-sig')).get('TimingPoints', [])
                if not any(line.strip() and not line.startswith('//') and float(line.split(',')[1]) > 0 for line in timing):
                    raise ValueError('参考谱没有有效红线')
                # Preserve tempo changes and meter, but never expose reference
                # hit objects or bookmarks to the prediction being evaluated.
                parts=sections(content);parts['TimingPoints']=timing;parts['HitObjects']=[]
                content='\n'.join(['\n'.join(parts.get('', ['osu file format v14']))] +
                    [f'[{name}]\n'+'\n'.join(body) for name,body in parts.items() if name])+'\n'
            if arrangement_reference is not None:
                # Explicitly a different task: consume an onset/source sketch
                # as GD context. Never infer this from a timing-only argument.
                from .domain import parse_osu
                sketch=Path(arrangement_reference).read_text(encoding='utf-8-sig')
                parsed=parse_osu(sketch)
                if parsed['keys']!=settings['keys']:
                    raise ValueError('编排参考与目标键数不一致')
                parts=sections(content)
                parts['TimingPoints']=parsed['timing']
                parts['HitObjects']=[n.raw for n in parsed['notes']]
                content='\n'.join(['\n'.join(parts.get('', ['osu file format v14']))] +
                    [f'[{name}]\n'+'\n'.join(body) for name,body in parts.items() if name])+'\n'
                reference=output/'arrangement-context.osu'
            reference.write_text(content,encoding='utf-8')
            argv += ['beatmap_path='+json.dumps(str(reference.resolve()).replace('\\','/')),
                     'in_context=[TIMING,GD]' if arrangement_reference is not None else 'in_context=[TIMING]',
                     'output_type=[MAP]','super_timing=false',
                     'year=2023',f'difficulty={difficulty}']
        log_text = self.run(argv, "原模型生成（CPU 推理可能较慢）", progress, cancelled, 40, self.source,
                            output/'inference.log')
        if arrangement_reference is not None and 'ManiaLab sketch context active:' not in log_text:
            raise ValueError('无法验证草稿已经进入模型，拒绝将此次结果视为编排输出')
        self.last_warnings = [line.strip() for line in log_text.splitlines() if "Warning:" in line][:30]
        files = sorted((p for p in output.rglob('*.osu') if p.name not in ('timing-context.osu','arrangement-context.osu')),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise ValueError("原模型没有导出谱面，请查看 engine.log")
        return files[0]
