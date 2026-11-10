"""Optional isolated bass/other transcription with verified evidence caches.

The worker owns neural inference. This module only writes unchanged float PCM,
launches the worker, verifies its output and returns physical note evidence.
Unknown raw tails remain evidence and must be made taps by expression policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
from scipy.io import wavfile

from .domain import UNOBSERVED_RELEASES, note_from
from .instrument_bundle import (MODEL_REVISION, MODELS, RUNTIME_FILES, SOURCE_COMMIT, VOICE_MODELS,
                                runtime_identity, sha, verify_model, verify_source)

CACHE_REVISION = 1
MODEL_RATE = 22050
EVIDENCE_FILES = frozenset(('notes.json', 'raw-notes.json', 'window-notes.json',
    'interval-constructions.json', 'candidates.json', 'candidate-trace.json', 'report.json'))
PREDICTED_RELEASE = 'model_predicted_offset'


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('专用采音结果包含重复字段')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('专用采音结果含非有限数值：' + value)

    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique, parse_constant=invalid)


def _require(condition, message):
    if not condition:
        raise ValueError('专用采音校验失败：' + message)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


class InstrumentDetector:
    """Explicit opt-in; independent cache directories never replace BP caches."""
    def __init__(self, resources):
        self.resources = Path(resources).resolve()
        self.assets = self.resources / 'assets/instruments'
        self.executable = self.assets / 'ManiaLabInstrument/ManiaLabInstrument.exe'
        self.python = os.getenv('MANIALAB_INSTRUMENT_PYTHON')
        self.dependencies = os.getenv('MANIALAB_INSTRUMENT_DEPS')

    def status(self):
        if self.python:
            runtime = Path(self.python).is_file() and all((self.resources / name).is_file() for name in RUNTIME_FILES)
            if self.dependencies:
                runtime = runtime and Path(self.dependencies).is_dir()
        else:
            runtime = self.executable.is_file()
            try:
                manifest = _read(self.assets / 'manifest.json')
                runtime = runtime and _digest(manifest.get('worker_sha256')) \
                    and set(manifest.get('worker_sources', {})) == set(RUNTIME_FILES) \
                    and all((self.executable.parent / '_internal/source-evidence' / name).is_file() for name in RUNTIME_FILES)
            except (OSError, ValueError, TypeError, AttributeError):
                runtime = False
        ready = {}
        for voice, model in VOICE_MODELS.items():
            ready[voice] = dict(available=bool(runtime and (self.assets / 'source').is_dir()
                and (self.assets / 'checkpoints' / ('best_model_' + model + '.pth')).is_file()), model=model)
        return dict(available=all(v['available'] for v in ready.values()),
            label='贝斯与其他伴奏专用采音（测试）', voices=list(VOICE_MODELS), models=ready,
            default=False, predicted_releases_default=False, cpu=True)

    def _runtime(self):
        if self.python:
            executable = Path(self.python).resolve()
            _require(executable.is_file(), '开发 Python 不存在')
            worker = self.resources / 'scripts/transcribe_instruments.py'
            sources = runtime_identity(self.resources)
            command = [str(executable), str(worker)]
            fingerprints = {}
            roots = [executable.parent / 'Lib/site-packages', executable.parent.parent / 'Lib/site-packages']
            if self.dependencies:
                dependencies = Path(self.dependencies).resolve()
                _require(dependencies.is_dir(), '开发依赖目录不存在')
                roots.append(dependencies)
                command += ['--dependencies', str(dependencies)]
            for root in sorted(set(roots)):
                if root.is_dir():
                    for name in ('METADATA', 'RECORD'):
                        for path in sorted(root.glob('*.dist-info/' + name)):
                            fingerprints[str(path)] = sha(path)
            for root in (executable.parent, executable.parent.parent):
                config = root / 'pyvenv.cfg'
                if config.is_file():
                    fingerprints[str(config)] = sha(config)
            identity = dict(kind='development', executable=str(executable), executable_sha256=sha(executable),
                worker_sources=sources, dependency_metadata=fingerprints)
        else:
            _require(self.executable.is_file(), '当前安装包未包含专用采音组件')
            manifest = _read(self.assets / 'manifest.json')
            _require(manifest.get('source_commit') == SOURCE_COMMIT and manifest.get('model_revision') == MODEL_REVISION,
                     '便携组件版本不符')
            executable_hash = sha(self.executable)
            _require(manifest.get('worker_sha256') == executable_hash, '便携运行程序哈希不符')
            sources = runtime_identity(self.executable.parent / '_internal/source-evidence')
            _require(manifest.get('worker_sources') == sources, '便携运行源码哈希不符')
            command = [str(self.executable)]
            identity = dict(kind='portable', executable_sha256=executable_hash,
                manifest_sha256=sha(self.assets / 'manifest.json'), worker_sources=sources)
        return command, identity

    def _validate(self, output, input_path, expected, result_hash=None):
        output, input_path = Path(output), Path(input_path)
        result_path = output / 'result.json'
        if result_hash is not None:
            _require(_digest(result_hash) and sha(result_path) == result_hash, '缓存完成标志已改变')
        result = _read(result_path)
        voice, release = expected['voice'], expected['allow_predicted_releases']
        _require(result.get('source_voice') == voice and result.get('model') == expected['model'], '输出声部或模型不符')
        _require(result.get('allow_predicted_releases') is release, '音尾模式不符')
        _require(result.get('source_semantics') == 'unit_weight_not_confidence', '候选权重语义不符')
        _require(result.get('musical_acceptance') is False, '输出不能宣称音乐验收')
        hashes = result.get('evidence_sha256')
        _require(isinstance(hashes, dict) and set(hashes) == EVIDENCE_FILES, '结果证据清单不完整')
        for name, digest in hashes.items():
            _require(_digest(digest) and sha(output / name) == digest, '证据文件已改变：' + name)
        identity = result.get('identity', {})
        wanted = dict(source_commit=SOURCE_COMMIT, source_files=expected['source_files'],
            model_revision=MODEL_REVISION, model=expected['model'], model_sha256=expected['model_sha256'],
            runtime_code_sha256=expected['runtime']['worker_sources'],
            runner_sha256=expected['runtime']['worker_sources']['scripts/transcribe_instruments.py'],
            shared_adapter_sha256=expected['runtime']['worker_sources']['scripts/research_tsumugi_transcription.py'],
            audio=str(input_path.resolve()), audio_sha256=sha(input_path),
            pcm_sha256=expected['pcm_sha256'], original_sample_rate=expected['sample_rate'],
            original_samples=expected['samples'], original_channels=expected['channels'],
            resampled_sample_rate=MODEL_RATE,
            resampled_samples=(expected['samples'] * MODEL_RATE + expected['sample_rate'] - 1) // expected['sample_rate'],
            normalization=False, inference_scope='full_original_audio', stem_filter='all', evidence_revision=2)
        for key, value in wanted.items():
            _require(identity.get(key) == value and type(identity.get(key)) is type(value), '输入或运行身份不符：' + key)
        _require(_digest(identity.get('resampled_pcm_sha256')), '缺少重采样证据')
        _require(_read(output / 'identity.json') == identity, '运行身份文件不一致')
        rate, pcm = wavfile.read(input_path)
        channels = 1 if pcm.ndim == 1 else pcm.shape[1] if pcm.ndim == 2 else 0
        _require(rate == expected['sample_rate'] and len(pcm) == expected['samples']
                 and channels == expected['channels'] and pcm.dtype == np.float32
                 and hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() == expected['pcm_sha256'], '缓存输入 PCM 已改变')
        report = _read(output / 'report.json')
        _require(report.get('identity') == identity and report.get('context_start_seconds') == 0.
            and report.get('context_end_seconds') == expected['samples'] / expected['sample_rate'], '输出原始时钟不符')
        raw_events = _read(output / 'notes.json')
        _require(report.get('notes') == len(raw_events), '原始事件数量不符')
        for name, key in (('notes.json', 'notes_sha256'), ('raw-notes.json', 'raw_notes_sha256'),
                ('window-notes.json', 'window_notes_sha256'), ('interval-constructions.json', 'interval_constructions_sha256')):
            _require(report.get(key) == hashes[name], '模型报告哈希不符：' + name)
        notes = [note_from(value) for value in _read(output / 'candidates.json')]
        _require(type(result.get('candidates')) is int and result['candidates'] == len(notes), '候选数量不符')
        _require(len({n.id for n in notes}) == len(notes), '候选标识重复')
        trace = _read(output / 'candidate-trace.json')
        _require(isinstance(trace, list) and all(isinstance(t, dict) and isinstance(t.get('id'), str) for t in trace), '来源记录无效')
        _require(len({t['id'] for t in trace}) == len(trace), '来源记录标识重复')
        candidate_trace = {t['id']: t for t in trace if t.get('decision') == 'candidate'}
        _require(set(candidate_trace) == {n.id for n in notes}, '候选来源不完整')
        end_ms = round(expected['samples'] * 1000 / expected['sample_rate'])
        for n in notes:
            row = candidate_trace[n.id]
            _require(n.id and n.voice == voice and n.source == 'pitch' and n.lane == -1
                and 0 <= n.pitch <= 127 and n.end <= end_ms and n.strength == 1.
                and not n.phrase and not n.support, '候选声部、时间或权重不符')
            predicted = n.release_reason == PREDICTED_RELEASE
            _require(n.release_reason in UNOBSERVED_RELEASES or (release and predicted), '未允许的音尾声明')
            _require(row.get('source_voice') == voice and row.get('candidate_start_ms') == n.start
                and row.get('candidate_end_ms') == n.end and row.get('release_reason') == n.release_reason
                and row.get('instrument_id') == n.instrument_id and row.get('authored_ln_verified') is False
                and row.get('weight_semantics') == 'unit_weight_not_confidence'
                and row.get('allow_predicted_releases') is release
                and row.get('model_predicted_release_hypothesis') is predicted
                and row.get('raw_event', {}).get('is_drum') is False
                and row.get('raw_event', {}).get('pitch') == n.pitch, '候选来源属性不符')
            _require(not predicted or (n.end > n.start and row.get('final_endpoint_model_supported') is True), '模型音尾缺少来源支持')
        predicted_count = sum(n.release_reason == PREDICTED_RELEASE for n in notes)
        _require(result.get('predicted_release_hypotheses') == predicted_count, '模型音尾计数不符')
        return notes, result

    def _run(self, command, log, progress, cancelled):
        started, position, message, pending = time.monotonic(), 0, '正在载入专用采音模型', b''
        with Path(log).open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                cwd=self.resources, env=os.environ | {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
                    'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
            try:
                with Path(log).open('rb') as reader:
                    while True:
                        if cancelled():
                            raise InterruptedError('已取消专用采音')
                        elapsed = time.monotonic() - started
                        if elapsed > 7200:
                            raise ValueError('专用采音超过两小时，请尝试较短片段')
                        pending += reader.read(65536)  # Regular-file reads never await a worker newline.
                        lines = pending.split(b'\n')
                        pending = lines.pop()
                        if len(pending) > 65536:
                            pending = b''  # Bound malformed/non-JSON diagnostic lines.
                        for line in lines:
                            try:
                                item = json.loads(line.decode('utf-8'))
                                if isinstance(item, dict):
                                    value = item.get('progress')
                                    if isinstance(value, (int, float)) and math.isfinite(value):
                                        position = max(position, min(99, max(0, round(value))))
                                    if isinstance(item.get('message'), str):
                                        message = item['message'][:500]
                            except (ValueError, UnicodeDecodeError, TypeError):
                                continue
                        progress(position, message + f' · 已运行 {int(elapsed)} 秒')
                        if process.poll() is not None:
                            break
                        time.sleep(.25)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
        if process.returncode:
            raise ValueError('专用采音失败，诊断日志：' + str(log))

    def analyze(self, samples, sample_rate, cache, stem_identity, voice,
                allow_predicted_releases=False, progress=lambda p, s: None, cancelled=lambda: False):
        _require(voice in VOICE_MODELS, '仅支持贝斯与其他伴奏声轨')
        _require(type(allow_predicted_releases) is bool, '音尾选项必须明确为布尔值')
        _require(type(sample_rate) is int and sample_rate > 0, '采样率无效')
        pcm = np.asarray(samples, dtype=np.float32)
        channels = 1 if pcm.ndim == 1 else pcm.shape[1] if pcm.ndim == 2 else 0
        _require(channels in (1, 2) and np.isfinite(pcm).all() and .1 <= len(pcm) / sample_rate <= 1200, '需要0.1至1200秒的有限单声道或双声道采样')
        _require(isinstance(stem_identity, str) and stem_identity, '声轨身份缺失')
        if cancelled():
            raise InterruptedError('已取消专用采音')
        progress(0, '核对专用采音组件与原始声轨')
        command, runtime = self._runtime()
        sources = verify_source(self.assets / 'source')
        model = VOICE_MODELS[voice]
        verify_model(self.assets / 'checkpoints', model)
        host_code = Path(sys.executable) if getattr(sys, 'frozen', False) else Path(__file__)
        expected = dict(revision=CACHE_REVISION, detector_sha256=sha(host_code),
            pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
            sample_rate=sample_rate, samples=len(pcm), channels=channels, stem_identity=stem_identity,
            voice=voice, allow_predicted_releases=allow_predicted_releases, model=model,
            model_sha256=MODELS[model], runtime=runtime, source_files=sources)
        cache = Path(cache)
        _require(not cache.exists() or cache.is_dir(), '专用缓存必须使用独立目录')
        folder = cache / _signature(expected)
        marker = folder / 'current.json'
        cache_error = None
        if marker.is_file():
            try:
                saved = _read(marker)
                _require(saved.get('identity') == expected, '缓存身份不符')
                attempt_name = saved.get('attempt')
                _require(isinstance(attempt_name, str) and attempt_name.startswith('attempt-')
                    and Path(attempt_name).name == attempt_name and '/' not in attempt_name and '\\' not in attempt_name, '缓存路径无效')
                attempt = folder / attempt_name
                _require(attempt.resolve().is_relative_to(folder.resolve()), '缓存路径越界')
                _require(_digest(saved.get('result_sha256')), '缓存缺少完成标志哈希')
                notes, result = self._validate(attempt / 'output', attempt / 'input.wav', expected, saved.get('result_sha256'))
                if cancelled():
                    raise InterruptedError('已取消专用采音')
                progress(100, '已核对并复用专用采音证据')
                return notes, self._diagnostics(expected, result, attempt, cached=True)
            except InterruptedError:
                raise
            except (ValueError, OSError, KeyError, TypeError) as error:
                cache_error = str(error)
        if cancelled():
            raise InterruptedError('已取消专用采音')
        attempt = folder / ('attempt-' + uuid.uuid4().hex)
        attempt.mkdir(parents=True)
        wave, output = attempt / 'input.wav', attempt / 'output'
        wavfile.write(wave, sample_rate, pcm)
        command += ['--audio', str(wave.resolve()), '--voice', voice, '--assets', str(self.assets), '--output', str(output.resolve())]
        if allow_predicted_releases:
            command.append('--allow-predicted-releases')
        _json_new(attempt / 'request.json', expected)
        self._run(command, attempt / 'worker.log', progress, cancelled)
        if cancelled():
            raise InterruptedError('已取消专用采音')
        try:
            notes, result = self._validate(output, wave, expected)
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise ValueError('专用采音未产生完整、有效的结果；诊断目录：' + str(attempt)) from error
        if cancelled():
            raise InterruptedError('已取消专用采音')
        temporary = folder / ('current-' + uuid.uuid4().hex + '.tmp')
        _json_new(temporary, dict(identity=expected, attempt=attempt.name, result_sha256=sha(output / 'result.json')))
        temporary.replace(marker)
        progress(100, '专用采音完成，起音与模型音尾证据已保存')
        return notes, self._diagnostics(expected, result, attempt, cached=False, cache_error=cache_error)

    @staticmethod
    def _diagnostics(identity, result, attempt, *, cached, cache_error=None):
        return dict(identity=identity, cached=cached, engine='tsumugi', evidence_directory=str(attempt / 'output'),
            result_sha256=sha(attempt / 'output/result.json'), source_semantics='unit_weight_not_confidence',
            allow_predicted_releases=identity['allow_predicted_releases'],
            predicted_release_hypotheses=result['predicted_release_hypotheses'], cache_rejection=cache_error,
            notice='模型类别是预测音色，权重1不是置信度。未决音尾保留为证据且不能自动生成长条；'
                   '显式启用的模型音尾仍是假设，需要试听确认放手。')
