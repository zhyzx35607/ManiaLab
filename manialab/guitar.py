"""Optional FL guitar transcription with float input and explicit tail policy.

Use only a labelled guitar stem. Model scores are retained as evidence, not
compared with other detectors' probabilities or treated as musical salience.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from math import gcd
from pathlib import Path
import sys
import uuid
import zipfile

import numpy as np
from scipy.signal import resample_poly

from .domain import Note, UNOBSERVED_RELEASES
from .piano import decode, OUTPUTS

MODEL_SHA256 = 'e02ee3b540717bcae07678764847f5e078213be12b1cba98400f091019df73f9'
CHECKPOINT_SHA256 = '50d93dba89bdd3401849bc735614478e83d9f46d21fa3f71d8aca5acc0a52028'
RATE, WINDOW, HOP, FPS = 16000, 160000, 80000, 100
THRESHOLDS = dict(onset_threshold=.3, offset_threshold=.3, frame_threshold=.1)
OBSERVED = frozenset(('frame_decay', 'release_regression'))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _require(value, message):
    if not value:
        raise ValueError('吉他采音校验失败：' + message)


def _signature(identity):
    return json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _validate_heads(heads, samples, sample_rate):
    frames = (samples * FPS + sample_rate - 1) // sample_rate
    _require(set(heads) == set(OUTPUTS) and all(np.asarray(a).shape == (frames, 88)
        and np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() for a in heads.values()),
        '输出帧时钟、数值或形状无效')


def infer(samples, sample_rate, session, progress=lambda p, s: None, cancelled=lambda: False):
    """Fixed 10-second windows, five-second hop, unchanged float waveform gain."""
    pcm = np.asarray(samples, dtype=np.float32)
    _require(pcm.ndim == 1 and len(pcm) > 0 and np.isfinite(pcm).all(), '需要有限单声道采样')
    _require(type(sample_rate) is int and sample_rate > 0, '采样率无效')
    original_samples = len(pcm)
    if sample_rate != RATE:
        common = gcd(sample_rate, RATE)
        pcm = resample_poly(pcm, RATE // common, sample_rate // common).astype(np.float32)
    _require(len(pcm) == (original_samples * RATE + sample_rate - 1) // sample_rate, '重采样时钟无效')
    padded = ((len(pcm) + WINDOW - 1) // WINDOW) * WINDOW
    starts = list(range(0, padded - WINDOW + 1, HOP))
    pieces = {key: [] for key in OUTPUTS}
    for index, start in enumerate(starts):
        if cancelled():
            raise InterruptedError('已取消吉他采音')
        chunk = np.zeros(WINDOW, dtype=np.float32)
        available = pcm[start:start + WINDOW]
        chunk[:len(available)] = available
        values = session.run(list(OUTPUTS), {'audio': chunk[None]})
        _require(len(values) == len(OUTPUTS), '网络输出头不完整')
        for key, value in zip(OUTPUTS, values):
            _require(value.shape == (1, 1001, 88) and np.isfinite(value).all(), '网络输出无效')
            left = 0 if index == 0 else 250
            right = 1001 if len(starts) == 1 else 1000 if index == len(starts) - 1 else 750
            pieces[key].append(value[0, left:right])
        progress(round((index + 1) * 100 / len(starts)), f'吉他起音与持续分析 {index + 1}/{len(starts)}')
    frames = (original_samples * FPS + sample_rate - 1) // sample_rate
    heads = {key: np.concatenate(value)[:frames] for key, value in pieces.items()}
    _validate_heads(heads, original_samples, sample_rate)
    return heads


def candidates(events, *, allow_predicted_releases=False):
    """Keep raw tails in trace; only explicit eligible model tails can make LNs."""
    _require(type(allow_predicted_releases) is bool, '音尾选项必须为布尔值')
    notes, trace = [], []
    for index, event in enumerate(events):
        _require(event['release_reason'] in OBSERVED | UNOBSERVED_RELEASES
            and type(event['release_observed']) is bool
            and event['release_observed'] == (event['release_reason'] in OBSERVED), '音尾来源无效')
        _require(all(np.isfinite(event[k]) for k in ('onset', 'offset', 'onset_score'))
            and 0 <= event['onset'] <= event['offset'] and 0 <= event['onset_score'] <= 1
            and type(event['pitch']) is int and 21 <= event['pitch'] <= 108, '原始事件无效')
        eligible = bool(allow_predicted_releases and event['release_observed'])
        reason = 'model_predicted_offset' if eligible else event['release_reason'] \
            if event['release_reason'] in UNOBSERVED_RELEASES else 'unresolved_or_postprocessed'
        note = Note(start=round(event['onset'] * 1000), end=round(event['offset'] * 1000),
            pitch=event['pitch'], strength=1., source='pitch', voice='guitar', id=f'guitar:fl:{index}',
            release_reason=reason)
        notes.append(note)
        trace.append(dict(id=note.id, raw_event=event, candidate_start_ms=note.start,
            candidate_end_ms=note.end, release_reason=reason,
            model_predicted_release_hypothesis=eligible, authored_ln_verified=False,
            weight_semantics='unit_weight_not_confidence', allow_predicted_releases=allow_predicted_releases))
    return notes, trace


class GuitarDetector:
    """Default off; isolated model, decoder identity and corruption-checked cache."""
    def __init__(self, resources):
        self.model = Path(resources) / 'assets/deep/guitar-fl.onnx'

    def status(self):
        return dict(available=self.model.is_file(), label='吉他专用采音（测试）',
            voice='guitar', default=False, predicted_releases_default=False, cpu=True)

    def analyze(self, samples, sample_rate, cache, stem_identity, *, allow_predicted_releases=False,
                progress=lambda p, s: None, cancelled=lambda: False):
        _require(type(allow_predicted_releases) is bool, '音尾选项必须为布尔值')
        _require(type(sample_rate) is int and sample_rate > 0, '采样率无效')
        pcm = np.asarray(samples, dtype=np.float32)
        _require(pcm.ndim == 1 and np.isfinite(pcm).all() and .1 <= len(pcm) / sample_rate <= 1200,
            '需要0.1至1200秒的有限单声道采样')
        _require(isinstance(stem_identity, str) and stem_identity, '声轨身份缺失')
        if cancelled():
            raise InterruptedError('已取消吉他采音')
        _require(self.model.is_file(), '当前安装包未包含吉他专用模型')
        _require(_sha(self.model) == MODEL_SHA256, '模型文件校验失败')
        host = Path(sys.executable) if getattr(sys, 'frozen', False) else Path(__file__)
        from . import piano
        decoder = host if getattr(sys, 'frozen', False) else Path(piano.__file__)
        identity = dict(revision=1, model_sha256=MODEL_SHA256, checkpoint_sha256=CHECKPOINT_SHA256,
            detector_sha256=_sha(host), decoder_sha256=_sha(decoder), thresholds=THRESHOLDS,
            pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
            sample_rate=sample_rate, samples=len(pcm), stem_identity=stem_identity,
            normalization=False, allow_predicted_releases=allow_predicted_releases,
            runtime={name: importlib.metadata.version(name) for name in ('onnxruntime', 'numpy', 'scipy')},
            device='CPUExecutionProvider', intra_threads=2, inter_threads=1)
        signature = _signature(identity)
        folder = Path(cache) / hashlib.sha256(signature.encode()).hexdigest()
        marker = folder / 'current.json'
        heads, cache_error = None, None
        if marker.is_file():
            try:
                saved = json.loads(marker.read_text(encoding='utf-8'))
                _require(saved['identity'] == identity, '缓存身份不符')
                filename = saved['file']
                _require(isinstance(filename, str) and filename.startswith('activations-')
                    and filename.endswith('.npz') and Path(filename).name == filename
                    and '/' not in filename and '\\' not in filename, '缓存路径无效')
                target = folder / filename
                _require(_sha(target) == saved['sha256'], '缓存内容已改变')
                with np.load(target, allow_pickle=False) as data:
                    _require(str(data['identity']) == signature, '激活缓存身份不符')
                    loaded = {key: data[key] for key in OUTPUTS}
                _validate_heads(loaded, len(pcm), sample_rate)
                heads = loaded
            except (ValueError, KeyError, OSError, EOFError, TypeError, zipfile.BadZipFile):
                cache_error = '旧缓存不完整或校验失败，已重新计算'
        cached = heads is not None
        if heads is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(self.model), sess_options=options, providers=['CPUExecutionProvider'])
            heads = infer(pcm, sample_rate, session, progress, cancelled)
            del session
            _validate_heads(heads, len(pcm), sample_rate)
        if cancelled():
            raise InterruptedError('已取消吉他采音')
        events = decode(heads, **THRESHOLDS)
        notes, trace = candidates(events, allow_predicted_releases=allow_predicted_releases)
        if cancelled():
            raise InterruptedError('已取消吉他采音')
        if not cached:
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / ('activations-' + uuid.uuid4().hex + '.npz')
            with target.open('xb') as stream:
                np.savez_compressed(stream, identity=signature, **heads)
            temporary = folder / ('current-' + uuid.uuid4().hex + '.tmp')
            temporary.write_text(_signature(dict(identity=identity, file=target.name, sha256=_sha(target))), encoding='utf-8')
            if cancelled():
                raise InterruptedError('已取消吉他采音')
            temporary.replace(marker)
        progress(100, '吉他起音与音尾依据已保存')
        if cancelled():
            raise InterruptedError('已取消吉他采音')
        return notes, dict(identity=identity, cached=cached, cache_rejection=cache_error,
            engine='guitar-fl', source_semantics='unit_weight_not_confidence', candidate_trace=trace,
            evidence_cache=str(marker), allow_predicted_releases=allow_predicted_releases,
            raw_release_reasons={reason: sum(e['release_reason'] == reason for e in events)
                for reason in sorted({e['release_reason'] for e in events})},
            predicted_release_hypotheses=sum(t['model_predicted_release_hypothesis'] for t in trace),
            notice='仅用于吉他声轨；候选权重1不代表置信度或响度。模型音尾是持续证据，'
                   '显式启用后才参与长条；重弹截断等未决尾仍保持短键。')
