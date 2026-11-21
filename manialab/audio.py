"""CPU audio evidence. Basic Pitch model/inference convention: Spotify, Apache-2.0.

Window timestamps are reconstructed from actual sample offsets, not a rounded FPS.
The detector is not a source separator and its activations are not calibrated probabilities.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from dataclasses import replace
import tempfile
import threading
import time
import zipfile
import zlib

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

from .domain import Note
from .timing import estimate_timing, cover_missing_timing_windows
from .audio_config import FRAME, OVERLAP, SR, WINDOW
from .audio_io import decode as decode_audio, run_hidden
from .audio_analysis import pitched_notes, rhythmic_evidence, fuse_attacks, refine_releases
from .audio_evidence import (
    align_stem_samples,
    reject_source_bleed,
    source_evidence,
    timing_evidence_quality,
)
from .audio_cache import (
    BASIC_PITCH_INFERENCE_REVISION,
    WHOLE_AUDIO_ACTIVATION_CACHE_VERSION,
    load_stem_activations,
    model_identity,
    save_stem_activations,
)

_session = None
_session_model_identity = None
_lock = threading.Lock()
_SNAPSHOT_CHUNK_SIZE = 1024 * 1024


def decode(path, work, cancelled=None):
    return decode_audio(path, work, runner=run_hidden, cancelled=cancelled)


@contextmanager
def _audio_snapshot(path, work, cancelled):
    """Hash and decode one private copy of the source, even if its path changes."""
    source = Path(path)
    snapshot = None
    try:
        digest = hashlib.sha256()
        with source.open("rb") as original:
            with tempfile.NamedTemporaryFile(
                    prefix=".analysis-input-", suffix=source.suffix,
                    dir=work, delete=False) as copied:
                snapshot = Path(copied.name)
                while True:
                    if cancelled():
                        raise InterruptedError("已取消音频快照")
                    chunk = original.read(_SNAPSHOT_CHUNK_SIZE)
                    if not chunk:
                        break
                    copied.write(chunk)
                    digest.update(chunk)
        if cancelled():
            raise InterruptedError("已取消音频快照")
        yield snapshot, digest.hexdigest()
    except BaseException as error:
        if snapshot is not None:
            try:
                snapshot.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(
                    f"分析音频快照无法删除（路径：{snapshot}）：{cleanup_error!r}")
        raise
    else:
        if snapshot is not None:
            try:
                snapshot.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_error.add_note(f"分析音频快照无法删除（路径：{snapshot}）")
                raise


def infer(samples, model, progress=lambda p, s: None, cancelled=lambda: False,
          model_identity_value=None):
    global _session, _session_model_identity
    if model_identity_value is None:
        model_identity_value = model_identity(model, cancelled)
    with _lock:
        if _session is None or _session_model_identity != model_identity_value:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"])
            _session = session
            _session_model_identity = model_identity_value
        session = _session
    padded = np.pad(samples, (OVERLAP // 2, 0))
    hop = WINDOW - OVERLAP
    frames, onsets, times = [], [], []
    positions = list(range(0, len(padded), hop))
    for index, pos in enumerate(positions):
        if cancelled():
            raise InterruptedError("已取消分析")
        chunk = np.pad(padded[pos:pos + WINDOW], (0, max(0, pos + WINDOW - len(padded))))
        output = session.run(["StatefulPartitionedCall:1", "StatefulPartitionedCall:2"],
                             {"serving_default_input_2:0": chunk.astype(np.float32)[None, :, None]})
        frames.append(output[0][0, 15:-15])
        onsets.append(output[1][0, 15:-15])
        times.extend((pos + np.arange(142) * FRAME) / SR)
        progress(8 + int(77 * (index + 1) / len(positions)), f"音高 / 持续音分析 {index + 1}/{len(positions)}")
    times = np.asarray(times)
    valid = times < len(samples) / SR
    return np.concatenate(frames)[valid], np.concatenate(onsets)[valid], times[valid]


def analyze(path, work, model, settings, progress=lambda p, s: None, cancelled=lambda: False):
    started = time.perf_counter()
    progress(2, "解码音频")
    with _audio_snapshot(path, work, cancelled) as (snapshot, audio_hash):
        samples = decode(snapshot, work, cancelled=cancelled)
    duration = len(samples) / SR
    cache = Path(work) / "activations.npz"
    model_fingerprint = model_identity(model, cancelled)
    cached = None
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as data:
                if (int(data["version"]) == WHOLE_AUDIO_ACTIVATION_CACHE_VERSION
                        and str(data["audio_sha256"]) == audio_hash
                        and str(data["model_identity"]) == model_fingerprint
                        and int(data["inference_revision"]) == BASIC_PITCH_INFERENCE_REVISION):
                    cached = (data["frames"], data["onsets"], data["times"])
        except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile, zlib.error):
            cached = None
    if cached is not None:
        frames, onsets, times = cached
        progress(85, "复用已分析的音频，重新提取候选")
    else:
        frames, onsets, times = infer(
            samples, model, progress, cancelled, model_fingerprint)
        temporary = None
        try:
            if cancelled():
                raise InterruptedError("已取消分析")
            with tempfile.NamedTemporaryFile(
                    prefix=f".{cache.name}.", suffix=".pending.npz",
                    dir=cache.parent, delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(
                    stream, frames=frames, onsets=onsets, times=times,
                    version=WHOLE_AUDIO_ACTIVATION_CACHE_VERSION, audio_sha256=audio_hash,
                    model_identity=model_fingerprint,
                    inference_revision=BASIC_PITCH_INFERENCE_REVISION)
            if cancelled():
                raise InterruptedError("已取消分析")
            temporary.replace(cache)
            temporary = None
        except BaseException as error:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    error.add_note(
                        f"激活缓存暂存文件无法删除（路径：{temporary}）：{cleanup_error!r}")
            raise
    if cancelled():
        raise InterruptedError("已取消分析")
    progress(90, "校准瞬态、比较 BPM 候选并检查全曲漂移")
    transients, tempo = rhythmic_evidence(samples, cancelled=cancelled)
    if cancelled():
        raise InterruptedError("已取消分析")
    notes = pitched_notes(frames, onsets, times, duration, settings["onset_threshold"], settings["frame_threshold"], transients)
    notes, releases = refine_releases(notes, samples)
    notes, unmatched, fused = fuse_attacks(notes, transients)
    chunks = np.array_split(np.abs(samples), min(2400, len(samples)))
    waveform = [round(float(np.max(x)), 4) for x in chunks]
    candidates = sorted(notes + unmatched, key=lambda n: (n.start, n.pitch))
    result = {"revision": 8, "duration": duration, "audio_sha256": audio_hash,
            "waveform": waveform, "tempo": tempo,
            "elapsed_seconds": round(time.perf_counter() - started, 2), "cached": cached is not None,
            "timing_attacks": [n.data() for n in transients], "fused_attacks": fused,
            "release_refinement": releases,
            "candidates": [n.data() for n in candidates], "thresholds": {k: settings[k] for k in ("onset_threshold", "frame_threshold")},
            "engine": "Basic Pitch 0.4.0 ONNX + onset refinement + phase regression", "notice": "音高候选并非乐器分轨；复杂混音、混响和音尾仍可能误判。"}
    from .structure import detect_structure
    progress(97, '比较音色与声部变化，建议音乐段落')
    result['structure'] = detect_structure(result, samples, SR, cancelled=cancelled)
    result['segments'] = result['structure']['segments']
    result['elapsed_seconds'] = round(time.perf_counter()-started, 2)
    progress(100, "采音候选与段落已准备好")
    return result


def analyze_deep(path, work, model, settings, separator, progress=lambda p, s: None, cancelled=lambda: False,
                 *, piano_detector=None, instrument_detector=None, allow_predicted_releases=False,
                 guitar_detector=None, allow_guitar_predicted_releases=False):
    """Separate sources, then run pitch/attack analysis on each isolated voice."""
    from .voices import vocal_phrases
    from .stems import archive_review_stems,separation_voices,separation_label,separation_identity
    if type(allow_predicted_releases) is not bool or (allow_predicted_releases and instrument_detector is None):
        raise ValueError('模型音尾选项需要显式选择专用采音')
    if type(allow_guitar_predicted_releases) is not bool or (allow_guitar_predicted_releases and guitar_detector is None):
        raise ValueError('吉他模型音尾选项需要显式选择吉他专用采音')
    started = time.perf_counter()
    with _audio_snapshot(path, work, cancelled) as (snapshot, audio_hash):
        samples = decode(snapshot, work, cancelled=cancelled)
        duration = len(samples) / SR
        stem_root, separation = separator.separate(snapshot, work, progress, cancelled)
        if separation.get('audio_sha256') != audio_hash:
            raise ValueError('声部来源摘要与音频快照不匹配，请重新分析')
    stem_identity = separation_identity(separation)
    beats_ms = separation.get("beat_evidence", {}).get("beats_ms")
    voices = separation_voices(separation)
    if piano_detector is not None and 'piano' not in voices:
        raise ValueError('钢琴专用采音需要六声部分离或已导入的钢琴声轨')
    if guitar_detector is not None and 'guitar' not in voices:
        raise ValueError('吉他专用采音需要六声部分离或已导入的吉他声轨')
    instrument_voices = set(voices).intersection(('bass','other')) if instrument_detector is not None else set()
    if instrument_detector is not None and not instrument_voices:
        raise ValueError('专用采音需要分离或导入的贝斯、其他伴奏声轨')
    unit_weight_voices = instrument_voices | ({'guitar'} if guitar_detector is not None else set())
    pitch_cached = []
    all_candidates, phrases, summaries, drum_attacks, tempos = [], [], [], [], {}
    basic_pitch_model_identity = None
    for index, voice in enumerate(voices):
        if cancelled():
            raise InterruptedError("已取消声部分析")
        stem = np.load(stem_root / f"{voice}.npy", allow_pickle=False)
        stem, end_padding = align_stem_samples(stem, len(samples))
        stage = 31 + round(index * 61 / len(voices))
        progress(stage, f"分析 {voice} 的起音与持续区间")
        attacks, _ = rhythmic_evidence(stem, estimate=False, cancelled=cancelled)
        if cancelled():
            raise InterruptedError("已取消声部分析")
        attacks, rejected_attacks = source_evidence(attacks, stem, samples)
        if voice == "drums":
            if cancelled():
                raise InterruptedError("已取消声部分析")
            tempos[voice] = estimate_timing(attacks, duration, beats_ms)
            if cancelled():
                raise InterruptedError("已取消声部分析")
        attacks = [replace(n, voice=voice, id=f"{voice}:{n.id}") for n in attacks]
        cached = False
        articulation = []
        specialized = None
        instrument_diagnostics = None
        guitar_diagnostics = None
        if voice == "drums":
            drum_attacks = attacks
            candidates = attacks
        elif voice == 'guitar' and guitar_detector is not None:
            pitched,guitar_diagnostics=guitar_detector.analyze(stem,SR,stem_root/'guitar-fl',stem_identity,
                allow_predicted_releases=allow_guitar_predicted_releases,
                progress=lambda p,s:progress(stage+round(p*.60/len(voices)),s),cancelled=cancelled)
            cached=guitar_diagnostics['cached'];pitch_cached.append(cached)
            before_source={n.id for n in pitched}
            pitched,rejected_pitch=source_evidence(pitched,stem,samples)
            guitar_diagnostics=dict(guitar_diagnostics,source_energy_filter=dict(
                rejected=rejected_pitch,rejected_ids=sorted(before_source-{n.id for n in pitched})),
                generic_release_refinement=False,generic_attack_fusion=False,
                confidence_based_bleed_filter=False)
            releases=guitar_diagnostics
            candidates=pitched
        elif voice in instrument_voices:
            pitched,instrument_diagnostics=instrument_detector.analyze(stem,SR,stem_root/('instrument-'+voice),
                stem_identity,voice,allow_predicted_releases=allow_predicted_releases,
                progress=lambda p,s:progress(stage+round(p*.60/len(voices)),s),cancelled=cancelled)
            cached=instrument_diagnostics['cached'];pitch_cached.append(cached)
            before_source={n.id for n in pitched}
            pitched,rejected_pitch=source_evidence(pitched,stem,samples)
            instrument_diagnostics=dict(instrument_diagnostics,source_energy_filter=dict(
                rejected=rejected_pitch,rejected_ids=sorted(before_source-{n.id for n in pitched})),
                generic_release_refinement=False,generic_attack_fusion=False,
                confidence_based_bleed_filter=False)
            # Unit event weights are not BP activation confidences. Preserve
            # model clocks/tails and keep unknown tails marked for tap policy.
            releases=instrument_diagnostics
            candidates=pitched
        elif voice == 'piano' and piano_detector is not None:
            pitched,specialized=piano_detector.analyze(stem,SR,stem_root/'piano-specialist-activations.npz',
                stem_identity,lambda p,s:progress(stage+round(p*.60/len(voices)),s),cancelled)
            cached=specialized['cached'];pitch_cached.append(cached)
            pitched,rejected_pitch=source_evidence(pitched,stem,samples)
            # The specialist already estimates onset timing. Generic transient
            # fusion would shift these events without a validated benefit.
            releases=specialized
            candidates=[replace(n,voice=voice,id=f'{voice}:{n.id}') for n in pitched]
        else:
            cache = stem_root / f"{voice}-activations.npz"
            if basic_pitch_model_identity is None:
                basic_pitch_model_identity = model_identity(model, cancelled)
            data = load_stem_activations(
                cache, separation["audio_sha256"], stem_identity,
                basic_pitch_model_identity, BASIC_PITCH_INFERENCE_REVISION)
            if data is None:
                data = infer(
                    stem, model,
                    lambda p, s: progress(stage + round(p * .60 / len(voices)), f"{voice} · {s}"),
                    cancelled, basic_pitch_model_identity)
                save_stem_activations(
                    cache, data, separation['audio_sha256'], stem_identity,
                    basic_pitch_model_identity, BASIC_PITCH_INFERENCE_REVISION,
                    cancelled=cancelled)
            else:
                cached = True
            pitch_cached.append(cached)
            frames, onsets, times = data
            if voice == 'bass':
                from .articulation import steady_bass_onsets
                articulation = steady_bass_onsets(frames, onsets, times, stem, SR,
                    settings['onset_threshold'], settings['frame_threshold'])
            pitched = pitched_notes(frames, onsets, times, duration, settings["onset_threshold"], settings["frame_threshold"], attacks,
                                    suppressed=articulation)
            pitched, rejected_pitch = source_evidence(pitched, stem, samples)
            pitched, releases = refine_releases(pitched, stem)
            pitched, unmatched, _ = fuse_attacks(pitched, attacks)
            pitched = [replace(n, voice=voice, id=f"{voice}:{n.id}") for n in pitched]
            # Rhythm transients come from drums; unmatched vocal/other energy
            # changes are too often pitch/reverb artifacts to duplicate as taps.
            candidates = pitched
            if voice == "vocals":
                phrases = vocal_phrases(frames, times, stem, pitched)
        all_candidates.extend(candidates)
        summaries.append(dict(voice=voice, candidates=len(candidates), attacks=len(attacks), cached=cached,
                              end_padding_samples=end_padding,
                              rejected_low_energy_attacks=rejected_attacks,
                              rejected_low_energy_pitch=0 if voice == "drums" else rejected_pitch,
                              articulation_rejections=articulation,
                              release_refinement=None if voice == "drums" else releases,
                              detector='guitar-fl' if guitar_diagnostics is not None else 'tsumugi' if instrument_diagnostics is not None else 'piano-specialist' if specialized is not None else 'transient' if voice=='drums' else 'basic-pitch',
                              sustains_over_800ms=sum(n.end - n.start >= 800 for n in candidates)))
        if separation.get('source')!='provided':
            peak = max(float(np.max(np.abs(stem))), 1.)
            wavfile.write(stem_root / f"{voice}.wav", SR, (np.clip(stem / peak, -1, 1) * 32767).astype(np.int16))
    progress(94, "核对声部与混音拍线，保留持续段证据")
    mix_attacks, mix_tempo = rhythmic_evidence(samples, beats_ms, cancelled=cancelled)
    if cancelled():
        raise InterruptedError("已取消声部分析")
    stem_signals = {v:np.load(stem_root/f'{v}.npy',allow_pickle=False,mmap_mode='r') for v in voices if v!='drums'}
    protected=[n for n in all_candidates if n.voice in unit_weight_voices]
    generic=[n for n in all_candidates if n.voice not in unit_weight_voices]
    generic, bleed_rejections = reject_source_bleed(generic,stem_signals,mix_attacks)
    all_candidates=generic+protected
    for summary in summaries:
        summary['rejected_cross_source_bleed'] = sum(r['note']['voice']==summary['voice'] for r in bleed_rejections)
        summary['cross_source_bleed_checked'] = summary['voice'] not in unit_weight_voices
        summary['candidates'] = sum(n.voice==summary['voice'] for n in all_candidates)
    separation = archive_review_stems(work, stem_root, separation, voices)
    drum_tempo = tempos.get("drums",mix_tempo)
    drum_quality, mix_quality = timing_evidence_quality(drum_tempo), timing_evidence_quality(mix_tempo)
    use_drums = len(drum_attacks) >= 24 and drum_quality.get("coverage", 0) >= .7 and drum_quality.get("score", 0) >= mix_quality.get("score", 0) * .9
    tempo = dict(drum_tempo if use_drums else mix_tempo)
    attacks = drum_attacks if use_drums else mix_attacks
    tempo["evidence_source"] = "drums" if use_drums else "mix"
    tempo["comparisons"] = ({"drums": drum_tempo, "mix": mix_tempo} if 'drums' in tempos else {"mix":mix_tempo})
    if use_drums:
        tempo, attacks = cover_missing_timing_windows(tempo,drum_attacks,mix_attacks)
    candidates = sorted(all_candidates, key=lambda n: (n.start, n.pitch, n.voice))
    result = dict(revision=14, duration=duration, audio_sha256=audio_hash, tempo=tempo,
        waveform=[round(float(np.max(x)), 4) for x in np.array_split(np.abs(samples), min(2400, len(samples)))],
        elapsed_seconds=round(time.perf_counter() - started, 2), cached=separation["cached"] and all(pitch_cached),
        timing_attacks=[n.data() for n in attacks], fused_attacks=0, candidates=[n.data() for n in candidates],
        phrase_candidates=[n.data() for n in phrases], voices=summaries, separation=separation,
        source_bleed_rejections=bleed_rejections,
        source_bleed_policy=dict(excluded_unit_weight_voices=sorted(unit_weight_voices),
            reason='Unit event weights are not calibrated confidence; excluded as both targets and parents.'),
        piano_engine='specialist' if piano_detector is not None else 'general',
        instrument_engine='tsumugi' if instrument_detector is not None else 'general',
        guitar_engine='fl' if guitar_detector is not None else 'general',
        allow_predicted_releases=allow_predicted_releases,
        allow_guitar_predicted_releases=allow_guitar_predicted_releases,
        thresholds={k: settings[k] for k in ("onset_threshold", "frame_threshold")},
        engine=separation_label(separation)+" + source-verified Basic Pitch + vocal phrases"+
            (' + piano specialist' if piano_detector is not None else '')+
            (' + optional Tsumugi bass/other' if instrument_detector is not None else '')+
            (' + optional FL guitar' if guitar_detector is not None else '')+
            (" + Beat This meter" if beats_ms else ''),
        notice="分离可能串音或遗漏乐器；声轨更多不代表采音必然更准。跨音高人声 LN 是编排选择，请用独立声部试听检查。")
    from .structure import detect_structure
    progress(97, '比较音色与声部变化，建议音乐段落')
    result['structure'] = detect_structure(result, samples, SR, cancelled=cancelled)
    result['segments'] = result['structure']['segments']
    result['elapsed_seconds'] = round(time.perf_counter()-started, 2)
    progress(100, "多声部采音、持续段与音乐分段已准备好")
    return result
