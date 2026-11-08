"""CPU audio evidence. Basic Pitch model/inference convention: Spotify, Apache-2.0.

Window timestamps are reconstructed from actual sample offsets, not a rounded FPS.
The detector is not a source separator and its activations are not calibrated probabilities.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import hashlib
import subprocess
import threading
import time

import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, stft
from scipy.io import wavfile

from .domain import Note
from .timing import estimate_timing, cover_missing_timing_windows

SR, FRAME, WINDOW, OVERLAP = 22050, 256, 43844, 7680
_session = None
_lock = threading.Lock()


def run_hidden(args, **kwargs):
    import os
    return subprocess.run(args, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, **kwargs)


def decode(path, work):
    result = run_hidden([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
                         "-vn", "-ac", "1", "-ar", str(SR), "-t", "1201", "-f", "f32le", "pipe:1"],
                        capture_output=True, timeout=180)
    if result.returncode:
        raise ValueError("音频无法解码，请换用 WAV / MP3 / FLAC / OGG")
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(samples) < SR // 4:
        raise ValueError("音频短于 0.25 秒")
    if len(samples) > SR * 1200:
        raise ValueError("首版支持不超过 20 分钟的音频")
    if not np.all(np.isfinite(samples)):
        raise ValueError("音频含无效采样值")
    wavfile.write(Path(work) / "preview.wav", SR, (np.clip(samples, -1, 1) * 32767).astype(np.int16))
    return samples


def infer(samples, model, progress=lambda p, s: None, cancelled=lambda: False):
    global _session
    with _lock:
        if _session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 3
            options.inter_op_num_threads = 1
            _session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    padded = np.pad(samples, (OVERLAP // 2, 0))
    hop = WINDOW - OVERLAP
    frames, onsets, times = [], [], []
    positions = list(range(0, len(padded), hop))
    for index, pos in enumerate(positions):
        if cancelled():
            raise InterruptedError("已取消分析")
        chunk = np.pad(padded[pos:pos + WINDOW], (0, max(0, pos + WINDOW - len(padded))))
        output = _session.run(["StatefulPartitionedCall:1", "StatefulPartitionedCall:2"],
                              {"serving_default_input_2:0": chunk.astype(np.float32)[None, :, None]})
        frames.append(output[0][0, 15:-15])
        onsets.append(output[1][0, 15:-15])
        times.extend((pos + np.arange(142) * FRAME) / SR)
        progress(8 + int(77 * (index + 1) / len(positions)), f"音高 / 持续音分析 {index + 1}/{len(positions)}")
    times = np.asarray(times)
    valid = times < len(samples) / SR
    return np.concatenate(frames)[valid], np.concatenate(onsets)[valid], times[valid]


def pitched_notes(frames, onsets, times, duration, onset_threshold=.45, frame_threshold=.3, attacks=None, suppressed=()):
    notes = []
    suppressed = {(r['pitch'], r['time_ms']) for r in suppressed}
    attack_times = np.array([n.start / 1000 for n in attacks]) if attacks is not None else None
    for pitch in range(88):
        peaks, _ = find_peaks(onsets[:, pitch], height=onset_threshold, distance=3)
        if len(onsets) > 1 and onsets[0, pitch] >= onset_threshold and onsets[0, pitch] > onsets[1, pitch]:
            peaks = np.r_[0, peaks]
        accepted = []
        for i in peaks:
            if (pitch+21, round(float(times[i])*1000)) in suppressed:
                continue
            before = frames[max(0, i - 6):i, pitch]
            after = frames[i + 3:i + 12, pitch]
            supported = attack_times is None or bool(np.any(np.abs(attack_times - times[i]) <= .045))
            # The model can fire another onset during a release, splitting one
            # long sound into an LN plus a fake tap. Reject only this falling
            # tail shape when independent attack evidence is also absent.
            release_peak = (len(before) >= 3 and len(after) >= 4 and
                            np.median(before) >= frame_threshold and
                            np.mean(after) < np.median(before) * .8 and
                            np.min(after) < frame_threshold)
            if not supported and release_peak:
                continue
            accepted.append(i)
        peaks = np.asarray(accepted, dtype=int)
        for j, start in enumerate(peaks):
            stop = int(peaks[j + 1]) if j + 1 < len(peaks) else len(times)
            last, gap = int(start), 0
            for k in range(int(start), stop):
                if frames[k, pitch] >= frame_threshold:
                    last, gap = k, 0
                else:
                    gap += 1
                    if gap > 4:
                        break
            if last - start < 3:
                continue
            a = max(0, round(float(times[start]) * 1000))
            b = min(round(duration * 1000), round((float(times[last]) + FRAME / SR) * 1000))
            if b <= a:
                continue
            strength = float(np.mean(frames[start:last + 1, pitch]))
            notes.append(Note(a, b, -1, pitch + 21, round(strength, 4), "pitch", f"p{pitch}:{a}"))
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def rhythmic_evidence(samples, beats_ms=None, estimate=True):
    _, times, z = stft(samples, SR, nperseg=1024, noverlap=768)
    magnitude = np.abs(z)
    flux = np.r_[0, np.maximum(0, np.diff(magnitude, axis=1)).sum(axis=0)]
    energy = np.sqrt(np.sum(magnitude ** 2, axis=0))
    rise = energy - np.r_[np.zeros(min(3, len(energy))), energy[:-3]]
    # Spectral spreading at a release is not a new attack. Require a small energy rise.
    flux[rise <= np.maximum(np.max(energy) * .002, energy * .01)] = 0
    amplitude = np.max(magnitude, axis=0)
    flux[amplitude < max(.00003, float(np.max(amplitude)) * .004)] = 0
    if not np.any(flux):
        return [], {"bpm": 120.0, "offset": 0.0, "estimated": True, "reliability": "无可靠节奏证据"}
    base = median_filter(flux, size=43)
    peaks, _ = find_peaks(flux, height=base * 1.6 + np.max(flux) * .025, distance=3, prominence=np.max(flux) * .02)
    scale = max(float(np.percentile(flux[peaks], 90)) if len(peaks) else 1, 1e-9)
    notes = []
    # Refine each STFT attack on a 1 ms RMS envelope. The flux peak itself can
    # precede or follow the physical attack by a window-dependent amount.
    step = 22
    usable = len(samples) // step * step
    rms = np.sqrt(np.mean(samples[:usable].reshape(-1, step) ** 2, axis=1))
    smooth = np.convolve(rms, np.ones(3) / 3, mode="same")
    for i in peaks:
        center = int(times[i] * SR / step)
        lo, hi = max(0, center - 30), min(len(smooth), center + 25)
        if hi - lo < 3:
            continue
        region = smooth[lo:hi]
        peak = int(np.argmax(region))
        floor = float(np.percentile(region[:max(1, peak)], 15))
        threshold = floor + (float(region[peak]) - floor) * .2
        crossing = peak
        while crossing > 0 and region[crossing - 1] > threshold:
            crossing -= 1
        at = max(0, round((lo + crossing) * step / SR * 1000))
        if notes and at - notes[-1].start < 35:
            continue
        notes.append(Note(at, at, -1, 48, round(min(1, float(flux[i]) / scale), 4), "transient", f"t{i}"))
    return notes, estimate_timing(notes, len(samples) / SR, beats_ms) if estimate else None


def fuse_attacks(pitched, transients):
    """Anchor nearby model onsets to measured attacks and avoid duplicate taps."""
    times = np.array([n.start for n in transients])
    if not len(times):
        return pitched, transients, 0
    used, corrected = set(), []
    for n in pitched:
        i = int(np.searchsorted(times, n.start))
        nearby = [j for j in (i - 1, i) if 0 <= j < len(times)]
        j = min(nearby, key=lambda j: abs(int(times[j]) - n.start))
        # A small correction is allowed; remote transients are unrelated voices.
        if abs(int(times[j]) - n.start) <= 40 and times[j] < n.end:
            used.add(j)
            n = replace(n, start=int(times[j]))
        corrected.append(n)
    return corrected, [n for j, n in enumerate(transients) if j not in used], len(used)


def refine_releases(notes, samples, sample_rate=SR):
    """Remove model tails extending into a source's near-silence.

    A 5ms waveform envelope can shorten a detected interval, never lengthen
    one. Other simultaneous sounds may keep a tail ambiguous; in that case
    the pitch model's original endpoint is preserved.
    """
    step = max(1, round(sample_rate * .005))
    padded = np.pad(samples, (0, (-len(samples)) % step))
    level = np.sqrt(np.mean(padded.reshape(-1, step) ** 2, axis=1))
    floor = max(.00001, float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) * .002)
    result, shifts = [], []
    for note in notes:
        a = max(0, int(note.start * sample_rate / 1000 / step))
        b = min(len(level), int(np.ceil(note.end * sample_rate / 1000 / step)))
        region = level[a:b]
        if not note.hold or not len(region):
            result.append(note)
            continue
        threshold = max(floor, float(np.percentile(region, 85)) * .02)
        active = np.flatnonzero(region >= threshold)
        end = min(note.end, round((a + int(active[-1]) + 1) * step / sample_rate * 1000)) if len(active) else note.start
        end = max(note.start, end)
        if end < note.end:
            shifts.append(note.end - end)
        result.append(replace(note, end=end))
    return result, dict(shortened=len(shifts), max_trim_ms=max(shifts, default=0),
                        reason="只收回进入声部近静音的模型音尾，不延长持续音")


def analyze(path, work, model, settings, progress=lambda p, s: None, cancelled=lambda: False):
    started = time.perf_counter()
    progress(2, "解码音频")
    samples = decode(path, work)
    duration = len(samples) / SR
    cache = Path(work) / "activations.npz"
    audio_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cached = None
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as data:
                if "version" in data and int(data["version"]) == 2 and str(data["audio_sha256"]) == audio_hash:
                    cached = (data["frames"], data["onsets"], data["times"])
        except (ValueError, KeyError, OSError, EOFError):
            cached = None
    if cached is not None:
        frames, onsets, times = cached
        progress(85, "复用已分析的音频，重新提取候选")
    else:
        frames, onsets, times = infer(samples, model, progress, cancelled)
        temporary = cache.with_name("activations.pending.npz")
        np.savez_compressed(temporary, frames=frames, onsets=onsets, times=times, version=2, audio_sha256=audio_hash)
        temporary.replace(cache)
    if cancelled():
        raise InterruptedError("已取消分析")
    progress(90, "校准瞬态、比较 BPM 候选并检查全曲漂移")
    transients, tempo = rhythmic_evidence(samples)
    notes = pitched_notes(frames, onsets, times, duration, settings["onset_threshold"], settings["frame_threshold"], transients)
    notes, releases = refine_releases(notes, samples)
    notes, unmatched, fused = fuse_attacks(notes, transients)
    chunks = np.array_split(np.abs(samples), min(2400, len(samples)))
    waveform = [round(float(np.max(x)), 4) for x in chunks]
    candidates = sorted(notes + unmatched, key=lambda n: (n.start, n.pitch))
    result = {"revision": 4, "duration": duration, "waveform": waveform, "tempo": tempo,
            "elapsed_seconds": round(time.perf_counter() - started, 2), "cached": cached is not None,
            "timing_attacks": [n.data() for n in transients], "fused_attacks": fused,
            "release_refinement": releases,
            "candidates": [n.data() for n in candidates], "thresholds": {k: settings[k] for k in ("onset_threshold", "frame_threshold")},
            "engine": "Basic Pitch 0.4.0 ONNX + onset refinement + phase regression", "notice": "音高候选并非乐器分轨；复杂混音、混响和音尾仍可能误判。"}
    from .structure import detect_structure
    progress(97, '比较音色与声部变化，建议音乐段落')
    result['structure'] = detect_structure(result, samples, SR)
    result['segments'] = result['structure']['segments']
    result['elapsed_seconds'] = round(time.perf_counter()-started, 2)
    progress(100, "采音候选与段落已准备好")
    return result


def align_stem_samples(stem, length):
    """Reconcile only final-sample rounding from two equivalent resample paths.

    FFmpeg's direct 48k->22.05k and 48k->44.1k->22.05k paths can differ by
    one sample at EOF. Never move the start or conceal a meaningful mismatch.
    """
    if stem.ndim != 1 or not np.isfinite(stem).all() or abs(len(stem)-length) > 2:
        raise ValueError("分离声部长度或采样值无效，请重新分析")
    difference = length-len(stem)
    if difference > 0:
        stem = np.pad(stem, (0, difference))
    elif difference < 0:
        stem = stem[:length]
    return stem, difference


def source_evidence(notes, stem, mixture, sample_rate=SR):
    """Reject numerical/leakage residue before normalizing a source's attacks.

    Per-source peak normalization can turn a -60dB residual into a supposedly
    strong event. Require waveform energy relative to the original local
    mixture, with a small whole-track floor for silent gaps.
    """
    global_level = float(np.sqrt(np.mean(mixture.astype(np.float64) ** 2)))
    kept = []
    for note in notes:
        at = round(note.start * sample_rate / 1000)
        a, b = max(0, at - round(sample_rate * .012)), min(len(mixture), at + round(sample_rate * .06))
        if b <= a:
            continue
        level = float(np.sqrt(np.mean(stem[a:b] ** 2)))
        reference = float(np.sqrt(np.mean(mixture[a:b] ** 2)))
        if level >= max(.00001, global_level * .0015, reference * .01):
            kept.append(note)
    return kept, len(notes) - len(kept)


def timing_evidence_quality(tempo):
    """Compare sources before UI rounding can damage a real fractional beat."""
    return next((c for c in tempo.get("precise_candidates", []) if c["bpm"] == tempo.get("exact_bpm")), tempo)


def reject_source_bleed(notes, stems, mixture_attacks, sample_rate=SR):
    """Reject short weak cross-source duplicates and release-edge imitations.

    A weak 100ms pitch at a stronger source's same-pitch onset or release can
    be separation leakage, not a second instrument. Restrict this test to a
    narrowly contained, low-activation candidate; preserve independent attacks
    and different-pitch voices. Unison instruments remain inherently ambiguous.
    """
    by_pitch = {}
    for note in notes:
        if note.source == 'pitch' and note.strength >= .55:
            by_pitch.setdefault(note.pitch, []).append(note)
    attack_times = np.array([n.start for n in mixture_attacks])
    result, rejected = [], []
    for note in notes:
        reason = None
        if note.source == 'pitch' and note.strength < .5 and 0 < note.end-note.start < 200:
            parents = [p for pitch in range(note.pitch-1,note.pitch+2) for p in by_pitch.get(pitch,())
                       if p.voice != note.voice and p.start <= note.start+40 and p.end >= note.end-40
                       and p.end-p.start >= max(250,(note.end-note.start)*3)]
            for parent in parents:
                a = max(0, round(max(note.start,parent.start)*sample_rate/1000))
                b = min(len(stems[note.voice]), round(min(note.end,parent.end,note.start+150)*sample_rate/1000))
                if b-a < sample_rate*.04:
                    continue
                x,y = stems[note.voice][a:b],stems[parent.voice][a:b]
                ratio = float(np.linalg.norm(x)/max(1e-9,np.linalg.norm(y)))
                if abs(note.start-parent.start) <= 40 and ratio < .1:
                    reason='weak_duplicate_attack'
                elif (note.start > parent.start+80 and abs(note.end-parent.end) <= 80 and ratio < .5
                      and not np.any(abs(attack_times-note.start) <= 45)):
                    correlation=float(abs(np.dot(x,y))/max(1e-9,float(np.linalg.norm(x)*np.linalg.norm(y))))
                    if correlation > .85:
                        reason='release_edge_bleed'
                if reason:
                    rejected.append(dict(note=note.data(),parent_id=parent.id,reason=reason,energy_ratio=round(ratio,4)))
                    break
        if not reason:
            result.append(note)
    return result, rejected


def analyze_deep(path, work, model, settings, separator, progress=lambda p, s: None, cancelled=lambda: False):
    """Separate sources, then run pitch/attack analysis on each isolated voice."""
    from .voices import vocal_phrases
    from .stems import profile_for
    started = time.perf_counter()
    samples = decode(path, work)
    duration = len(samples) / SR
    stem_root, separation = separator.separate(path, work, progress, cancelled)
    beats_ms = separation.get("beat_evidence", {}).get("beats_ms")
    voices = profile_for(separation.get('profile','four'))['voices']
    pitch_cached = []
    all_candidates, phrases, summaries, drum_attacks, tempos = [], [], [], [], {}
    for index, voice in enumerate(voices):
        if cancelled():
            raise InterruptedError("已取消声部分析")
        stem = np.load(stem_root / f"{voice}.npy", allow_pickle=False)
        stem, end_padding = align_stem_samples(stem, len(samples))
        stage = 31 + round(index * 61 / len(voices))
        progress(stage, f"分析 {voice} 的起音与持续区间")
        attacks, _ = rhythmic_evidence(stem, estimate=False)
        attacks, rejected_attacks = source_evidence(attacks, stem, samples)
        if voice == "drums":
            tempos[voice] = estimate_timing(attacks, duration, beats_ms)
        attacks = [replace(n, voice=voice, id=f"{voice}:{n.id}") for n in attacks]
        cached = False
        articulation = []
        if voice == "drums":
            drum_attacks = attacks
            candidates = attacks
        else:
            cache = stem_root / f"{voice}-activations.npz"
            data = None
            if cache.is_file():
                try:
                    with np.load(cache, allow_pickle=False) as d:
                        if (int(d["version"]) == 2 and str(d["audio_sha256"]) == separation["audio_sha256"]
                                and str(d['separation_sha256']) == separation['model_sha256']):
                            data = d["frames"], d["onsets"], d["times"]
                except (ValueError, KeyError, OSError, EOFError):
                    pass
            if data is None:
                data = infer(stem, model, lambda p, s: progress(stage + round(p * .60 / len(voices)), f"{voice} · {s}"), cancelled)
                np.savez_compressed(cache, frames=data[0], onsets=data[1], times=data[2], version=2,
                    audio_sha256=separation['audio_sha256'],separation_sha256=separation['model_sha256'])
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
                              sustains_over_800ms=sum(n.end - n.start >= 800 for n in candidates)))
        peak = max(float(np.max(np.abs(stem))), 1.)
        wavfile.write(stem_root / f"{voice}.wav", SR, (np.clip(stem / peak, -1, 1) * 32767).astype(np.int16))
    progress(94, "比较鼓声与混音拍线，保留人声持续段证据")
    mix_attacks, mix_tempo = rhythmic_evidence(samples, beats_ms)
    stem_signals = {v:np.load(stem_root/f'{v}.npy',allow_pickle=False,mmap_mode='r') for v in voices if v!='drums'}
    all_candidates, bleed_rejections = reject_source_bleed(all_candidates,stem_signals,mix_attacks)
    for summary in summaries:
        summary['rejected_cross_source_bleed'] = sum(r['note']['voice']==summary['voice'] for r in bleed_rejections)
        summary['candidates'] = sum(n.voice==summary['voice'] for n in all_candidates)
    drum_tempo = tempos["drums"]
    drum_quality, mix_quality = timing_evidence_quality(drum_tempo), timing_evidence_quality(mix_tempo)
    use_drums = len(drum_attacks) >= 24 and drum_quality.get("coverage", 0) >= .7 and drum_quality.get("score", 0) >= mix_quality.get("score", 0) * .9
    tempo = dict(drum_tempo if use_drums else mix_tempo)
    attacks = drum_attacks if use_drums else mix_attacks
    tempo["evidence_source"] = "drums" if use_drums else "mix"
    tempo["comparisons"] = {"drums": drum_tempo, "mix": mix_tempo}
    if use_drums:
        tempo, attacks = cover_missing_timing_windows(tempo,drum_attacks,mix_attacks)
    candidates = sorted(all_candidates, key=lambda n: (n.start, n.pitch, n.voice))
    result = dict(revision=9, duration=duration, tempo=tempo,
        waveform=[round(float(np.max(x)), 4) for x in np.array_split(np.abs(samples), min(2400, len(samples)))],
        elapsed_seconds=round(time.perf_counter() - started, 2), cached=separation["cached"] and all(pitch_cached),
        timing_attacks=[n.data() for n in attacks], fused_attacks=0, candidates=[n.data() for n in candidates],
        phrase_candidates=[n.data() for n in phrases], voices=summaries, separation=separation,
        source_bleed_rejections=bleed_rejections,
        thresholds={k: settings[k] for k in ("onset_threshold", "frame_threshold")},
        engine=profile_for(separation.get('profile','four'))['label']+" + source-verified Basic Pitch + vocal phrases + Beat This meter",
        notice="分离可能串音或遗漏乐器；声轨更多不代表采音必然更准。跨音高人声 LN 是编排选择，请用独立声部试听检查。")
    from .structure import detect_structure
    progress(97, '比较音色与声部变化，建议音乐段落')
    result['structure'] = detect_structure(result, samples, SR)
    result['segments'] = result['structure']['segments']
    result['elapsed_seconds'] = round(time.perf_counter()-started, 2)
    progress(100, "多声部采音、持续段与音乐分段已准备好")
    return result
