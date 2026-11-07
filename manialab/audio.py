"""CPU audio evidence. Basic Pitch model/inference convention: Spotify, Apache-2.0.

Window timestamps are reconstructed from actual sample offsets, not a rounded FPS.
The detector is not a source separator and its activations are not calibrated probabilities.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
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
from .timing import estimate_timing

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


def pitched_notes(frames, onsets, times, duration, onset_threshold=.45, frame_threshold=.3, attacks=None):
    notes = []
    attack_times = np.array([n.start / 1000 for n in attacks]) if attacks is not None else None
    for pitch in range(88):
        peaks, _ = find_peaks(onsets[:, pitch], height=onset_threshold, distance=3)
        if len(onsets) > 1 and onsets[0, pitch] >= onset_threshold and onsets[0, pitch] > onsets[1, pitch]:
            peaks = np.r_[0, peaks]
        accepted = []
        for i in peaks:
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


def rhythmic_evidence(samples):
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
    return notes, estimate_timing(notes, len(samples) / SR)


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


def analyze(path, work, model, settings, progress=lambda p, s: None, cancelled=lambda: False):
    started = time.perf_counter()
    progress(2, "解码音频")
    samples = decode(path, work)
    duration = len(samples) / SR
    cache = Path(work) / "activations.npz"
    cached = None
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as data:
                if "version" in data and int(data["version"]) == 1:
                    cached = (data["frames"], data["onsets"], data["times"])
        except (ValueError, OSError, EOFError):
            cached = None
    if cached is not None:
        frames, onsets, times = cached
        progress(85, "复用已分析的音频，重新提取候选")
    else:
        frames, onsets, times = infer(samples, model, progress, cancelled)
        temporary = cache.with_name("activations.pending.npz")
        np.savez_compressed(temporary, frames=frames, onsets=onsets, times=times, version=1)
        temporary.replace(cache)
    if cancelled():
        raise InterruptedError("已取消分析")
    progress(90, "校准瞬态、比较 BPM 候选并检查全曲漂移")
    transients, tempo = rhythmic_evidence(samples)
    notes = pitched_notes(frames, onsets, times, duration, settings["onset_threshold"], settings["frame_threshold"], transients)
    notes, unmatched, fused = fuse_attacks(notes, transients)
    chunks = np.array_split(np.abs(samples), min(2400, len(samples)))
    waveform = [round(float(np.max(x)), 4) for x in chunks]
    candidates = sorted(notes + unmatched, key=lambda n: (n.start, n.pitch))
    segments = []
    for start in range(0, int(duration * 1000) + 1, 15000):
        local = [n for n in candidates if start <= n.start < start + 15000]
        segments.append({"start_ms": start, "end_ms": min(round(duration * 1000), start + 15000),
                         "pitched": sum(n.source == "pitch" for n in local),
                         "transients": sum(n.source == "transient" for n in local),
                         "sustains_over_800ms": sum(n.end - n.start >= 800 for n in local)})
    progress(100, "采音候选已准备好")
    return {"revision": 2, "duration": duration, "waveform": waveform, "tempo": tempo, "segments": segments,
            "elapsed_seconds": round(time.perf_counter() - started, 2), "cached": cached is not None,
            "timing_attacks": [n.data() for n in transients], "fused_attacks": fused,
            "candidates": [n.data() for n in candidates], "thresholds": {k: settings[k] for k in ("onset_threshold", "frame_threshold")},
            "engine": "Basic Pitch 0.4.0 ONNX + onset refinement + phase regression", "notice": "音高候选并非乐器分轨；复杂混音、混响和音尾仍可能误判。"}
