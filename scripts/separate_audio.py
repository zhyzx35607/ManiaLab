"""CPU Hybrid Demucs worker; communicates progress as JSON lines.

The worker runs in a separate process so cancellation also releases model memory.
Input is a decoded stereo 44.1 kHz PCM WAV. It never reads API keys or uploads audio.
"""
from __future__ import annotations

if __name__ == '__main__':
    from worker_lifetime import watch_parent
    watch_parent()

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch
from torchaudio.models import hdemucs_high

SOURCES = ["drums", "bass", "other", "vocals"]
MODEL_URL = "https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt"
MODEL_SHA256 = "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"
BEAT_MODEL_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"


def progress(percent, message, **extra):
    print(json.dumps(dict(progress=percent, message=message, **extra), ensure_ascii=True), flush=True)


def beat_evidence(source, model_path, seconds=0):
    from beat_this.inference import Audio2Beats
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != BEAT_MODEL_SHA256:
        raise ValueError("Beat This model checksum mismatch")
    progress(94, "Checking beat structure and half/double time")
    sample_rate, data = wavfile.read(source)
    samples = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        samples /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if seconds:
        samples = samples[:round(seconds * sample_rate)]
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError("Invalid beat-model audio input")
    started = time.perf_counter()
    model = Audio2Beats(checkpoint_path=str(model_path), device="cpu", dbn=False)
    beats, downbeats = model(samples, sample_rate)
    return dict(model="Beat This! final0", model_sha256=BEAT_MODEL_SHA256,
        beats_ms=np.round(beats * 1000).astype(int).tolist(),
        downbeats_ms=np.round(downbeats * 1000).astype(int).tolist(),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        purpose="beat-unit selection only; onset audio establishes the offset")


def separate(source, output, model_path, seconds=0, threads=3, beat_model=None):
    started = time.perf_counter()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        progress(1, "Downloading official Hybrid Demucs weights")
        temporary = model_path.with_suffix(".pending")
        torch.hub.download_url_to_file(MODEL_URL, str(temporary), progress=False)
        temporary.replace(model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != MODEL_SHA256:
        raise ValueError("Hybrid Demucs model checksum mismatch")
    progress(3, "Loading four-source separation model", model_sha256=model_hash)
    model = hdemucs_high(sources=SOURCES)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    sample_rate, data = wavfile.read(source)
    if sample_rate != 44100 or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Expected a stereo 44100 Hz WAV")
    samples = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        samples /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if seconds:
        samples = samples[:round(seconds * sample_rate)]
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Invalid audio input")
    mixture = torch.from_numpy(samples.T.copy())[None]
    length = mixture.shape[-1]
    # Continuous overlap-add, with strictly positive normalized weights. A
    # half-second of context at both edges avoids padding at every seam.
    segment, hop, context = 8 * sample_rate, 6 * sample_rate, sample_rate // 2
    result = np.zeros((4, length), dtype=np.float32)
    weights = np.zeros(length, dtype=np.float32)
    positions = list(range(0, length, hop))
    for index, start in enumerate(positions):
        stop = min(length, start + segment)
        left, right = max(0, start - context), min(length, stop + context)
        chunk = mixture[:, :, left:right]
        minimum = sample_rate
        extra = max(0, minimum - chunk.shape[-1])
        if extra:
            chunk = torch.nn.functional.pad(chunk, (0, extra))
        if float(chunk.abs().max()) < 1e-7:
            separated = np.zeros((4, stop - start), dtype=np.float32)
        else:
            with torch.inference_mode():
                estimate = model(chunk)[0, :, :, start - left:stop - left]
            separated = estimate.mean(dim=1).numpy()
        weight = np.ones(stop - start, dtype=np.float32)
        fade = min(segment - hop, len(weight))
        if start > 0:
            weight[:fade] *= np.linspace(1e-5, 1, fade)
        if stop < length:
            weight[-fade:] *= np.linspace(1, 1e-5, fade)
        result[:, start:stop] += separated * weight
        weights[start:stop] += weight
        progress(5 + round(88 * (index + 1) / len(positions)), "Separating drums / bass / other / vocals",
                 chunk=index + 1, chunks=len(positions), elapsed_seconds=round(time.perf_counter() - started, 2))
    result /= np.maximum(weights, 1e-8)
    if not np.isfinite(result).all():
        raise ValueError("Source separation produced non-finite samples")
    files = {}
    for name, stem in zip(SOURCES, result):
        path = output / f"{name}.npy"
        np.save(path, resample_poly(stem, 1, 2).astype(np.float32))
        files[name] = path.name
    report = dict(revision=1, sample_rate=22050, frames=(length + 1) // 2,
        model_url=MODEL_URL, model_sha256=model_hash, sources=files,
        elapsed_seconds=round(time.perf_counter() - started, 2),
        reconstruction_rmse=float(np.sqrt(np.mean((result.sum(axis=0) - samples.mean(axis=1)) ** 2))))
    if beat_model is not None:
        del model, mixture, result
        gc.collect()
        report["beat_evidence"] = beat_evidence(source, beat_model, seconds)
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    (output / "separation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress(100, "Separation complete", **report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--beat-model", type=Path)
    parser.add_argument("--beats-only", action="store_true")
    parser.add_argument('--profile',choices=('four','six'),default='four')
    args = parser.parse_args()
    if args.beats_only:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        manifest = args.output / "separation.json"
        report = json.loads(manifest.read_text(encoding="utf-8"))
        report["beat_evidence"] = beat_evidence(args.input, args.beat_model, args.seconds)
        manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        progress(100, "Beat structure complete")
    elif args.profile=='six':
        from separate_six import separate_six
        separate_six(args.input,args.output,args.model,args.seconds,args.threads,args.beat_model,progress,beat_evidence)
    else:
        separate(args.input, args.output, args.model, args.seconds, args.threads, args.beat_model)
