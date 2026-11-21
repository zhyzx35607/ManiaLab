import hashlib
import io
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, stft

from manialab import audio_analysis
from manialab.audio_config import SR
from manialab.domain import Note
from manialab.timing import estimate_timing


def _click_track(seconds):
    samples = np.zeros(seconds * SR, dtype=np.float64)
    pulse_size = 500
    pulse = .8 * np.hanning(pulse_size) * np.sin(
        2 * np.pi * 1000 * np.arange(pulse_size) / SR)
    for start in range(0, len(samples) - pulse_size, SR // 2):
        samples[start:start + pulse_size] += pulse
    return samples


def _chunk_boundary_click_track(offset_samples):
    samples = _click_track(8)
    pulse_size = 500
    pulse = .8 * np.hanning(pulse_size) * np.sin(
        2 * np.pi * 1000 * np.arange(pulse_size) / SR)
    regular_start = 3 * SR
    samples[regular_start:regular_start + pulse_size] -= pulse
    boundary = audio_analysis._RHYTHM_STFT_CHUNK_FRAMES * audio_analysis._RHYTHM_STFT_HOP
    start = boundary + offset_samples - pulse_size // 2
    samples[start:start + pulse_size] += pulse
    return samples


def _legacy_rhythmic_evidence(samples, beats_ms=None, estimate=True):
    _, times, spectrum = stft(samples, SR, nperseg=1024, noverlap=768)
    magnitude = np.abs(spectrum)
    flux = np.r_[0, np.maximum(0, np.diff(magnitude, axis=1)).sum(axis=0)]
    energy = np.sqrt(np.sum(magnitude ** 2, axis=0))
    rise = energy - np.r_[np.zeros(min(3, len(energy))), energy[:-3]]
    flux[rise <= np.maximum(np.max(energy) * .002, energy * .01)] = 0
    amplitude = np.max(magnitude, axis=0)
    flux[amplitude < max(.00003, float(np.max(amplitude)) * .004)] = 0
    if not np.any(flux):
        return [], {"bpm": 120.0, "offset": 0.0, "estimated": True,
                    "reliability": "无可靠节奏证据"}
    baseline = median_filter(flux, size=43)
    peaks, _ = find_peaks(flux, height=baseline * 1.6 + np.max(flux) * .025,
                          distance=3, prominence=np.max(flux) * .02)
    scale = max(float(np.percentile(flux[peaks], 90)) if len(peaks) else 1, 1e-9)
    notes = []
    step = 22
    usable = len(samples) // step * step
    rms = np.sqrt(np.mean(samples[:usable].reshape(-1, step) ** 2, axis=1))
    smooth = np.convolve(rms, np.ones(3) / 3, mode="same")
    for index in peaks:
        center = int(times[index] * SR / step)
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
        notes.append(Note(at, at, -1, 48,
                          round(min(1, float(flux[index]) / scale), 4),
                          "transient", f"t{index}"))
    timing = estimate_timing(notes, len(samples) / SR, beats_ms) if estimate else None
    return notes, timing


def _assert_timing_equal_with_raw_bpm_roundoff(actual, expected):
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            if key == "raw_bpm":
                assert abs(actual[key] - expected[key]) <= .000011
            else:
                _assert_timing_equal_with_raw_bpm_roundoff(actual[key], expected[key])
        return
    if isinstance(actual, list):
        assert isinstance(expected, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_timing_equal_with_raw_bpm_roundoff(actual_item, expected_item)
        return
    assert actual == expected


def test_chunked_rhythmic_evidence_matches_legacy_synthetic_oracle():
    samples = _click_track(8)

    actual_notes, actual_timing = audio_analysis.rhythmic_evidence(samples)
    expected_notes, expected_timing = _legacy_rhythmic_evidence(samples)

    assert actual_notes == expected_notes
    assert actual_timing == expected_timing
    assert actual_timing["bpm"] == 120.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_chunked_rhythmic_evidence_preserves_legacy_frame_time_rounding(dtype):
    samples = np.random.default_rng(20261121).normal(0, .03, 131071).astype(dtype)

    actual_notes, actual_timing = audio_analysis.rhythmic_evidence(samples)
    expected_notes, expected_timing = _legacy_rhythmic_evidence(samples)

    assert actual_notes == expected_notes
    assert actual_timing == expected_timing


def test_chunked_float32_stft_rounding_preserves_discrete_timing_outputs():
    samples = np.random.default_rng(1).normal(0, .03, 131071).astype(np.float32)

    actual_notes, actual_timing = audio_analysis.rhythmic_evidence(samples)
    expected_notes, expected_timing = _legacy_rhythmic_evidence(samples)

    actual_events = [(n.start, n.end, n.pitch, n.voice, n.id) for n in actual_notes]
    expected_events = [(n.start, n.end, n.pitch, n.voice, n.id) for n in expected_notes]
    assert actual_events == expected_events
    assert len(actual_notes) == len(expected_notes)
    assert all(abs(n.strength - expected.strength) <= .0001
               for n, expected in zip(actual_notes, expected_notes))

    _assert_timing_equal_with_raw_bpm_roundoff(actual_timing, expected_timing)


@pytest.mark.parametrize("offset_samples", [-255, -1, 0, 1, 255])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_chunk_boundary_pulses_match_legacy_synthetic_oracle(offset_samples, dtype):
    samples = _chunk_boundary_click_track(offset_samples).astype(dtype)

    actual_notes, actual_timing = audio_analysis.rhythmic_evidence(samples)
    expected_notes, expected_timing = _legacy_rhythmic_evidence(samples)

    assert actual_notes == expected_notes
    assert actual_timing == expected_timing


def test_chunked_float32_rise_threshold_matches_legacy_boundary_oracle(monkeypatch):
    frame_count = 2100
    samples = np.zeros((frame_count - 1) * audio_analysis._RHYTHM_STFT_HOP,
                       dtype=np.float32)
    magnitude = np.zeros((99, frame_count), dtype=np.float32)
    magnitude[:, 0] = np.float32(10)
    previous_energy = np.float32(9.750861)
    current_energy = np.float32(9.949859)
    magnitude[0, 1:4] = previous_energy
    magnitude[0, 4:] = current_energy
    global_energy = np.sqrt(np.sum(magnitude[:, 0] ** 2))
    legacy_threshold = np.float32(global_energy * .002)
    rise = float(current_energy) - float(previous_energy)
    assert rise == float(legacy_threshold)
    assert rise > float(global_energy) * .002
    spectrum = magnitude.astype(np.complex64)
    times = np.arange(frame_count) * audio_analysis._RHYTHM_STFT_HOP / SR
    chunk_cursor = [0]

    def synthetic_stft(values, *args, **kwargs):
        if kwargs.get("boundary", "scipy-default") is None:
            first = chunk_cursor[0]
            stop = min(frame_count, first + audio_analysis._RHYTHM_STFT_CHUNK_FRAMES)
            chunk_cursor[0] = stop
            return np.zeros(1), times[first:stop], spectrum[:, first:stop]
        return np.zeros(1), times, spectrum

    monkeypatch.setattr(audio_analysis, "stft", synthetic_stft)
    monkeypatch.setitem(globals(), "stft", synthetic_stft)
    expected_notes, expected_timing = _legacy_rhythmic_evidence(samples)
    actual_notes, actual_timing = audio_analysis.rhythmic_evidence(samples)

    assert expected_notes == []
    assert actual_notes == expected_notes
    assert actual_timing == expected_timing


def test_chunk_boundary_flux_keeps_legacy_peak_threshold_decision(monkeypatch):
    frame_count = 600
    samples = np.zeros((frame_count - 1) * audio_analysis._RHYTHM_STFT_HOP,
                       dtype=np.float64)
    pulse_size = 500
    pulse = .8 * np.hanning(pulse_size) * np.sin(
        2 * np.pi * 1000 * np.arange(pulse_size) / SR)
    for frame in (40, 256):
        start = frame * audio_analysis._RHYTHM_STFT_HOP - pulse_size // 2
        samples[start:start + pulse_size] += pulse
    magnitude = np.zeros((513, frame_count), dtype=np.float32)
    magnitude[0, 40] = 1.0
    magnitude[:8, 256] = np.array(
        [.0049999999, .0015919221, .0002417712, .0000975241,
         .0047988384, .0053858687, .0035795569, .0043045180],
        dtype=np.float32,
    )
    spectrum = magnitude.astype(np.complex64)
    times = np.arange(frame_count) * audio_analysis._RHYTHM_STFT_HOP / SR
    chunk_cursor = [0]

    def synthetic_stft(values, *args, **kwargs):
        if kwargs.get("boundary", "scipy-default") is None:
            first = chunk_cursor[0]
            stop = min(frame_count, first + audio_analysis._RHYTHM_STFT_CHUNK_FRAMES)
            chunk_cursor[0] = stop
            return np.zeros(1), times[first:stop], spectrum[:, first:stop]
        return np.zeros(1), times, spectrum

    monkeypatch.setattr(audio_analysis, "stft", synthetic_stft)
    monkeypatch.setitem(globals(), "stft", synthetic_stft)

    expected_notes, _ = _legacy_rhythmic_evidence(samples)
    actual_notes, _ = audio_analysis.rhythmic_evidence(samples)

    assert [note.id for note in expected_notes] == ["t40"]
    assert [note.id for note in actual_notes] == ["t40"]


@pytest.mark.parametrize("sample_count", [768, 769, 1023, 1024])
def test_short_stft_boundaries_match_scipy_legacy_oracle(sample_count):
    samples = np.zeros(sample_count, dtype=np.float32)
    pulse = np.hanning(min(64, sample_count)) * .5
    samples[100:100 + len(pulse)] = pulse

    if sample_count == 768:
        with pytest.raises(ValueError, match="noverlap must be less than nperseg"):
            _legacy_rhythmic_evidence(samples)
        with pytest.raises(ValueError, match="noverlap must be less than nperseg"):
            audio_analysis.rhythmic_evidence(samples)
        return

    if sample_count < 1024:
        with pytest.warns(UserWarning, match="nperseg = 1024"):
            expected_notes, expected_timing = _legacy_rhythmic_evidence(samples)
        with pytest.warns(UserWarning, match="nperseg = 1024"):
            actual_notes, actual_timing = audio_analysis.rhythmic_evidence(samples)
    else:
        expected_notes, expected_timing = _legacy_rhythmic_evidence(samples)
        actual_notes, actual_timing = audio_analysis.rhythmic_evidence(samples)

    assert actual_notes == expected_notes
    assert actual_timing == expected_timing


def test_chunked_rms_matches_legacy_whole_array_result():
    samples = _click_track(40)
    step = 22
    usable = len(samples) // step * step
    expected = np.sqrt(np.mean(samples[:usable].reshape(-1, step) ** 2, axis=1))

    actual = audio_analysis._chunked_rms(samples, step)

    assert np.array_equal(actual, expected)


def test_long_rhythmic_evidence_bounds_stft_and_rms_work_blocks(monkeypatch):
    samples = _click_track(40)
    real_stft = audio_analysis.stft
    real_rms_window_block = audio_analysis._rms_window_block
    observed = []
    observed_rms_blocks = []

    def spy(values, *args, **kwargs):
        result = real_stft(values, *args, **kwargs)
        observed.append((len(values), result[2].shape[-1]))
        return result

    def rms_spy(values, step):
        observed_rms_blocks.append((len(values), step))
        return real_rms_window_block(values, step)

    monkeypatch.setattr(audio_analysis, "stft", spy)
    monkeypatch.setattr(audio_analysis, "_rms_window_block", rms_spy)
    notes, timing = audio_analysis.rhythmic_evidence(samples)

    max_input_samples = (audio_analysis._RHYTHM_STFT_NPERSEG +
                         (audio_analysis._RHYTHM_STFT_CHUNK_FRAMES - 1) *
                         audio_analysis._RHYTHM_STFT_HOP)
    max_rms_samples = audio_analysis._RHYTHM_RMS_CHUNK_WINDOWS * 22
    assert len(observed) > 1
    assert all(input_samples <= max_input_samples and frame_count <=
               audio_analysis._RHYTHM_STFT_CHUNK_FRAMES
               for input_samples, frame_count in observed)
    assert len(observed_rms_blocks) > 1
    assert all(sample_count <= max_rms_samples and step == 22
               for sample_count, step in observed_rms_blocks)
    assert notes and timing["bpm"] == 120.0


def test_rhythmic_evidence_cancels_between_stft_chunks(monkeypatch):
    samples = _click_track(40)
    real_stft = audio_analysis.stft
    observed = []

    def spy(values, *args, **kwargs):
        result = real_stft(values, *args, **kwargs)
        observed.append(len(values))
        return result

    monkeypatch.setattr(audio_analysis, "stft", spy)
    with pytest.raises(InterruptedError, match="取消"):
        audio_analysis.rhythmic_evidence(samples, cancelled=lambda: len(observed) >= 2)

    assert len(observed) == 2


def test_rhythmic_evidence_cancels_between_rms_chunks(monkeypatch):
    samples = _click_track(40)
    real_rms_window_block = audio_analysis._rms_window_block
    observed = []

    def spy(values, step):
        result = real_rms_window_block(values, step)
        observed.append(len(values))
        return result

    monkeypatch.setattr(audio_analysis, "_rms_window_block", spy)
    with pytest.raises(InterruptedError, match="取消"):
        audio_analysis.rhythmic_evidence(samples, cancelled=lambda: bool(observed))

    assert len(observed) == 1


def test_basic_and_deep_analysis_forward_cancellation_to_rhythm(tmp_path, monkeypatch):
    from manialab import audio

    source = tmp_path / "mix.wav"
    source_bytes = b"synthetic mix"
    source.write_bytes(source_bytes)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"synthetic model identity")
    samples = np.full(SR * 2, .1, dtype=np.float32)
    settings = {"onset_threshold": .45, "frame_threshold": .3}
    forwarded = []
    cancellation = lambda: False

    def fake_infer(*args, **kwargs):
        values = np.zeros((1, 88), dtype=np.float32)
        return values, values.copy(), np.zeros(1, dtype=np.float64)

    def fake_rhythm(*args, **kwargs):
        forwarded.append(kwargs.get("cancelled"))
        return [], {"bpm": 120.0, "offset": 0.0, "estimated": True}

    monkeypatch.setattr(audio, "decode", lambda *args, **kwargs: samples)
    monkeypatch.setattr(audio, "infer", fake_infer)
    monkeypatch.setattr(audio, "rhythmic_evidence", fake_rhythm)
    monkeypatch.setattr(audio, "pitched_notes", lambda *args, **kwargs: [])
    monkeypatch.setattr(audio, "refine_releases", lambda notes, samples: (notes, {}))
    monkeypatch.setattr(audio, "fuse_attacks", lambda notes, attacks: (notes, [], 0))
    monkeypatch.setattr(audio, "source_evidence", lambda notes, *args: (notes, 0))
    monkeypatch.setattr(audio, "reject_source_bleed", lambda notes, *args: (notes, []))
    monkeypatch.setattr(audio, "timing_evidence_quality", lambda tempo: {"coverage": 0, "score": 0})
    monkeypatch.setattr(
        "manialab.structure.detect_structure",
        lambda analysis, samples, sample_rate, **kwargs: {"segments": []},
    )

    audio.analyze(source, tmp_path, model, settings, cancelled=cancellation)

    separation = dict(
        source="provided", voices=["drums"],
        audio_sha256=hashlib.sha256(source_bytes).hexdigest(),
        set_sha256="a" * 64, model_sha256="b" * 64, cached=True,
    )
    np.save(tmp_path / "drums.npy", samples, allow_pickle=False)

    class Separator:
        def separate(self, path, work, progress, cancelled):
            return tmp_path, separation

    audio.analyze_deep(source, tmp_path, model, settings, Separator(), cancelled=cancellation)

    assert forwarded == [cancellation, cancellation, cancellation]

    cancel_state = {"cancelled": False}
    drum_note = Note(100, 100, -1, 48, .8, "transient", "drum:100")
    rhythm_calls = []
    replaced = []
    real_replace = audio.replace

    def cancel_during_timing(*args):
        cancel_state["cancelled"] = True
        return {"bpm": 120.0, "offset": 0.0, "estimated": True}

    def drum_rhythm(*args, **kwargs):
        rhythm_calls.append(True)
        return [drum_note], None

    def replace_spy(*args, **kwargs):
        replaced.append(True)
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(audio, "rhythmic_evidence", drum_rhythm)
    monkeypatch.setattr(audio, "estimate_timing", cancel_during_timing)
    monkeypatch.setattr(audio, "replace", replace_spy)
    with pytest.raises(InterruptedError, match="取消"):
        audio.analyze_deep(
            source, tmp_path, model, settings, Separator(),
            cancelled=lambda: cancel_state["cancelled"],
        )

    assert rhythm_calls == [True]
    assert replaced == []


def _prepare_synthetic_basic_analysis(tmp_path, monkeypatch, infer_result):
    from manialab import audio

    source = tmp_path / "mix.wav"
    source_bytes = b"synthetic mix"
    source.write_bytes(source_bytes)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"synthetic model identity")
    samples = np.full(SR * 2, .1, dtype=np.float32)
    settings = {"onset_threshold": .45, "frame_threshold": .3}

    monkeypatch.setattr(audio, "decode", lambda *args, **kwargs: samples)
    monkeypatch.setattr(audio, "infer", lambda *args, **kwargs: infer_result())
    monkeypatch.setattr(
        audio, "rhythmic_evidence",
        lambda *args, **kwargs: ([], {"bpm": 120.0, "offset": 0.0, "estimated": True}),
    )
    monkeypatch.setattr(audio, "pitched_notes", lambda *args, **kwargs: [])
    monkeypatch.setattr(audio, "refine_releases", lambda notes, values: (notes, {}))
    monkeypatch.setattr(audio, "fuse_attacks", lambda notes, attacks: (notes, [], 0))
    monkeypatch.setattr(
        "manialab.structure.detect_structure",
        lambda analysis, values, sample_rate, **kwargs: {"segments": []},
    )
    return audio, source, source_bytes, model, settings


def _write_stored_activation_cache(path, audio_sha256, model_identity):
    from manialab.audio_cache import (
        BASIC_PITCH_INFERENCE_REVISION,
        WHOLE_AUDIO_ACTIVATION_CACHE_VERSION,
    )

    arrays = {
        "frames": np.full((2, 3), .25, dtype=np.float32),
        "onsets": np.full((2, 3), .5, dtype=np.float32),
        "times": np.array([0.0, .01, .02], dtype=np.float64),
        "version": np.asarray(WHOLE_AUDIO_ACTIVATION_CACHE_VERSION),
        "audio_sha256": np.asarray(audio_sha256),
        "model_identity": np.asarray(model_identity),
        "inference_revision": np.asarray(BASIC_PITCH_INFERENCE_REVISION),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            archive.writestr(f"{name}.npy", payload.getvalue())


def _corrupt_stored_zip_member(path, member):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    with path.open("r+b") as archive:
        archive.seek(info.header_offset)
        local_header = archive.read(30)
        filename_size, extra_size = struct.unpack_from("<HH", local_header, 26)
        data_offset = info.header_offset + 30 + filename_size + extra_size
        archive.seek(data_offset + info.file_size - 1)
        original = archive.read(1)
        archive.seek(data_offset + info.file_size - 1)
        archive.write(bytes([original[0] ^ 1]))


def _synthetic_activations():
    frames = np.full((2, 3), .75, dtype=np.float32)
    return frames, frames.copy(), np.array([0.0, .01, .02], dtype=np.float64)


def test_whole_audio_cache_recomputes_after_member_crc_error(tmp_path, monkeypatch):
    calls = []
    audio, source, source_bytes, model, settings = _prepare_synthetic_basic_analysis(
        tmp_path, monkeypatch, lambda: (calls.append(True), _synthetic_activations())[1])
    cache = tmp_path / "activations.npz"
    audio_hash = hashlib.sha256(source_bytes).hexdigest()
    fingerprint = audio.model_identity(model)
    _write_stored_activation_cache(cache, audio_hash, fingerprint)
    _corrupt_stored_zip_member(cache, "frames.npy")

    result = audio.analyze(source, tmp_path, model, settings)

    assert calls == [True]
    assert result["cached"] is False
    with np.load(cache, allow_pickle=False) as data:
        assert np.array_equal(data["frames"], _synthetic_activations()[0])


def test_whole_audio_cache_write_failure_preserves_canonical_and_cleans_temp(
        tmp_path, monkeypatch):
    audio, source, source_bytes, model, settings = _prepare_synthetic_basic_analysis(
        tmp_path, monkeypatch, _synthetic_activations)
    cache = tmp_path / "activations.npz"
    _write_stored_activation_cache(cache, "0" * 64, audio.model_identity(model))
    original_cache = cache.read_bytes()
    real_savez = np.savez_compressed

    def fail_after_write(*args, **kwargs):
        real_savez(*args, **kwargs)
        raise OSError("injected cache write failure")

    monkeypatch.setattr(audio.np, "savez_compressed", fail_after_write)
    with pytest.raises(OSError, match="injected cache write failure"):
        audio.analyze(source, tmp_path, model, settings)

    assert cache.read_bytes() == original_cache
    assert list(tmp_path.glob(".activations.npz.*.pending.npz")) == []


def test_whole_audio_cache_replace_failure_preserves_canonical_and_cleans_temp(
        tmp_path, monkeypatch):
    audio, source, source_bytes, model, settings = _prepare_synthetic_basic_analysis(
        tmp_path, monkeypatch, _synthetic_activations)
    cache = tmp_path / "activations.npz"
    _write_stored_activation_cache(cache, "0" * 64, audio.model_identity(model))
    original_cache = cache.read_bytes()
    real_replace = Path.replace

    def fail_cache_replace(path, target):
        if path.name.startswith(".activations.npz."):
            raise OSError("injected cache replace failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_cache_replace)
    with pytest.raises(OSError, match="injected cache replace failure"):
        audio.analyze(source, tmp_path, model, settings)

    assert cache.read_bytes() == original_cache
    assert list(tmp_path.glob(".activations.npz.*.pending.npz")) == []


def test_whole_audio_cache_cancellation_after_infer_does_not_write(tmp_path, monkeypatch):
    state = {"cancelled": False}

    def infer_result():
        state["cancelled"] = True
        return _synthetic_activations()

    audio, source, source_bytes, model, settings = _prepare_synthetic_basic_analysis(
        tmp_path, monkeypatch, infer_result)
    cache = tmp_path / "activations.npz"
    _write_stored_activation_cache(cache, "0" * 64, audio.model_identity(model))
    original_cache = cache.read_bytes()

    with pytest.raises(InterruptedError, match="取消"):
        audio.analyze(source, tmp_path, model, settings,
                      cancelled=lambda: state["cancelled"])

    assert cache.read_bytes() == original_cache
    assert list(tmp_path.glob(".activations.npz.*.pending.npz")) == []


def test_whole_audio_cache_cancellation_after_write_preserves_canonical_and_cleans_temp(
        tmp_path, monkeypatch):
    state = {"cancelled": False}
    audio, source, source_bytes, model, settings = _prepare_synthetic_basic_analysis(
        tmp_path, monkeypatch, _synthetic_activations)
    cache = tmp_path / "activations.npz"
    _write_stored_activation_cache(cache, "0" * 64, audio.model_identity(model))
    original_cache = cache.read_bytes()
    real_savez = np.savez_compressed

    def cancel_after_write(*args, **kwargs):
        real_savez(*args, **kwargs)
        state["cancelled"] = True

    monkeypatch.setattr(audio.np, "savez_compressed", cancel_after_write)
    with pytest.raises(InterruptedError, match="取消"):
        audio.analyze(source, tmp_path, model, settings,
                      cancelled=lambda: state["cancelled"])

    assert cache.read_bytes() == original_cache
    assert list(tmp_path.glob(".activations.npz.*.pending.npz")) == []


def test_deep_analysis_passes_cancellation_to_stem_cache_save(tmp_path, monkeypatch):
    from manialab import audio

    source = tmp_path / "mix.wav"
    source_bytes = b"synthetic mix"
    source.write_bytes(source_bytes)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"synthetic model identity")
    samples = np.full(SR * 2, .1, dtype=np.float32)
    np.save(tmp_path / "other.npy", samples, allow_pickle=False)
    separation = dict(
        source="provided", voices=["other"],
        audio_sha256=hashlib.sha256(source_bytes).hexdigest(),
        set_sha256="a" * 64, cached=True,
    )

    class Separator:
        def separate(self, path, work, progress, cancelled):
            return tmp_path, separation

    monkeypatch.setattr(audio, "decode", lambda *args, **kwargs: samples)
    monkeypatch.setattr(audio, "infer", lambda *args, **kwargs: _synthetic_activations())
    monkeypatch.setattr(audio, "load_stem_activations", lambda *args, **kwargs: None)
    saved_cancellations = []
    monkeypatch.setattr(
        audio, "save_stem_activations",
        lambda *args, **kwargs: saved_cancellations.append(kwargs.get("cancelled")),
    )
    monkeypatch.setattr(
        audio, "rhythmic_evidence",
        lambda *args, **kwargs: ([], {"bpm": 120.0, "offset": 0.0, "estimated": True}),
    )
    monkeypatch.setattr(audio, "pitched_notes", lambda *args, **kwargs: [])
    monkeypatch.setattr(audio, "source_evidence", lambda notes, *args: (notes, 0))
    monkeypatch.setattr(audio, "refine_releases", lambda notes, values: (notes, {}))
    monkeypatch.setattr(audio, "fuse_attacks", lambda notes, attacks: (notes, [], 0))
    monkeypatch.setattr(audio, "reject_source_bleed", lambda notes, *args: (notes, []))
    monkeypatch.setattr(
        "manialab.structure.detect_structure",
        lambda analysis, values, sample_rate, **kwargs: {"segments": []},
    )
    cancellation = lambda: False

    audio.analyze_deep(
        source, tmp_path, model,
        {"onset_threshold": .45, "frame_threshold": .3}, Separator(),
        cancelled=cancellation,
    )

    assert saved_cancellations == [cancellation]
