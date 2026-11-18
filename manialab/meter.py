"""Meter for a single fixed audio clock, expressed as beats per measure."""
import math


def valid_audio_clock(tempo):
    if not isinstance(tempo, dict):
        return False
    bpm, offset = tempo.get("bpm"), tempo.get("offset")
    return (type(bpm) in (int, float) and 10 <= bpm <= 1000
            and type(offset) in (int, float) and -600000 <= offset <= 600000)


def stable_audio_clock(tempo):
    revision = tempo.get("timing_coverage_revision", 0)
    return tempo.get("stable") is True and (tempo.get("evidence_source") != "drums"
                                           or (type(revision) is int and revision >= 1))


def _indices(times, offset, period, tolerance):
    if not isinstance(times, (list, tuple)):
        return []
    result = []
    for time_ms in times:
        if type(time_ms) not in (int, float) or not 0 <= time_ms <= 24 * 3600 * 1000:
            return []
        beat = round((time_ms - offset) / period)
        if abs(time_ms - (offset + beat * period)) > tolerance or (result and beat <= result[-1]):
            return []
        result.append(beat)
    return result


def detected_meter(analysis):
    """Require at least three complete, consistent bars on the detected audio clock."""
    tempo = analysis.get("tempo") if isinstance(analysis, dict) else None
    if not valid_audio_clock(tempo) or not stable_audio_clock(tempo):
        return 4
    separation = analysis.get("separation")
    evidence = separation.get("beat_evidence") if isinstance(separation, dict) else None
    if not isinstance(evidence, dict):
        return 4
    period, offset = 60000 / tempo["bpm"], tempo["offset"]
    tolerance = min(30, period / 8)
    beats = _indices(evidence.get("beats_ms"), offset, period, tolerance)
    downbeats = _indices(evidence.get("downbeats_ms"), offset, period, tolerance)
    if len(downbeats) < 4 or not beats or beats[-1] - beats[0] + 1 != len(beats):
        return 4
    meter = downbeats[1] - downbeats[0]
    if not 2 <= meter <= 12 or downbeats != [beat for beat in beats if beat % meter == 0]:
        return 4
    for index, time_ms in zip(downbeats, evidence["downbeats_ms"]):
        if abs(time_ms - evidence["beats_ms"][index - beats[0]]) > tolerance:
            return 4
    # A short island of regular beats cannot establish a whole song's fixed meter.
    duration = analysis.get("duration")
    if duration is not None:
        if type(duration) not in (int, float) or not 0 < duration <= 24 * 3600:
            return 4
        start, end = evidence["beats_ms"][0], evidence["beats_ms"][-1]
        if start > meter * period + tolerance or not -tolerance <= duration * 1000 - end <= meter * period + tolerance:
            return 4
    return meter


def meter_for_clock(analysis, clock):
    """Beat evidence is on the audio clock. A chart's red line includes the mapping shift."""
    detected = analysis.get("tempo") if isinstance(analysis, dict) else None
    if not valid_audio_clock(detected) or not valid_audio_clock(clock):
        return 4
    shift = clock.get("audio_shift_ms", 0)
    if type(shift) not in (int, float) or not -100 <= shift <= 100:
        return 4
    if (clock["bpm"] != detected["bpm"] or
            clock["offset"] != math.floor(detected["offset"] - shift + .5)):
        return 4
    return detected_meter(analysis)


def project_with_meter(state):
    """Expose the draft meter without rewriting saved analysis or historical candidates."""
    return dict(state, detected_meter=detected_meter(state.get("analysis") or {}))
