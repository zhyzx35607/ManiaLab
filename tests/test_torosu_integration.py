from dataclasses import replace

import pytest

from manialab import arrange_config, generation
from manialab.arrange import assign_lanes
from manialab.domain import Note
from manialab.torosu_bridge import TorosuBridgeError


@pytest.mark.parametrize("keys,layout,playstyle", [
    (6, "6k_3_3", "3+3"),
    (7, "7k_3_1_3", "3+1+3"),
    (8, "8k_3_2_3", "3+2+3"),
    (9, "9k_4_1_4", "4+1+4"),
    (10, "10k_5_5", "5+5"),
])
def test_torosu_parameters_map_canonical_profiles(keys, layout, playstyle):
    prefs = arrange_config.parameters({"keys": keys, "layout_engine": "torosu"})

    assert prefs["key_layout"] == layout
    assert arrange_config.TOROSU_PLAYSTYLES[layout] == playstyle


def test_torosu_rejects_unsupported_eight_key_four_four_profile():
    with pytest.raises(ValueError, match="Torosu 不支持 8K 4\\+4"):
        arrange_config.parameters({
            "keys": 8, "key_layout": "8k_4_4", "layout_engine": "torosu",
        })


def test_torosu_assignment_passes_source_clock_and_strict_contract(monkeypatch):
    note = Note(1000, 1500, lane=1, id="hold")
    observed = {}

    def convert(notes, keys, **kwargs):
        observed.update(keys=keys, notes=list(notes), **kwargs)
        return [replace(notes[0], lane=4)]

    monkeypatch.setattr("manialab.torosu_bridge.convert_notes", convert)
    output = assign_lanes(
        [note], {"keys": 8, "layout_engine": "torosu"}, source_keys=6,
        timing_points=["0,500,4,1,0,70,1,0"], beat_ms=500, offset_ms=-20, meter=3,
        cancelled=lambda: False,
    )

    assert output == [replace(note, lane=4)]
    assert observed["keys"] == 8
    assert observed["notes"] == [note]
    assert {key: observed[key] for key in (
        "source_keys", "timing_points", "beat_ms", "offset_ms", "meter",
        "playstyle", "strict",
    )} == {
        "source_keys": 6,
        "timing_points": ["0,500,4,1,0,70,1,0"],
        "beat_ms": 500,
        "offset_ms": -20,
        "meter": 3,
        "playstyle": "3+2+3",
        "strict": True,
    }
    assert observed["cancelled"]() is False


def test_torosu_errors_and_cancellation_never_fall_back(monkeypatch):
    note = Note(1000, 1000, lane=0, id="tap")
    calls = []

    def fail(*_args, **_kwargs):
        calls.append("torosu")
        raise TorosuBridgeError("native core unavailable")

    monkeypatch.setattr("manialab.torosu_bridge.convert_notes", fail)
    with pytest.raises(TorosuBridgeError, match="native core unavailable"):
        assign_lanes([note], {"keys": 6, "layout_engine": "torosu"})
    assert calls == ["torosu"]

    with pytest.raises(ValueError, match="sequence prior"):
        assign_lanes(
            [note], {"keys": 6, "layout_engine": "torosu"},
            sequence_prior={},
        )
    with pytest.raises(InterruptedError, match="已取消"):
        assign_lanes(
            [note], {"keys": 6, "layout_engine": "torosu"},
            cancelled=lambda: True,
        )
    with pytest.raises(ValueError, match="锁定音符"):
        assign_lanes(
            [note], {"keys": 6, "layout_engine": "torosu"},
            locked=[replace(note, id="locked")],
        )


def test_remap_keeps_original_lanes_and_passes_input_timing_to_torosu(monkeypatch):
    notes = [
        Note(1000, 1600, lane=0, id="hold", raw="42,192,1000,128,0,1600:0:0:0:0:"),
        Note(2000, 2000, lane=5, id="tap", raw="469,192,2000,1,0,0:0:0:0:"),
    ]
    timing = ["0,500,4,1,0,70,1,0", "3000,-50,4,1,0,70,0,0"]
    state = dict(
        input_notes=[note.data() for note in notes], input_keys=6, input_timing=timing,
        variants=[], active=-1,
    )
    prefs = arrange_config.parameters({"keys": 10, "layout_engine": "torosu"})
    observed = {}

    def capture(selected, _prefs, _locked, **kwargs):
        observed["notes"] = list(selected)
        observed.update(kwargs)
        return [replace(note, lane=index) for index, note in enumerate(selected)]

    monkeypatch.setattr(generation, "assign_lanes", capture)
    generation.generate_variant(
        state, {"mode": "remap"}, prefs,
        {"bpm": 120, "offset": -35, "division": 4, "audio_shift_ms": 0},
        lambda *_: None, lambda: False,
    )

    assert [(note.lane, note.start, note.end, note.raw) for note in observed["notes"]] == [
        (0, 1000, 1600, notes[0].raw),
        (5, 2000, 2000, notes[1].raw),
    ]
    assert observed["source_keys"] == 6
    assert observed["timing_points"] == timing
    assert observed["offset_ms"] == -35
    assert observed["meter"] == 4
    assert [(n["id"], n["lane"], n["start"], n["end"], n["raw"])
            for n in state["variants"][-1]["notes"]] == [
        ("hold", 0, 1000, 1600, notes[0].raw),
        ("tap", 1, 2000, 2000, notes[1].raw),
    ]


def test_torosu_generation_rejects_regional_rewrite_before_mutating_state():
    state = dict(variants=[{"candidate_id": "existing"}], active=0)
    prefs = arrange_config.parameters({"layout_engine": "torosu"})

    with pytest.raises(ValueError, match="不支持局部重写"):
        generation.generate_variant(
            state, {"mode": "remap", "region": [1000, 2000]}, prefs,
            {"bpm": 120, "offset": 0, "division": 4, "audio_shift_ms": 0},
            lambda *_: None, lambda: False,
        )

    assert state["variants"] == [{"candidate_id": "existing"}]
    assert state["active"] == 0


def test_audio_generation_passes_effective_offset_and_detected_meter(monkeypatch):
    note = Note(1000, 1000, source="pitch", voice="piano", id="note")
    state = dict(
        analysis=dict(
            revision=13, duration=5, candidates=[note.data()], phrase_candidates=[],
            timing_attacks=[], structure=dict(segments=[dict(start_ms=0, end_ms=5000)]),
        ),
        variants=[], active=-1,
    )
    prefs = arrange_config.parameters({"layout_engine": "torosu", "ln_ratio": 0})
    observed = {}

    def capture(selected, _prefs, _locked, **kwargs):
        observed.update(kwargs)
        return [replace(selected[0], lane=0)]

    monkeypatch.setattr(generation, "assign_lanes", capture)
    generation.generate_variant(
        state, {"mode": "audio", "timing_confirmed": True}, prefs,
        {"bpm": 120, "offset": 0, "division": 4, "audio_shift_ms": 0, "meter": 3},
        lambda *_: None, lambda: False,
    )

    assert observed["source_keys"] is None
    assert observed["offset_ms"] == 0
    assert observed["meter"] == 3
    assert state["variants"][-1]["meter"] == 3
