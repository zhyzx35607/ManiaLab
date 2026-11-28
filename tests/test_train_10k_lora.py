from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import zipfile

import pytest

from training.train_10k_lora import (
    chart_generation_metrics,
    active_hold_prefix_events,
    active_holds_before_window,
    append_metrics_curve,
    estimate_cuda_training_memory,
    build_sample_weights,
    cuda_training_profile,
    find_latest_checkpoint,
    load_dataset_manifest,
    manialab_dataloader_worker_init,
    read_sample_weight_table,
    resolve_mapperator_root,
    sample_ids_from_batch,
    should_stop_at_optimizer_boundary,
    summarize_sample_consumption,
    validate_dataset_review_gate,
    validate_options,
    weighted_draw_indices,
    ensure_free_disk_space,
    system_free_space_at_or_below_floor,
    ensure_system_free_space,
    package_progress_osz,
    require_repo_drive,
    _normalize_generated_zero_length_holds,
    _repair_conflicting_chart_for_torosu,
    write_adapter_manifest,
)
import training.train_10k_lora as trainer


def _chart() -> str:
    return """osu file format v14

[General]
Mode:3
AudioFilename:song.mp3

[Metadata]
Title:Smoke
Artist:Test

[Difficulty]
CircleSize:10
OverallDifficulty:7.4

[TimingPoints]
0,500,4,2,0,70,1,0

[HitObjects]
25,192,1000,128,0,3000:0:0:0:0:
76,192,1000,1,0,0:0:0:0:
486,192,4000,128,0,6000:0:0:0:0:
"""


def test_cuda_training_profile_enforces_6gb_profile():
    profile = cuda_training_profile(6 * 1024**3, True)

    assert profile["precision"] == "bf16"
    assert profile["gradient_checkpointing"] is True
    assert profile["max_safe_micro_batch_size"] == 1

    with pytest.raises(ValueError, match="at least 5 GiB"):
        cuda_training_profile(4 * 1024**3, True)
    with pytest.raises(ValueError, match="native bf16"):
        cuda_training_profile(6 * 1024**3, False)


class _EventType(Enum):
    TIME_SHIFT = "t"
    MANIA_COLUMN = "column"
    HOLD_NOTE = "hold_note"
    HOLD_NOTE_SUSTAIN = "hold_note_sustain"
    HOLD_NOTE_END = "hold_note_end"


@dataclass
class _Event:
    type: _EventType
    value: int = 0


@dataclass
class _Parameter:
    count: int
    bytes_per_value: int
    requires_grad: bool

    def numel(self):
        return self.count

    def element_size(self):
        return self.bytes_per_value


class _MemoryModel:
    def parameters(self):
        return iter(
            (
                _Parameter(count=1000, bytes_per_value=2, requires_grad=False),
                _Parameter(count=100000, bytes_per_value=4, requires_grad=True),
            )
        )


class _AdamW8bit:
    pass


def test_cuda_memory_budget_accounts_for_frozen_lora_and_workspace():
    result = estimate_cuda_training_memory(_MemoryModel(), _AdamW8bit())

    assert result["parameter_weight_mib"] == pytest.approx(0.38)
    assert result["trainable_parameter_count"] == 100000
    assert result["optimizer_state_mib_estimate"] == pytest.approx(0.19)
    assert result["gradient_mib_estimate"] == pytest.approx(0.38)
    assert result["activation_and_cuda_workspace_allowance_gib"] == 2.5
    assert result["estimate_is_lower_bound"] is True


def test_cross_window_hold_state_tracks_sustain_and_encodes_at_window_zero():
    events = [
        _Event(_EventType.TIME_SHIFT, 1000),
        _Event(_EventType.MANIA_COLUMN, 3),
        _Event(_EventType.HOLD_NOTE),
        _Event(_EventType.TIME_SHIFT, 17000),
        _Event(_EventType.MANIA_COLUMN, 3),
        _Event(_EventType.HOLD_NOTE_SUSTAIN),
        _Event(_EventType.TIME_SHIFT, 33000),
        _Event(_EventType.MANIA_COLUMN, 3),
        _Event(_EventType.HOLD_NOTE_SUSTAIN),
        _Event(_EventType.TIME_SHIFT, 40000),
        _Event(_EventType.MANIA_COLUMN, 3),
        _Event(_EventType.HOLD_NOTE_END),
    ]
    times = [1000, 1000, 1000, 17000, 17000, 17000, 33000, 33000, 33000, 40000, 40000, 40000]

    active = active_holds_before_window(events, times, 39000, _EventType)
    prefix = active_hold_prefix_events(active, _EventType, _Event, 39000)

    assert len(active) == 1
    assert active[0]["lane"] == 3
    assert active[0]["last_sustain_ms"] == 33000
    assert [(event.type, event.value) for event in prefix] == [
        (_EventType.TIME_SHIFT, 39000),
        (_EventType.MANIA_COLUMN, 3),
        (_EventType.HOLD_NOTE, 0),
        (_EventType.HOLD_NOTE_SUSTAIN, 0),
    ]
    assert active_holds_before_window(events, times, 40000, _EventType) == []


def test_ln_sample_weights_emphasize_ln_without_changing_real_labels(tmp_path):
    records = [
        {"split": "train", "bid": 1, "keycount": 10, "tier": 1, "bucket": "10k_ln"},
        {"split": "train", "bid": 2, "keycount": 10, "tier": 1, "bucket": "10k_hybrid"},
        {"split": "train", "bid": 3, "keycount": 10, "tier": 1, "bucket": "10k_rice"},
        {"split": "train", "bid": 4, "keycount": 7, "tier": 2, "bucket": "7k_ln_aux"},
        {"split": "train", "bid": 6, "keycount": 6, "tier": 3, "bucket": "6k_hybrid"},
        {"split": "train", "bid": 7, "keycount": 6, "tier": 3, "bucket": "6k_ln"},
        {"split": "val", "bid": 5, "keycount": 10, "bucket": "10k_ln"},
    ]

    result = build_sample_weights(records, tmp_path / "sample-weights.csv")

    table = read_sample_weight_table(tmp_path / "sample-weights.csv")
    rows = {key: row["weight"] for key, row in table.items()}
    assert rows[1] > rows[2] > rows[3]
    assert rows[4] < rows[1] and 6 in rows and 5 not in rows
    assert rows[1] > rows[7] > rows[6]
    assert result["normalized_mean"] == pytest.approx(1.0)
    assert table[4]["tier"] == 2
    assert table[4]["style"] == "ln"


def test_consumed_sample_report_uses_actual_beatmap_ids_and_tracks(tmp_path):
    records = [
        {"split": "train", "bid": 10, "keycount": 6, "tier": 1, "bucket": "6k_ln"},
        {"split": "train", "bid": 20, "keycount": 7, "tier": 3, "bucket": "7k_rice"},
    ]
    weight_path = tmp_path / "weights.csv"
    build_sample_weights(records, weight_path)
    table = read_sample_weight_table(weight_path)

    batch = {"manialab_beatmap_id": [10, 20, 20], "decoder_input_ids": "model input"}
    assert sample_ids_from_batch(batch) == [10, 20, 20]
    assert "manialab_beatmap_id" not in batch
    summary = summarize_sample_consumption({10: 1, 20: 2}, table)

    assert summary["sequences_consumed"] == 3
    assert summary["by_tier"] == {"tier1": 1, "tier3": 2}
    assert summary["by_style"] == {"ln": 1, "rice": 2}
    assert summary["by_keycount"] == {"6k": 1, "7k": 2}


def test_weighted_sampler_is_deterministic_and_changes_draw_frequency():
    draws = weighted_draw_indices([1.0, 9.0], seed=42, epoch=3, count=2000)
    repeat = weighted_draw_indices([1.0, 9.0], seed=42, epoch=3, count=2000)
    low_weight_share = draws.count(0) / len(draws)

    assert draws == repeat
    assert low_weight_share == pytest.approx(0.1, abs=0.025)
    assert weighted_draw_indices([1.0, 9.0], seed=42, epoch=4, count=20) != draws[:20]
    with pytest.raises(ValueError, match="positive finite"):
        weighted_draw_indices([1.0, 0.0], seed=1, epoch=1)


def test_smoke_mode_is_cpu_and_limited_to_two_steps():
    args = trainer.parse_args(
        [
            "--device", "cpu",
            "--smoke",
            "--max-steps", "2",
            "--generation-window-seconds", "20",
        ]
    )
    validate_options(args)

    with pytest.raises(ValueError, match="requires --device cpu"):
        validate_options(
            trainer.parse_args(["--smoke", "--max-steps", "1"])
        )
    with pytest.raises(ValueError, match="requires --device cpu"):
        validate_options(
            trainer.parse_args(["--device", "cpu", "--smoke", "--max-steps", "3"])
        )


def test_provisional_review_gate_and_sigint_wait_for_optimizer_boundary():
    smoke = trainer.parse_args(
        ["--device", "cpu", "--smoke", "--max-steps", "1"]
    )
    reviewed = trainer.parse_args(["--device", "cpu"])

    validate_dataset_review_gate("provisional", smoke)
    validate_dataset_review_gate("human_reviewed", reviewed)
    with pytest.raises(ValueError, match="dataset is provisional"):
        validate_dataset_review_gate("provisional", reviewed)
    assert should_stop_at_optimizer_boundary(
        stop_requested=True,
        max_steps=None,
        global_step=1,
        sync_gradients=False,
    ) is False
    assert should_stop_at_optimizer_boundary(
        stop_requested=True,
        max_steps=None,
        global_step=1,
        sync_gradients=True,
    ) is True
    assert should_stop_at_optimizer_boundary(
        stop_requested=False,
        max_steps=2,
        global_step=2,
        sync_gradients=True,
    ) is True


def test_metrics_curve_appends_training_and_generation_measurements(tmp_path):
    path = tmp_path / "metrics-curve.csv"
    append_metrics_curve(path, [{"global_step": 1, "epoch": 1, "kind": "train", "event_nll": 0.8}])
    append_metrics_curve(
        path,
        [{"global_step": 1, "epoch": 1, "kind": "generation", "peak_held": 2, "pinky_balance": 0.9}],
    )

    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("global_step,epoch,kind,event_nll")
    assert len(rows) == 3
    assert "0.8" in rows[1]
    assert "generation" in rows[2]


def test_generated_chart_metrics_report_ln_and_outer_lane_balance():
    metrics = chart_generation_metrics(
        _chart(),
        0.6,
        {"minijack": 2, "pill": 3, "hand_balance": 0.9},
    )

    assert metrics["objects"] == 3
    assert metrics["hold_notes"] == 2
    assert metrics["actual_hold_note_ratio"] == pytest.approx(2 / 3)
    assert metrics["hold_ratio_error"] == pytest.approx(1 / 15)
    assert metrics["peak_held"] == 1
    assert metrics["difficulty"] == pytest.approx(7.4)
    assert metrics["minijack"] == 2
    assert metrics["pill"] == 3
    assert metrics["pinky_balance"] == 1


@pytest.mark.parametrize("keycount", (6, 7, 8, 9, 10))
def test_generated_chart_metrics_accept_every_supported_keycount(keycount):
    chart = _chart().replace("CircleSize:10", f"CircleSize:{keycount}")
    chart = chart.replace("76,192,", f"{int(1.5 * 512 / keycount)},192,")

    assert chart_generation_metrics(chart, 0.6, {})["keycount"] == keycount


def test_generated_chart_metrics_reject_wrong_keycount():
    with pytest.raises(ValueError, match="must be 6K–10K"):
        chart_generation_metrics(_chart().replace("CircleSize:10", "CircleSize:5"), 0.5, {})


def test_progress_osz_is_written_under_requested_kit_and_crc_checked(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"real-song-audio-sample")
    output = tmp_path / "progress_osz"
    record = {
        "keycount": 10,
        "tier": 1,
        "set_id": 22,
        "bid": 33,
        "bucket": "10k_ln",
    }

    result = package_progress_osz(
        _chart(),
        audio,
        destination_root=output,
        step=2,
        epoch=1,
        record=record,
        torosu={"minijack": 1, "pill": 0},
        target_hold_ratio=0.6,
        minimum_free_gib=0.1,
    )

    assert Path(result["path"]).parent == output
    with zipfile.ZipFile(result["path"]) as archive:
        assert archive.testzip() is None
        assert "song.mp3" in archive.namelist()
        assert any(name.endswith(".osu") for name in archive.namelist())


def test_progress_osz_never_reuses_stale_chart_for_same_song_step(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"real-song-audio-sample")
    output = tmp_path / "progress_osz"
    record = {
        "keycount": 10,
        "tier": 1,
        "set_id": 22,
        "bid": 33,
        "bucket": "10k_ln",
    }
    first_chart = _chart()
    second_chart = first_chart.replace("76,192,1000,1", "150,192,1000,1")
    options = {
        "destination_root": output,
        "step": 2,
        "epoch": 1,
        "record": record,
        "torosu": {"minijack": 1, "pill": 0},
        "target_hold_ratio": 0.6,
        "minimum_free_gib": 0.1,
    }

    first = package_progress_osz(first_chart, audio, **options)
    second = package_progress_osz(second_chart, audio, **options)

    assert first["reused"] is False
    assert second["reused"] is False
    assert "-attempt-" in Path(second["path"]).name
    with zipfile.ZipFile(second["path"]) as archive:
        chart_name = next(name for name in archive.namelist() if name.endswith(".osu"))
        assert b"150,192,1000,1" in archive.read(chart_name)


def test_same_key_preview_repair_deduplicates_lane_stacks_and_truncates_prior_ln(tmp_path):
    from manialab.domain import parse_osu, validate

    source = tmp_path / "stacked-10k.osu"
    chart = _chart().replace(
        "[HitObjects]\n",
        "[HitObjects]\n"
        "25,192,1500,1,0,0:0:0:0:\n"
        "25,192,1500,1,0,0:0:0:0:\n",
    )
    source.write_text(chart, encoding="utf-8")

    repaired_path, changes = _repair_conflicting_chart_for_torosu(source, 10)
    repaired = parse_osu(repaired_path.read_text(encoding="utf-8"))
    validation = validate(repaired["notes"], repaired["keys"])

    assert validation["valid"] is True
    assert len(repaired["notes"]) == 4
    assert changes == {"removed_objects": 1, "truncated_holds": 1}
    assert any(note.hold and note.end == 1500 for note in repaired["notes"])


@pytest.mark.parametrize("hold_end", (1000, 900))
def test_generated_zero_or_negative_length_holds_are_preserved_as_taps(tmp_path, hold_end):
    from manialab.domain import parse_osu

    source = tmp_path / "generated.osu"
    source.write_text(
        _chart().replace("25,192,1000,128,0,3000:", f"25,192,1000,128,0,{hold_end}:"),
        encoding="utf-8",
    )

    normalized, converted = _normalize_generated_zero_length_holds(source)
    parsed = parse_osu(normalized.read_text(encoding="utf-8"))

    assert converted == 1
    assert normalized != source
    assert parsed["notes"][0].start == 1000
    assert parsed["notes"][0].end == 1000
    assert parsed["notes"][0].hold is False
    repeated_path, repeated_count = _normalize_generated_zero_length_holds(source)
    assert repeated_path == normalized
    assert repeated_count == 1


def test_free_disk_guard_and_runtime_paths_are_portable(tmp_path, monkeypatch):
    assert ensure_free_disk_space(tmp_path, 0.001) > 0
    monkeypatch.setattr(trainer, "repo_root", lambda: Path("D:/ManiaLab"))
    assert require_repo_drive(Path("D:/experiment"), "output").drive == "D:"
    assert require_repo_drive(Path("C:/temp"), "output").drive == "C:"
    assert trainer.resolve_model_cache(
        trainer.parse_args(["--model-cache", str(tmp_path / "portable-models")]), tmp_path
    ) == (tmp_path / "portable-models").resolve()
    portable_runtime = tmp_path / "kit-on-another-drive"
    paths = trainer.pin_runtime_paths(portable_runtime)
    assert paths["HF_HOME"] == portable_runtime / "runtime" / "cache" / "huggingface"
    assert paths["TMP"].is_dir()


def test_runtime_paths_preserve_selected_offline_whisper_cache(tmp_path, monkeypatch):
    whisper_cache = tmp_path / "assets" / "huggingface"
    monkeypatch.setenv("HF_HUB_CACHE", str(whisper_cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(whisper_cache))

    paths = trainer.pin_runtime_paths(tmp_path / "portable")

    assert paths["HF_HUB_CACHE"] == whisper_cache
    assert Path(os.environ["TRANSFORMERS_CACHE"]) == whisper_cache
    assert whisper_cache.is_dir()


def test_portable_defaults_use_4050_safe_checkpoint_cadence():
    args = trainer.parse_args([])
    assert args.model == "v32-mini"
    assert args.gradient_accumulation_steps == 16
    assert args.checkpoint_every_steps == 50


@pytest.mark.parametrize("free_gib", [5.0, 4.99])
def test_direct_training_system_space_guard_stops_at_or_below_five_gib(
    monkeypatch,
    free_gib,
):
    monkeypatch.setenv("SystemDrive", "C:")
    monkeypatch.setattr(
        trainer.shutil,
        "disk_usage",
        lambda path: type("DiskUsage", (), {"free": int(free_gib * 1024**3)})(),
    )

    with pytest.raises(OSError, match="free-space floor"):
        ensure_system_free_space()


def test_direct_training_system_space_guard_accepts_space_above_floor(
    monkeypatch,
):
    monkeypatch.setenv("SystemDrive", "C:")
    free = int(5.01 * 1024**3)
    monkeypatch.setattr(
        trainer.shutil,
        "disk_usage",
        lambda path: type("DiskUsage", (), {"free": free})(),
    )

    assert ensure_system_free_space() == free


@pytest.mark.parametrize("free_gib,expected", [
    (5.01, False),
    (5.0, True),
    (4.99, True),
    (None, False),
])
def test_optimizer_disk_watch_stops_at_five_gib_floor(free_gib, expected):
    free_bytes = None if free_gib is None else int(free_gib * 1024**3)
    assert system_free_space_at_or_below_floor(free_bytes) is expected


@pytest.mark.parametrize("free_gib,should_stop", [
    (5.01, False),
    (5.0, True),
    (4.99, True),
    (None, False),
])
def test_training_requests_checkpoint_stop_at_c_drive_floor(free_gib, should_stop):
    free_bytes = None if free_gib is None else int(free_gib * 1024**3)
    assert system_free_space_at_or_below_floor(free_bytes) is should_stop


def test_manifest_loader_requires_zero_song_audio_and_set_leakage(tmp_path):
    records = []
    for split, sid in (("train", 1), ("val", 2)):
        for keycount in trainer.SUPPORTED_KEYCOUNTS:
            chart = tmp_path / split / f"{keycount}k-chart.osu"
            audio = tmp_path / split / f"{keycount}k-audio.ogg"
            chart.parent.mkdir(exist_ok=True)
            chart.write_text("osu", encoding="utf-8")
            audio.write_bytes(b"audio")
            records.append(
                {
                    "split": split,
                    "keycount": keycount,
                    "tier": 1 if keycount == 10 else 3,
                    "difficulty": 7.0,
                    "hold_note_ratio": 0.5,
                    "audio_sha256": f"audio-{sid}-{keycount}",
                    "set_id": sid * 10 + keycount,
                    "leakage_group_id": f"group-{sid}-{keycount}",
                    "audio_relpath": f"{split}/{audio.name}",
                    "data_relpath": f"{split}/{chart.name}",
                }
            )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"review_status": "provisional", "summary": {}, "records": records}),
        encoding="utf-8",
    )

    manifest, selected = load_dataset_manifest(tmp_path)
    assert len(selected) == 10
    assert manifest["review_status"] == "provisional"
    assert manifest["summary"]["upstream_ranked_statuses"] == [0, 1, 4]

    records[6]["leakage_group_id"] = records[0]["leakage_group_id"]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"review_status": "provisional", "summary": {}, "records": records}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leaks a song"):
        load_dataset_manifest(tmp_path)


def test_manifest_review_gate_rejects_missing_status(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"summary": {}, "records": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="review_status"):
        load_dataset_manifest(tmp_path)


def test_manifest_loader_recomputes_song_alias_groups_across_splits(tmp_path):
    records = []
    alias = "artist\0shared chinese title"
    for split, sid in (("train", 1), ("val", 2)):
        for keycount in trainer.SUPPORTED_KEYCOUNTS:
            chart = tmp_path / split / f"{keycount}k-chart.osu"
            audio = tmp_path / split / f"{keycount}k-audio.ogg"
            chart.parent.mkdir(exist_ok=True)
            chart.write_text("osu", encoding="utf-8")
            audio.write_bytes(b"audio")
            records.append(
                {
                    "split": split,
                    "keycount": keycount,
                    "tier": 1 if keycount == 10 else 3,
                    "difficulty": 7.0,
                    "hold_note_ratio": 0.5,
                    "audio_sha256": f"audio-{sid}-{keycount}",
                    "set_id": sid * 10 + keycount,
                    "leakage_group_id": f"group-{sid}-{keycount}",
                    "SongKeyAliases": [alias] if keycount == 7 else [],
                    "audio_relpath": f"{split}/{audio.name}",
                    "data_relpath": f"{split}/{chart.name}",
                }
            )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"review_status": "provisional", "summary": {}, "records": records}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="song alias"):
        load_dataset_manifest(tmp_path)


def test_free_generation_samples_distinct_songs_and_prioritizes_ln(tmp_path):
    records = [
        {
            "split": "val",
            "keycount": 10,
            "set_id": 1,
            "bid": 10,
            "audio_sha256": "song-a",
            "artist": "Artist A",
            "title": "Song A",
            "bucket": "10k_rice",
            "hold_note_ratio": 0.02,
            "difficulty": 2.0,
        },
        {
            "split": "val",
            "keycount": 10,
            "set_id": 1,
            "bid": 11,
            "audio_sha256": "song-a",
            "artist": "Artist A",
            "title": "Song A",
            "bucket": "10k_ln",
            "hold_note_ratio": 0.95,
            "difficulty": 4.0,
        },
        {
            "split": "val",
            "keycount": 10,
            "set_id": 2,
            "bid": 20,
            "audio_sha256": "song-b",
            "artist": "Artist B",
            "title": "Song B",
            "bucket": "10k_rice",
            "hold_note_ratio": 0.0,
            "difficulty": 3.0,
        },
        {
            "split": "val",
            "keycount": 10,
            "set_id": 3,
            "bid": 30,
            "audio_sha256": "song-c",
            "artist": "Artist C",
            "title": "Song C",
            "bucket": "10k_hybrid",
            "hold_note_ratio": 0.25,
            "difficulty": 3.5,
        },
        {
            "split": "train",
            "keycount": 10,
            "set_id": 4,
            "bid": 40,
            "audio_sha256": "song-d",
            "artist": "Artist D",
            "title": "Song D",
            "bucket": "10k_ln",
            "hold_note_ratio": 0.8,
            "difficulty": 4.0,
        },
        {
            "split": "val",
            "keycount": 10,
            "set_id": 5,
            "bid": 50,
            "audio_sha256": "song-e",
            "artist": "Artist E",
            "title": "Song E",
            "bucket": "10k_ln",
            "hold_note_ratio": 0.99,
            "difficulty": 5.0,
        },
    ]

    selected = trainer._resolve_validation_records(tmp_path, records, 2)

    assert [record["bid"] for record in selected] == [50, 30]
    assert [record["bucket"] for record in selected] == ["10k_ln", "10k_hybrid"]
    assert len({record["audio_sha256"] for record in selected}) == 2
    assert len({record["set_id"] for record in selected}) == 2


def test_v32_validation_caps_only_the_engine_argument_for_high_ln_targets():
    assert trainer._v32_engine_hold_ratio(0.72) == pytest.approx(0.72)
    assert trainer._v32_engine_hold_ratio(0.947) == pytest.approx(0.85)
    with pytest.raises(ValueError, match="between 0 and 1"):
        trainer._v32_engine_hold_ratio(1.01)


def test_free_generation_cache_identity_tracks_weights_inputs_and_parameters(tmp_path):
    adapter = tmp_path / "adapter"
    model_cache = tmp_path / "model-cache"
    model_root = model_cache / "Mapperatorinator-v32-mini-7807f0d"
    adapter.mkdir()
    model_root.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    revision = model_root / "model-revision.json"
    chart = tmp_path / "reference.osu"
    audio = tmp_path / "audio.mp3"
    weights.write_bytes(b"weights-1")
    config.write_text("{}", encoding="utf-8")
    revision.write_text(json.dumps({"revision": "fixed-revision"}), encoding="utf-8")
    chart.write_text("reference-1", encoding="utf-8")
    audio.write_bytes(b"audio-1")

    def identity(*, difficulty=7.0):
        return trainer._free_generation_cache_identity(
            adapter_dir=adapter,
            model_cache=model_cache,
            reference=chart,
            audio=audio,
            model_variant="v32-mini",
            keycount=10,
            difficulty=difficulty,
            target_hold_note_ratio=0.95,
            engine_hold_note_ratio=0.85,
            device="cpu",
            start_ms=1000,
            end_ms=21000,
            seed=42,
        )

    original = identity()
    weights.write_bytes(b"weights-2")
    assert identity() != original
    weights.write_bytes(b"weights-1")
    chart.write_text("reference-2", encoding="utf-8")
    assert identity() != original
    chart.write_text("reference-1", encoding="utf-8")
    assert identity(difficulty=7.1) != original


def test_free_generation_cache_finds_only_matching_attempts(tmp_path):
    sample_root = tmp_path / "free-generation"
    adapter = tmp_path / "adapter"
    sample = sample_root / "sample-01-attempt-02"
    adapter.mkdir()
    sample.mkdir(parents=True)
    generated_map = sample / "chart.osu"
    generated_map.write_text("generated chart", encoding="utf-8")
    request = {"adapter_sha256": "new-weights", "seed": 42}
    report = {
        "map": str(generated_map),
        "model_variant": "v32-mini",
        "lora_loaded": True,
        "lora_path": str(adapter),
    }
    (sample / "run-report.json").write_text(json.dumps(report), encoding="utf-8")
    (sample / "manialab-cache.json").write_text(
        json.dumps({
            "request": request,
            "map_sha256": trainer._sha256_file(generated_map),
        }),
        encoding="utf-8",
    )

    cached = trainer._find_free_generation_cache(
        sample_root,
        1,
        request,
        adapter,
        "v32-mini",
    )
    assert cached is not None
    assert cached[0] == sample
    assert cached[1]["map"] == generated_map
    assert trainer._find_free_generation_cache(
        sample_root,
        1,
        {"adapter_sha256": "old-weights", "seed": 42},
        adapter,
        "v32-mini",
    ) is None


def test_latest_checkpoint_prefers_epoch_end_checkpoint(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    checkpoints = (
        ("checkpoint-00000001", 1, 1, False),
        ("checkpoint-00000001-epoch-0001", 1, 1, True),
        ("checkpoint-00000002", 2, 1, False),
    )
    for name, step, epoch, complete in checkpoints:
        path = checkpoint_root / name
        path.mkdir()
        (path / "train-state.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "epoch": epoch,
                    "epoch_complete": complete,
                }
            ),
            encoding="utf-8",
        )

    assert find_latest_checkpoint(tmp_path).name == "checkpoint-00000002"


def test_checkpoint_persists_batch_cursor_and_prunes_only_after_save(tmp_path):
    class Model:
        def save_pretrained(self, destination, safe_serialization=True):
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "adapter_model.safetensors").write_bytes(b"adapter")

    class Accelerator:
        is_main_process = True

        def wait_for_everyone(self):
            return None

        def save_state(self, destination, safe_serialization=False):
            Path(destination, "accelerator-state.bin").write_bytes(b"state")

        def unwrap_model(self, model):
            return model

    args = trainer.parse_args(["--device", "cpu"])
    root = tmp_path / "checkpoints"
    root.mkdir()
    for cursor in (5, 8):
        _write_path = root / f"checkpoint-00000003-cursor-{cursor:04d}"
        trainer._write_checkpoint(
            Accelerator(),
            Model(),
            _write_path,
            global_step=3,
            epoch=1,
            args=args,
            model_spec={"repo": "example/model", "revision": "abc123"},
            epoch_complete=False,
            cross_window_context_seen=False,
            training_seconds=12.5,
            optimizer_steps=3,
            peak_rss_bytes=456,
            epoch_batch_cursor=cursor,
            sample_consumption={42: 4},
            dataset_manifest_sha256="manifest-hash",
            sample_weighting={"sequences_consumed": 4},
        )

    latest = find_latest_checkpoint(tmp_path)
    state = trainer._read_checkpoint_state(latest)
    assert latest.name.endswith("cursor-0008")
    assert state["epoch_batch_cursor"] == 8
    assert state["sample_consumption"] == {"42": 4}
    assert state["dataset_manifest_sha256"] == "manifest-hash"

    removed = trainer.prune_checkpoints(tmp_path, keep=1)
    assert len(removed) == 1
    assert find_latest_checkpoint(tmp_path) == latest


def test_adapter_manifest_pins_model_variant_and_revision(tmp_path):
    from argparse import Namespace

    args = Namespace(
        model="v32-mini",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
    )
    model_spec = {
        "repo": "OliBomby/Mapperatorinator-v32-mini",
        "revision": "7807f0dc70cab671be012e1f5ddf945b0b8b7278",
    }
    path = write_adapter_manifest(
        tmp_path,
        model_spec=model_spec,
        args=args,
        global_step=2,
        epoch=1,
        validation={"event_nll": 0.3},
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["base_model"]["variant"] == "v32-mini"
    assert manifest["base_model"]["gamemode"] == 3
    assert manifest["base_model"]["revision"] == model_spec["revision"]


def test_dataloader_worker_initializer_is_spawn_picklable():
    assert manialab_dataloader_worker_init.__module__ == "training.train_10k_lora"
    assert callable(manialab_dataloader_worker_init)


def test_unmarked_portable_upstream_source_is_preserved(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    kit = fake_root / "workspace_exp" / "lora_10k_kit"
    source = kit / "runtime" / "source"
    source.mkdir(parents=True)
    inference = source / "inference.py"
    inference.write_text("# user-provided source\n", encoding="utf-8")
    monkeypatch.setattr(trainer, "repo_root", lambda: fake_root)
    monkeypatch.setattr(trainer, "kit_root", lambda: kit)
    monkeypatch.delenv("MANIALAB_MAPPER_REPO", raising=False)

    with pytest.raises(ValueError, match="no ManiaLab marker"):
        resolve_mapperator_root()

    assert inference.read_text(encoding="utf-8") == "# user-provided source\n"
