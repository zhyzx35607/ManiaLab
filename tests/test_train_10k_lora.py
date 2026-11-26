from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import pytest

from training.train_10k_lora import (
    chart_generation_metrics,
    active_hold_prefix_events,
    active_holds_before_window,
    estimate_cuda_training_memory,
    build_sample_weights,
    cuda_training_profile,
    find_latest_checkpoint,
    load_dataset_manifest,
    manialab_dataloader_worker_init,
    resolve_mapperator_root,
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
        {"split": "train", "bid": 1, "keycount": 10, "bucket": "10k_ln"},
        {"split": "train", "bid": 2, "keycount": 10, "bucket": "10k_hybrid"},
        {"split": "train", "bid": 3, "keycount": 10, "bucket": "10k_rice"},
        {"split": "train", "bid": 4, "keycount": 7, "bucket": "7k_ln_aux"},
        {"split": "val", "bid": 5, "keycount": 10, "bucket": "10k_ln"},
    ]

    result = build_sample_weights(records, tmp_path / "sample-weights.csv")

    rows = {
        int(line.split(",")[0]): float(line.split(",")[1])
        for line in (tmp_path / "sample-weights.csv").read_text(encoding="ascii").splitlines()
    }
    assert rows[1] > rows[2] > rows[3]
    assert 4 in rows and 5 not in rows
    assert result["normalized_mean"] == pytest.approx(1.0)


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


def test_generated_chart_metrics_reject_wrong_keycount():
    with pytest.raises(ValueError, match="must be 10K"):
        chart_generation_metrics(_chart().replace("CircleSize:10", "CircleSize:7"), 0.5, {})


def test_manifest_loader_requires_zero_song_audio_and_set_leakage(tmp_path):
    records = []
    for split, sid in (("train", 1), ("val", 2)):
        chart = tmp_path / split / "chart.osu"
        audio = tmp_path / split / "audio.ogg"
        chart.parent.mkdir()
        chart.write_text("osu", encoding="utf-8")
        audio.write_bytes(b"audio")
        records.append(
            {
                "split": split,
                "keycount": 10,
                "difficulty": 7.0,
                "hold_note_ratio": 0.5,
                "audio_sha256": f"audio-{sid}",
                "set_id": sid,
                "leakage_group_id": f"group-{sid}",
                "audio_relpath": f"{split}/audio.ogg",
                "data_relpath": f"{split}/chart.osu",
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"summary": {}, "records": records}),
        encoding="utf-8",
    )

    _, selected = load_dataset_manifest(tmp_path)
    assert len(selected) == 2

    records[1]["leakage_group_id"] = records[0]["leakage_group_id"]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"summary": {}, "records": records}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leaks a song"):
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
            "hold_note_ratio": 0.72,
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
    ]

    selected = trainer._resolve_validation_records(tmp_path, records, 2)

    assert [record["bid"] for record in selected] == [11, 30]
    assert len({record["audio_sha256"] for record in selected}) == 2
    assert len({record["set_id"] for record in selected}) == 2


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
