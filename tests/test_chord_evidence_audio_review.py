from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from scripts.build_chord_evidence_audio_review import (
    _write_audition,
    build_review_cases,
)
from scripts.summarize_chord_evidence_audio_review import summarize_review_labels


def _fixture():
    candidates = [
        {
            "id": "drums:old",
            "voice": "drums",
            "start": 980,
            "end": 980,
            "evidence_start": 990,
            "pitch": 48,
            "source": "transient",
            "strength": 0.7,
        },
        {
            "id": "other:new",
            "voice": "other",
            "start": 995,
            "end": 1200,
            "evidence_start": 1010,
            "pitch": 60,
            "source": "pitch",
            "strength": 1.0,
        },
        {
            "id": "other:old",
            "voice": "other",
            "start": 1970,
            "end": 2200,
            "evidence_start": 1980,
            "pitch": 62,
            "source": "pitch",
            "strength": 1.0,
        },
        {
            "id": "bass:new",
            "voice": "bass",
            "start": 1985,
            "end": 2200,
            "evidence_start": 1990,
            "pitch": 45,
            "source": "pitch",
            "strength": 0.8,
        },
    ]
    old_selection = {
        "decisions": [
            {"id": "drums:old", "voice": "drums", "decision": "selected", "reason": "onsets", "chart_start": 1000},
            {"id": "other:new", "voice": "other", "decision": "omitted", "reason": "capacity"},
            {"id": "other:old", "voice": "other", "decision": "selected", "reason": "support", "chart_start": 2000},
            {"id": "bass:new", "voice": "bass", "decision": "omitted", "reason": "capacity"},
        ]
    }
    new_selection = {
        "decisions": [
            {"id": "drums:old", "voice": "drums", "decision": "omitted", "reason": "capacity"},
            {"id": "other:new", "voice": "other", "decision": "selected", "reason": "support", "chart_start": 1000},
            {"id": "other:old", "voice": "other", "decision": "omitted", "reason": "capacity"},
            {"id": "bass:new", "voice": "bass", "decision": "selected", "reason": "support", "chart_start": 2000},
        ]
    }
    changes = {
        "removed": ["drums:old", "other:old"],
        "added": ["other:new", "bass:new"],
        "removed_attacks": [["drums", 1000], ["other", 2000]],
        "added_attacks": [["other", 1000], ["bass", 2000]],
    }
    old_selected = [{"id": "drums:old"}, {"id": "other:old"}]
    new_selected = [{"id": "other:new"}, {"id": "bass:new"}]
    return changes, candidates, old_selection, new_selection, old_selected, new_selected


def test_build_review_cases_maps_changed_candidates_and_attack_clicks():
    cases = build_review_cases(*_fixture(), windows_ms=((0, 1500), (1500, 2500)))

    assert [len(case["attacks"]) for case in cases] == [2, 2]
    first = cases[0]
    assert {row["id"] for row in first["attacks"]} == {
        "removed:drums:1000",
        "added:other:1000",
    }
    assert {row["id"] for row in first["candidate_changes"]} == {
        "drums:old",
        "other:new",
    }
    assert first["attacks"][0]["candidate_ids"] == ["other:new"]


def test_build_review_cases_rejects_unmapped_attack_time():
    changes, candidates, old, new, old_selected, new_selected = _fixture()
    changes["removed_attacks"][0][1] = 999
    with pytest.raises(ValueError, match="maps to 0 candidate IDs"):
        build_review_cases(changes, candidates, old, new, old_selected, new_selected)


def test_write_audition_keeps_fixed_gain_and_valid_pcm16(tmp_path):
    rate = 22050
    time = np.arange(rate, dtype=np.float32) / rate
    source = tmp_path / "mix.wav"
    destination = tmp_path / "preview.wav"
    wavfile.write(source, rate, (0.4 * np.sin(2 * np.pi * 220 * time)).astype(np.float32))

    metadata = _write_audition(source, destination, 0, 1000, [(500, "head")])

    restored_rate, restored = wavfile.read(destination)
    assert restored_rate == rate
    assert restored.shape == (rate, 2)
    assert restored.dtype == np.int16
    assert metadata["gain"] == 1.0
    assert metadata["events"] == [{"time_ms": 500, "sample": 11025, "kind": "head"}]
    assert len(metadata["sha256"]) == 64


def _label_fixture():
    manifest = {
        "schema": "manialab-chord-evidence-audio-review-v1",
        "review_key": "review-key",
        "cases": [
            {
                "case_id": "case-a",
                "attacks": [
                    {"id": "removed:drums:1000", "voice": "drums", "direction": "removed"},
                    {"id": "added:other:1000", "voice": "other", "direction": "added"},
                ],
            }
        ],
    }
    labels = {
        "schema": "manialab-chord-evidence-audio-review-labels-v1",
        "review_key": "review-key",
        "manifest_sha256": "a" * 64,
        "exported_at": "2026-11-13T12:00:00.000Z",
        "events": {
            "removed:drums:1000": {"audibility": "audible", "note": "清晰起音"},
            "added:other:1000": {"audibility": "", "note": ""},
        },
        "caseNotes": {"case-a": ""},
    }
    return manifest, labels


def test_summarize_labels_checks_identity_and_never_accepts_music_quality():
    manifest, labels = _label_fixture()
    result = summarize_review_labels(manifest, labels, "a" * 64)

    assert result["attack_event_count"] == 2
    assert result["labeled_event_count"] == 1
    assert result["audibility"] == {"audible": 1, "unlabeled": 1}
    assert result["by_voice"]["drums"] == {"audible": 1}
    assert result["notes_emitted"] is False
    assert result["accuracy_computed"] is False
    assert result["musical_acceptance"] is False


def test_summarize_labels_rejects_wrong_manifest_or_event_ids():
    manifest, labels = _label_fixture()
    labels["manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Manifest SHA-256"):
        summarize_review_labels(manifest, labels, "a" * 64)

    labels["manifest_sha256"] = "a" * 64
    labels["events"]["unknown:event"] = {"audibility": "audible", "note": ""}
    with pytest.raises(ValueError, match="Event IDs"):
        summarize_review_labels(manifest, labels, "a" * 64)
