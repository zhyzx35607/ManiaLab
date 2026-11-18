from scripts.check_piano_transcription_chart_ab import (
    normalize_kong_attacks,
    transkun_attack_candidates,
)


def test_transkun_chart_adapter_uses_only_onset_and_pitch():
    events = [
        {
            "onset": 1.2345,
            "offset": "untrusted offset",
            "pitch": 69,
            "velocity": "not confidence",
            "has_onset": True,
            "has_offset": False,
        }
    ]

    result = transkun_attack_candidates(events, prefix="test", strength=0.5)

    assert result == [
        {
            "start": 1235,
            "end": 1235,
            "lane": -1,
            "pitch": 69,
            "strength": 0.5,
            "source": "pitch",
            "id": "piano:test:00000",
            "raw": "",
            "voice": "piano",
            "phrase": False,
            "support": [],
            "support_coverage": 0.0,
            "evidence_start": None,
            "evidence_end": None,
            "release_reason": "unresolved_or_postprocessed",
        }
    ]


def test_kong_chart_adapter_uses_same_attack_only_strength_and_release_policy():
    candidates = [
        {
            "start": 1000,
            "end": 1800,
            "pitch": 60,
            "strength": 0.91,
            "id": "piano:kong:0",
            "voice": "piano",
            "source": "pitch",
        },
        {
            "start": 1100,
            "end": 1300,
            "pitch": 48,
            "strength": 0.8,
            "id": "drums:0",
            "voice": "drums",
            "source": "transient",
        },
    ]

    result = normalize_kong_attacks(candidates, strength=0.5)

    assert len(result) == 1
    assert result[0]["start"] == result[0]["end"] == 1000
    assert result[0]["strength"] == 0.5
    assert result[0]["release_reason"] == "unresolved_or_postprocessed"
    assert result[0]["id"] == "piano:kong:0"
