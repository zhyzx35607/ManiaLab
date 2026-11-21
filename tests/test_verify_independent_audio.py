import pytest

from scripts.verify_independent_audio import variant_selection_payload


def test_variant_selection_payload_pairs_saved_index_with_candidate_identity():
    saved = {
        "active": 0,
        "variants": [
            {"candidate_id": "candidate-a"},
            {"candidate_id": "candidate-b"},
        ],
    }

    assert variant_selection_payload(saved, 1) == {
        "index": 1,
        "candidate_id": "candidate-b",
    }


def test_variant_selection_payload_rejects_candidate_without_identity():
    with pytest.raises(ValueError, match="Saved candidate has no persistent identity"):
        variant_selection_payload({"variants": [{"settings": {"keys": 6}}]}, 0)
