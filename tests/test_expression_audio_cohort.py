from __future__ import annotations

from collections import Counter

from scripts.select_expression_audio_cohort import (
    _has_existing_pair,
    select_cohort,
)
from scripts.train_pattern_prior import split_for


def record(beatmap_id, song, keys, category, *, set_id=None, approved=1):
    return {
        "beatmap_id": beatmap_id,
        "beatmapset_id": set_id or beatmap_id,
        "keys": keys,
        "approved": approved,
        "eligible": True,
        "reference_role": "expression_and_layout",
        "sha256": f"chart-{beatmap_id}",
        "song_key": song,
        "related_song_keys": [song],
        "related_audio_hashes": [],
        "audio_hash": "",
        "ln_class": category,
    }


def empty_existing():
    return {
        "beatmap_ids": set(),
        "beatmapset_ids": set(),
        "songs": set(),
        "audio_hashes": set(),
    }


def test_existing_audio_match_excludes_song_aliases_and_set_ids():
    row = record(1, "artist:song", 10, "ln_majority", set_id=20)
    assert _has_existing_pair([row], empty_existing() | {"songs": {"artist:song"}})
    assert _has_existing_pair([row], empty_existing() | {"beatmapset_ids": {20}})
    assert not _has_existing_pair([row], empty_existing())


def test_cohort_is_reproducible_and_never_selects_a_song_twice():
    rows = [
        record(1, "same:song", 10, "ln_majority", set_id=10),
        record(2, "same:song", 7, "hybrid", set_id=10),
    ]
    for keys in (6, 7, 8, 9, 10):
        for category in ("ln_majority", "hybrid", "tap_majority"):
            for i in range(12):
                song = f"{keys}:{category}:{i}"
                rows.append(record(
                    1000 + len(rows), song, keys, category,
                    set_id=1000 + len(rows),
                ))

    first, report = select_cohort(
        rows, empty_existing(), train_per_cell=1,
        validation_per_cell=1, test_per_cell=1,
    )
    second, _ = select_cohort(
        rows, empty_existing(), train_per_cell=1,
        validation_per_cell=1, test_per_cell=1,
    )
    assert [row["beatmap_id"] for row in first] == [
        row["beatmap_id"] for row in second
    ]
    assert len({row["group"] for row in first}) == len(first)
    assert all(
        row["layout_training_split"] == split_for(row["group"])
        for row in first
    )
    assert report["selected_song_groups"] == len(first)
    assert {row["ln_class"] for row in first} == {
        "ln_majority", "hybrid", "tap_majority"
    }


def test_existing_groups_are_not_selected_and_shortfalls_are_reported():
    rows = [
        record(1, "paired:song", 10, "ln_majority", set_id=10),
        record(2, "new:song", 10, "ln_majority", set_id=20),
    ]
    existing = empty_existing() | {"beatmapset_ids": {10}}
    selected, report = select_cohort(
        rows, existing, train_per_cell=1,
        validation_per_cell=0, test_per_cell=0,
    )
    assert len(selected) <= 1
    assert all(row["beatmapset_id"] != 10 for row in selected)
    assert report["existing_pair_groups_excluded"] == 1
    assert sum(cell["shortfall"] for cell in report["coverage"]) > 0


def test_ranked_chart_is_preferred_over_loved_in_the_same_song_group():
    rows = [
        record(10, "same:song", 10, "hybrid", set_id=10, approved=4),
        record(11, "same:song", 10, "hybrid", set_id=11, approved=1),
    ]
    selected, _ = select_cohort(
        rows,
        empty_existing(),
        train_per_cell=1,
        validation_per_cell=1,
        test_per_cell=1,
    )
    assert len(selected) == 1
    assert selected[0]["approved"] == 1
