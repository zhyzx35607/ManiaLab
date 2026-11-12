"""Train and evaluate research-only mania lane-pattern priors.

The model ranks lane assignments for fixed event rows. It does not select
attacks, change LN endpoints, or certify that a chart is musical.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from manialab.domain import parse_osu, validate
from manialab.layout import fingers, release_choices
from manialab.pattern_prior import score_network
from manialab.pattern_sequence import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_COUNTS,
    FEATURE_SCHEMA_NAMES,
    LN_INTERACTION_SCHEMA,
    LN_PHRASE_CONTEXT_BEATS,
    LN_PHRASE_CONTEXT_SCHEMA,
    WINDOW_ROWS,
    mirror_window_features,
    window_features,
    window_features_with_ln_interactions,
    window_features_with_ln_phrase_context,
)
from manialab.sequence_prior import (
    MAX_SEQUENCE_BEAT_MS,
    sequence_beat_clock as _sequence_beat_clock,
    sequence_beat_positions as _sequence_beat_positions,
)
from scripts.train_layout_network import records_and_groups, split_for

SEED = 20261111
DEFAULT_WINDOWS_PER_CHART = 80
DEFAULT_NEGATIVES = 2
DEFAULT_GROUP_WINDOW_BUDGET = 1000
CACHE_REVISION = "sequence-window-cache-v3"
LN_TRANSITION_CACHE_REVISION = "sequence-window-cache-v4-ln-transition"
LN_INTERACTION_CACHE_REVISION = "sequence-window-cache-v5-ln-interactions"
LN_PHRASE_CONTEXT_CACHE_REVISION = "sequence-window-cache-v7-ln-phrase-context"
NEGATIVE_POLICY_RANDOM = "random-feasible"
NEGATIVE_POLICY_LN_TRANSITION = "ln-transition-hard"
NEGATIVE_POLICIES = (NEGATIVE_POLICY_RANDOM, NEGATIVE_POLICY_LN_TRANSITION)
SAKURA_SET_ID = 2644803
SAKURA_BEATMAP_IDS = frozenset((5969639, 5969640))
UNKNOWN_TIME = -100000
LN_TRANSITION_KINDS = (
    "inverse_heads",
    "reverse_shield_heads",
    "shield_heads",
    "same_hand_hybrid_taps",
    "split_release_hands",
    "release_press_hands",
    "near_release_press_hands",
)


def _feature_schema_sha256(names=FEATURE_NAMES):
    payload = json.dumps(list(names), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _empty_chart_arrays(negatives, feature_count=FEATURE_COUNT):
    return (
        np.empty((0, feature_count), np.float16),
        np.empty((0, negatives, feature_count), np.float16),
        np.empty((0,), bool),
    )


def _encode_window(rows, keys, beat_lengths, initial_ends, initial_last,
                   feature_schema, *, beat_positions=None,
                   initial_end_positions=None, initial_last_positions=None):
    if feature_schema == LN_PHRASE_CONTEXT_SCHEMA:
        return window_features_with_ln_phrase_context(
            rows,
            keys,
            beat_lengths,
            initial_ends=initial_ends,
            initial_last=initial_last,
            beat_positions=beat_positions,
            initial_end_positions=initial_end_positions,
            initial_last_positions=initial_last_positions,
        )
    if feature_schema == LN_INTERACTION_SCHEMA:
        return window_features_with_ln_interactions(
            rows, keys, beat_lengths,
            initial_ends=initial_ends,
            initial_last=initial_last,
        )
    if feature_schema != "sequence-window-v1":
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    return window_features(
        rows, keys, beat_lengths,
        initial_ends=initial_ends,
        initial_last=initial_last,
    )


def _empty_sampling_diagnostics():
    return dict(
        candidate_event_windows=0,
        sampled_window_attempts=0,
        retained_windows=0,
        negative_failed_windows=0,
        negative_examples_found=0,
        negative_search_trials=0,
        ignored_red_timing_points=0,
        sampled_ln_windows=0,
        sampled_tap_windows=0,
        retained_ln_windows=0,
        retained_tap_windows=0,
        ln_transition_search_trials=0,
        ln_transition_negative_examples_found=0,
        ln_transition_failed_windows=0,
        ln_transition_changed_kinds={kind: 0 for kind in LN_TRANSITION_KINDS},
        phrase_multirow_search_trials=0,
        phrase_single_row_fallback_trials=0,
        phrase_multirow_negatives_found=0,
        phrase_single_row_negatives_found=0,
        max_chord_size_histograms=dict(sampled={}, retained={}, negative_failed={}),
    )


def _histogram_add(histogram, value):
    key = str(int(value))
    histogram[key] = histogram.get(key, 0) + 1


def _load_chart_cache(path, expected_config):
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["config"].item()))
        if config != expected_config:
            raise ValueError(f"Sequence cache configuration mismatch: {path}")
        return (
            data["positive"],
            data["negative"],
            data["context"],
            json.loads(str(data["diagnostics"].item())),
        )


def _as_rows(rows, assignments, beat_position_by_time=None):
    encoded = []
    for (start, notes), lanes in zip(rows, assignments):
        row = {
            "start_ms": start,
            "notes": [(int(lane), int(note.end)) for lane, note in zip(lanes, notes)],
        }
        if beat_position_by_time is not None:
            row["beat_position"] = float(beat_position_by_time[start])
            row["note_end_positions"] = [
                float(beat_position_by_time[int(note.end)]) for note in notes
            ]
        encoded.append(row)
    return encoded


def _negative_path(rows, options, actual, targets, initial_ends, initial_last, rng):
    """Find one feasible path that changes a chosen subset of event rows."""
    chosen = [None] * len(rows)

    def visit(index, ends, last):
        if index == len(rows):
            return list(chosen)
        candidates = options[index]
        if index not in targets:
            candidates = [actual[index]]
        else:
            candidates = [value for value in candidates if value != actual[index]]
            if not candidates:
                return None
            order = rng.permutation(len(candidates))
            candidates = [candidates[int(i)] for i in order]
        start, notes = rows[index]
        for lanes in candidates:
            if any(ends[lane] > start or last[lane] == start for lane in lanes):
                continue
            next_ends, next_last = ends.copy(), last.copy()
            for lane, note in zip(lanes, notes):
                next_ends[lane] = note.end
                next_last[lane] = start
            chosen[index] = lanes
            result = visit(index + 1, next_ends, next_last)
            if result is not None:
                return result
        return None

    return visit(0, np.asarray(initial_ends, dtype=np.int64),
                 np.asarray(initial_last, dtype=np.int64))


def _same_under_mirroring(a, b, keys):
    return (np.array_equal(a, b)
            or np.array_equal(a, mirror_window_features(b, keys)))


def _ln_transition_signature(rows, assignments, keys, beat_lengths,
                             initial_ends, initial_last, initial_lns=()):
    """Count lane-dependent LN transitions for one feasible window layout."""
    profile = fingers(keys)
    hands = np.array([
        (.5, .5) if hand == 2 else (1., 0.) if hand == 0 else (0., 1.)
        for hand, _ in profile
    ])
    ends = np.asarray(initial_ends, dtype=np.int64).copy()
    last = np.asarray(initial_last, dtype=np.int64).copy()
    counts = Counter()
    prior_lns = list(initial_lns)

    for (start, notes), lanes, beat_ms in zip(rows, assignments, beat_lengths):
        prior_hold = (last != UNKNOWN_TIME) & (ends > last)
        active = prior_hold & (ends > start)
        released = prior_hold & (ends == start)
        assigned = [int(lane) for lane in lanes]
        tap_lanes = [
            lane for lane, note in zip(assigned, notes) if note.end == start
        ]
        release_window = max(40., min(150., beat_ms * .25))
        inverse_window = min(150., beat_ms * .3)

        for lane, note in zip(assigned, notes):
            if last[lane] != UNKNOWN_TIME:
                gap = start - int(ends[lane])
                if prior_hold[lane] and 0 < gap <= inverse_window:
                    kind = "inverse_heads" if note.end > start else "reverse_shield_heads"
                    counts[kind] += 1
                elif (not prior_hold[lane] and note.end > start
                      and 0 < start - int(last[lane]) <= inverse_window):
                    counts["shield_heads"] += 1

        for hand_index in (0, 1):
            in_hand = hands[:, hand_index] > 0
            held_in_hand = bool(np.any(active & in_hand))
            released_in_hand = bool(np.any(released & in_hand))
            continuing_in_hand = held_in_hand
            heads_in_hand = any(in_hand[lane] for lane in assigned)

            if released_in_hand and continuing_in_hand:
                counts["split_release_hands"] += 1
            if released_in_hand and heads_in_hand:
                counts["release_press_hands"] += 1
            nearby_release = any(
                note_start < start
                and 1 < abs(note_end - start) < release_window
                and in_hand[lane]
                for note_start, note_end, lane in prior_lns
            )
            if nearby_release and heads_in_hand:
                counts["near_release_press_hands"] += 1

        for lane in tap_lanes:
            if np.any(active & (hands @ hands[lane] > 0)):
                counts["same_hand_hybrid_taps"] += 1

        for lane, note in zip(assigned, notes):
            ends[lane], last[lane] = note.end, start
            if note.end > start:
                prior_lns.append((start, note.end, lane))

    return tuple((kind, int(counts[kind])) for kind in LN_TRANSITION_KINDS)


def _find_ln_transition_negatives(rows, options, actual, keys, beat_lengths,
                                  initial_ends, initial_last, rng, count,
                                  initial_lns=(),
                                  feature_schema="sequence-window-v1",
                                  *, beat_positions=None,
                                  beat_position_by_time=None,
                                  initial_end_positions=None,
                                  initial_last_positions=None,
                                  phrase_search_diagnostics=None):
    """Find feasible LN-window alternatives that change an LN transition type."""
    # v5 cannot encode releases older than its per-lane edge state.
    signature_initial_lns = (
        () if feature_schema == LN_INTERACTION_SCHEMA else initial_lns
    )
    positive_signature = _ln_transition_signature(
        rows, actual, keys, beat_lengths, initial_ends, initial_last,
        signature_initial_lns)
    positive_features = _encode_window(
        _as_rows(rows, actual, beat_position_by_time), keys, beat_lengths,
        initial_ends, initial_last, feature_schema,
        beat_positions=beat_positions,
        initial_end_positions=initial_end_positions,
        initial_last_positions=initial_last_positions)
    eligible = [
        index for index, candidates in enumerate(options)
        if any(value != actual[index] for value in candidates)
    ]
    if not eligible:
        return [], 0, {}

    found, seen = [], []
    trials = 0
    changed_kind_counts = Counter()
    for _ in range(max(count * 80, 80)):
        if len(found) >= count:
            break
        trials += 1
        if feature_schema == LN_PHRASE_CONTEXT_SCHEMA:
            eligible_set = set(eligible)
            contiguous_starts = [
                index for index in eligible
                if index + 1 in eligible_set
            ]
            if contiguous_starts:
                target_start = int(rng.choice(contiguous_starts))
                max_length = 1
                while (max_length < min(4, len(rows) - target_start)
                       and target_start + max_length in eligible_set):
                    max_length += 1
                target_count = int(rng.integers(2, max_length + 1))
                targets = set(range(target_start, target_start + target_count))
                if phrase_search_diagnostics is not None:
                    phrase_search_diagnostics[
                        "phrase_multirow_search_trials"
                    ] += 1
            else:
                targets = {int(rng.choice(eligible))}
                if phrase_search_diagnostics is not None:
                    phrase_search_diagnostics[
                        "phrase_single_row_fallback_trials"
                    ] += 1
        else:
            targets = {int(rng.choice(eligible))}
        path = _negative_path(
            rows, options, actual, targets, initial_ends, initial_last, rng
        )
        if path is None or path == actual:
            continue
        negative_signature = _ln_transition_signature(
            rows, path, keys, beat_lengths, initial_ends, initial_last,
            signature_initial_lns)
        positive_counts = dict(positive_signature)
        negative_counts = dict(negative_signature)
        changed_kinds = [
            kind for kind in LN_TRANSITION_KINDS
            if positive_counts[kind] != negative_counts[kind]
        ]
        if not changed_kinds:
            continue
        negative = _encode_window(
            _as_rows(rows, path, beat_position_by_time), keys, beat_lengths,
            initial_ends, initial_last, feature_schema,
            beat_positions=beat_positions,
            initial_end_positions=initial_end_positions,
            initial_last_positions=initial_last_positions)
        if _same_under_mirroring(positive_features, negative, keys):
            continue
        if any(_same_under_mirroring(old, negative, keys) for old in seen):
            continue
        found.append(negative)
        seen.append(negative)
        changed_kind_counts.update(changed_kinds)
        if phrase_search_diagnostics is not None:
            diagnostic_key = (
                "phrase_multirow_negatives_found"
                if len(targets) > 1
                else "phrase_single_row_negatives_found"
            )
            phrase_search_diagnostics[diagnostic_key] += 1
    return found, trials, dict(changed_kind_counts)


def chart_samples(notes, keys, seed, *, windows_per_chart=DEFAULT_WINDOWS_PER_CHART,
                  negatives=DEFAULT_NEGATIVES, timing_points=(),
                  negative_policy=NEGATIVE_POLICY_RANDOM,
                  feature_schema="sequence-window-v1",
                  return_diagnostics=False):
    """Build positive and feasible local-perturbation windows for one chart."""
    if keys not in range(6, 11):
        raise ValueError("only 6K through 10K charts are supported")
    if windows_per_chart <= 0 or negatives <= 0:
        raise ValueError("window and negative budgets must be positive")
    if negative_policy not in NEGATIVE_POLICIES:
        raise ValueError(f"unsupported negative policy: {negative_policy}")
    if feature_schema not in FEATURE_SCHEMA_COUNTS:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    if (feature_schema == LN_INTERACTION_SCHEMA
            and negative_policy != NEGATIVE_POLICY_LN_TRANSITION):
        raise ValueError(
            "LN interaction features require the LN-transition hard-negative policy"
        )
    if (feature_schema == LN_PHRASE_CONTEXT_SCHEMA
            and negative_policy != NEGATIVE_POLICY_LN_TRANSITION):
        raise ValueError(
            "LN phrase-context features require the LN-transition hard-negative policy"
        )
    feature_count = FEATURE_SCHEMA_COUNTS[feature_schema]
    rows_by_start = defaultdict(list)
    clock, ignored_red_timing_points = _sequence_beat_clock(timing_points, 500)
    for note in notes:
        rows_by_start[note.start].append(note)
    ordered = [(start, sorted(row, key=lambda n: n.lane))
               for start, row in sorted(rows_by_start.items())]
    if feature_schema == LN_PHRASE_CONTEXT_SCHEMA:
        event_positions = _sequence_beat_positions(
            [start for start, _ in ordered], timing_points, 500
        )
        event_position_by_start = {
            start: float(position)
            for (start, _), position in zip(ordered, event_positions)
        }
        all_relevant_times = sorted(
            {start for start, _ in ordered}
            | {int(note.end) for _, row in ordered for note in row}
        )
        time_to_beat_position = dict(zip(
            all_relevant_times,
            _sequence_beat_positions(all_relevant_times, timing_points, 500),
        ))
    else:
        event_positions = None
        event_position_by_start = None
        time_to_beat_position = None
    ln_releases = sorted(
        (note.end, note.start, note.lane)
        for _, row in ordered for note in row if note.hold
    )
    ln_release_ends = [event[0] for event in ln_releases]
    diagnostics = _empty_sampling_diagnostics()
    diagnostics["candidate_event_windows"] = len(ordered)
    diagnostics["ignored_red_timing_points"] = ignored_red_timing_points
    if not ordered:
        arrays = _empty_chart_arrays(negatives, feature_count)
        return (*arrays, diagnostics) if return_diagnostics else arrays

    ends = np.full(keys, -100000, dtype=np.int64)
    last = ends.copy()
    pre_ends, pre_last = [], []
    for start, row in ordered:
        pre_ends.append(ends.copy())
        pre_last.append(last.copy())
        for note in row:
            ends[note.lane], last[note.lane] = note.end, start

    contexts, windows = [], []
    for end_index in range(len(ordered)):
        if event_positions is not None:
            first_index = int(np.searchsorted(
                event_positions,
                event_positions[end_index] - LN_PHRASE_CONTEXT_BEATS,
                side="left",
            ))
        else:
            first_index = max(0, end_index - WINDOW_ROWS + 1)
        window = ordered[first_index:end_index + 1]
        window_start = window[0][0]
        context = (any(note.hold for _, row in window for note in row)
                   or bool(np.any(pre_ends[first_index] > window_start)))
        contexts.append(context)
        windows.append((first_index, end_index, window))

    rng = np.random.default_rng(seed)
    context_indices = np.flatnonzero(contexts)
    tap_indices = np.flatnonzero(~np.asarray(contexts, dtype=bool))
    take = set()
    for pool, budget in ((context_indices, (windows_per_chart + 1) // 2),
                         (tap_indices, windows_per_chart // 2)):
        if len(pool):
            take.update(rng.choice(pool, min(len(pool), budget), replace=False).tolist())
    if len(take) < min(windows_per_chart, len(ordered)):
        remaining = np.array(sorted(set(range(len(ordered))) - take), dtype=int)
        if len(remaining):
            take.update(rng.choice(
                remaining, min(len(remaining), min(windows_per_chart, len(ordered)) - len(take)),
                replace=False).tolist())

    positives, negative_sets, selected_contexts = [], [], []
    sampled_chord_sizes = {}
    retained_chord_sizes = {}
    failed_chord_sizes = {}
    for end_index in sorted(take):
        first_index, _, window = windows[end_index]
        max_chord_size = max(len(row) for _, row in window)
        _histogram_add(sampled_chord_sizes, max_chord_size)
        if contexts[end_index]:
            diagnostics["sampled_ln_windows"] += 1
        else:
            diagnostics["sampled_tap_windows"] += 1
        initial_ends, initial_last = pre_ends[first_index], pre_last[first_index]
        beat_lengths = [clock(start) for start, _ in window]
        window_beat_positions = (
            [event_position_by_start[start] for start, _ in window]
            if time_to_beat_position is not None else None
        )
        initial_end_positions = (
            tuple(
                time_to_beat_position.get(int(end)) if end >= 0 else None
                for end in initial_ends
            )
            if time_to_beat_position is not None else None
        )
        initial_last_positions = (
            tuple(
                time_to_beat_position.get(int(start)) if start >= 0 else None
                for start in initial_last
            )
            if time_to_beat_position is not None else None
        )
        initial_lns = set()
        for start, _ in window:
            begin = bisect_left(ln_release_ends, start - 150)
            end = bisect_right(ln_release_ends, start + 150)
            initial_lns.update(
                event for event in ln_releases[begin:end]
                if event[1] < window[0][0] and event[1] < start
            )
        initial_lns = tuple(sorted(initial_lns))
        actual = [tuple(note.lane for note in row) for _, row in window]
        options = []
        for (_, row), actual_lanes in zip(window, actual):
            durations = np.asarray([note.end - note.start for note in row], dtype=float)
            generated = release_choices(keys, durations)
            unique = {tuple(map(int, choice)) for choice in generated}
            unique.add(actual_lanes)
            options.append(sorted(unique))
        positive = _encode_window(
            _as_rows(window, actual, time_to_beat_position),
            keys, beat_lengths,
            initial_ends, initial_last, feature_schema,
            beat_positions=window_beat_positions,
            initial_end_positions=initial_end_positions,
            initial_last_positions=initial_last_positions)
        if (negative_policy == NEGATIVE_POLICY_LN_TRANSITION
                and contexts[end_index]):
            found, search_trials, changed_kinds = _find_ln_transition_negatives(
                window, options, actual, keys, beat_lengths,
                initial_ends, initial_last, rng, negatives, initial_lns,
                feature_schema,
                beat_positions=window_beat_positions,
                beat_position_by_time=time_to_beat_position,
                initial_end_positions=initial_end_positions,
                initial_last_positions=initial_last_positions,
                phrase_search_diagnostics=(
                    diagnostics
                    if feature_schema == LN_PHRASE_CONTEXT_SCHEMA else None
                ))
            diagnostics["ln_transition_search_trials"] += search_trials
            diagnostics["ln_transition_negative_examples_found"] += len(found)
            for kind, count in changed_kinds.items():
                diagnostics["ln_transition_changed_kinds"][kind] += count
            if len(found) != negatives:
                diagnostics["ln_transition_failed_windows"] += 1
        else:
            # Keep the original policy for tap windows and existing experiments.
            found = []
            seen = []
            search_trials = 0
            eligible = [index for index, choices in enumerate(options)
                        if any(value != actual[index] for value in choices)]
            for _ in range(max(negatives * 40, 40)):
                if len(found) >= negatives or not eligible:
                    break
                search_trials += 1
                count = int(rng.choice([1, 1, 2, 2, 3]))
                count = min(count, len(eligible))
                targets = set(map(int, rng.choice(eligible, count, replace=False)))
                path = _negative_path(window, options, actual, targets,
                                      initial_ends, initial_last, rng)
                if path is None or path == actual:
                    continue
                negative = _encode_window(
                    _as_rows(window, path, time_to_beat_position),
                    keys, beat_lengths,
                    initial_ends, initial_last, feature_schema,
                    beat_positions=window_beat_positions,
                    initial_end_positions=initial_end_positions,
                    initial_last_positions=initial_last_positions)
                if _same_under_mirroring(positive, negative, keys):
                    continue
                if any(_same_under_mirroring(old, negative, keys) for old in seen):
                    continue
                found.append(negative)
                seen.append(negative)
        diagnostics["negative_examples_found"] += len(found)
        diagnostics["negative_search_trials"] += search_trials
        if len(found) != negatives:
            _histogram_add(failed_chord_sizes, max_chord_size)
            diagnostics["negative_failed_windows"] += 1
            continue
        positives.append(positive)
        negative_sets.append(found)
        selected_contexts.append(bool(contexts[end_index]))
        _histogram_add(retained_chord_sizes, max_chord_size)

    diagnostics["sampled_window_attempts"] = len(take)
    diagnostics["retained_windows"] = len(positives)
    diagnostics["retained_ln_windows"] = int(np.count_nonzero(selected_contexts))
    diagnostics["retained_tap_windows"] = (
        len(selected_contexts) - diagnostics["retained_ln_windows"]
    )
    diagnostics["max_chord_size_histograms"] = dict(
        sampled=dict(sorted(sampled_chord_sizes.items(), key=lambda item: int(item[0]))),
        retained=dict(sorted(retained_chord_sizes.items(), key=lambda item: int(item[0]))),
        negative_failed=dict(sorted(failed_chord_sizes.items(), key=lambda item: int(item[0]))),
    )
    if not positives:
        arrays = _empty_chart_arrays(negatives, feature_count)
    else:
        arrays = (
            np.asarray(positives, dtype=np.float16),
            np.asarray(negative_sets, dtype=np.float16),
            np.asarray(selected_contexts, dtype=bool),
        )
    return (*arrays, diagnostics) if return_diagnostics else arrays


def _sakura_holdout_groups(records):
    return {
        row["group"] for row in records
        if int(row.get("beatmapset_id") or 0) == SAKURA_SET_ID
        or int(row.get("beatmap_id") or 0) in SAKURA_BEATMAP_IDS
    }


def prepare(root, cache, *, windows_per_chart=DEFAULT_WINDOWS_PER_CHART,
            negatives=DEFAULT_NEGATIVES,
            group_window_budget=DEFAULT_GROUP_WINDOW_BUDGET,
            max_maps_per_split=None,
            negative_policy=NEGATIVE_POLICY_RANDOM,
            feature_schema="sequence-window-v1",
            exclude_test_split=False,
            split_seed=None):
    if negative_policy not in NEGATIVE_POLICIES:
        raise ValueError(f"unsupported negative policy: {negative_policy}")
    if feature_schema not in FEATURE_SCHEMA_COUNTS:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    if not isinstance(exclude_test_split, bool):
        raise ValueError("exclude_test_split must be a boolean")
    if exclude_test_split and feature_schema != LN_PHRASE_CONTEXT_SCHEMA:
        raise ValueError(
            "test split exclusion is reserved for the phrase-context experiment"
        )
    if (feature_schema == LN_INTERACTION_SCHEMA
            and negative_policy != NEGATIVE_POLICY_LN_TRANSITION):
        raise ValueError(
            "LN interaction features require the LN-transition hard-negative policy"
        )
    if (feature_schema == LN_PHRASE_CONTEXT_SCHEMA
            and negative_policy != NEGATIVE_POLICY_LN_TRANSITION):
        raise ValueError(
            "LN phrase-context features require the LN-transition hard-negative policy"
        )
    records, excluded = records_and_groups(
        root,
        exclude_splits=("test",) if exclude_test_split else (),
        split_seed=split_seed,
    )
    holdout_groups = _sakura_holdout_groups(records)
    selected = [row for row in records if row["group"] not in holdout_groups]
    cache.mkdir(parents=True, exist_ok=True)
    if feature_schema == LN_PHRASE_CONTEXT_SCHEMA:
        cache_revision = LN_PHRASE_CONTEXT_CACHE_REVISION
    elif feature_schema == LN_INTERACTION_SCHEMA:
        cache_revision = LN_INTERACTION_CACHE_REVISION
    else:
        cache_revision = (
            LN_TRANSITION_CACHE_REVISION
            if negative_policy == NEGATIVE_POLICY_LN_TRANSITION
            else CACHE_REVISION
        )
    feature_names = FEATURE_SCHEMA_NAMES[feature_schema]
    feature_count = FEATURE_SCHEMA_COUNTS[feature_schema]
    configs = dict(windows_per_chart=windows_per_chart, negatives=negatives,
                   feature_count=feature_count,
                   cache_revision=cache_revision,
                   feature_schema_sha256=_feature_schema_sha256(feature_names),
                   seed=SEED,
                   beat_reference=(
                       "integrated_redline_beats"
                       if feature_schema == LN_PHRASE_CONTEXT_SCHEMA
                       else "tempo_at_each_row"
                   ),
                   excluded_splits=["test"] if exclude_test_split else [],
                   max_sequence_beat_ms=MAX_SEQUENCE_BEAT_MS)
    if feature_schema != "sequence-window-v1":
        configs["feature_schema"] = feature_schema
    if feature_schema == LN_PHRASE_CONTEXT_SCHEMA:
        configs["phrase_context_beats"] = LN_PHRASE_CONTEXT_BEATS
    if negative_policy != NEGATIVE_POLICY_RANDOM:
        configs["negative_policy"] = negative_policy
    arrays = defaultdict(lambda: [[], [], []])
    statistics = defaultdict(lambda: dict(maps=0, groups=set(), windows=0, ln_windows=0))
    sampling_statistics = defaultdict(lambda: dict(
        maps_seen=0, candidate_event_windows=0, sampled_window_attempts=0,
        retained_windows=0, negative_failed_windows=0,
        negative_examples_found=0, negative_search_trials=0,
        ignored_red_timing_points=0,
        sampled_ln_windows=0, sampled_tap_windows=0,
        retained_ln_windows=0, retained_tap_windows=0,
        ln_transition_search_trials=0,
        ln_transition_negative_examples_found=0,
        ln_transition_failed_windows=0,
        ln_transition_changed_kinds={kind: 0 for kind in LN_TRANSITION_KINDS},
        phrase_multirow_search_trials=0,
        phrase_single_row_fallback_trials=0,
        phrase_multirow_negatives_found=0,
        phrase_single_row_negatives_found=0,
        max_chord_size_histograms=dict(sampled={}, retained={}, negative_failed={}),
    ))
    per_group_maps = Counter((row["keys"], row["group"]) for row in selected)
    seen_split_maps = Counter()
    manifest = []
    sampling_maps = []
    skipped = Counter()
    if exclude_test_split:
        skipped["excluded_test_split_maps"] = int(
            excluded.pop("split_test_excluded", 0)
        )
    for index, row in enumerate(selected):
        key, group = row["keys"], row["group"]
        split = split_for(group, seed=split_seed)
        if exclude_test_split and split == "test":
            skipped["excluded_test_split_maps"] += 1
            continue
        split_key = (key, split)
        if (max_maps_per_split is not None
                and seen_split_maps[split_key] >= max_maps_per_split):
            continue
        seen_split_maps[split_key] += 1
        cache_suffix = {
            CACHE_REVISION: "sequence-window-v3",
            LN_TRANSITION_CACHE_REVISION: "sequence-window-v4-ln-transition",
            LN_INTERACTION_CACHE_REVISION: "sequence-window-v5-ln-interactions",
            LN_PHRASE_CONTEXT_CACHE_REVISION:
                "sequence-window-v7-ln-phrase-context",
        }[cache_revision]
        path = cache / f"{row['sha256']}-{row['beatmap_id']}-{cache_suffix}.npz"
        if path.exists():
            positive, negative, context, diagnostics = _load_chart_cache(path, configs)
        else:
            raw = (root / "maps" / f"{row['beatmap_id']}.osu").read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"Chart hash mismatch: {row['beatmap_id']}")
            chart = parse_osu(raw.decode("utf-8-sig"))
            if not validate(chart["notes"], key)["valid"]:
                raise ValueError(f"Invalid chart in training snapshot: {row['beatmap_id']}")
            positive, negative, context, diagnostics = chart_samples(
                chart["notes"], key, SEED + int(row["beatmap_id"]),
                windows_per_chart=windows_per_chart, negatives=negatives,
                negative_policy=negative_policy,
                feature_schema=feature_schema,
                timing_points=chart["timing"], return_diagnostics=True)
            np.savez_compressed(path, positive=positive, negative=negative,
                                context=context,
                                diagnostics=np.asarray(json.dumps(diagnostics, sort_keys=True)),
                                config=np.asarray(json.dumps(configs, sort_keys=True)))

        sampling = sampling_statistics[(key, split)]
        sampling["maps_seen"] += 1
        for field in (
            "candidate_event_windows", "sampled_window_attempts",
            "retained_windows", "negative_failed_windows",
            "negative_examples_found", "negative_search_trials",
            "ignored_red_timing_points",
            "sampled_ln_windows", "sampled_tap_windows",
            "retained_ln_windows", "retained_tap_windows",
            "ln_transition_search_trials", "ln_transition_negative_examples_found",
            "ln_transition_failed_windows",
            "phrase_multirow_search_trials",
            "phrase_single_row_fallback_trials",
            "phrase_multirow_negatives_found",
            "phrase_single_row_negatives_found",
        ):
            sampling[field] += int(diagnostics.get(field, 0))
        for kind, count in diagnostics.get("ln_transition_changed_kinds", {}).items():
            sampling["ln_transition_changed_kinds"][kind] += int(count)
        for kind, histogram in diagnostics["max_chord_size_histograms"].items():
            aggregate = sampling["max_chord_size_histograms"][kind]
            for chord_size, count in histogram.items():
                aggregate[chord_size] = aggregate.get(chord_size, 0) + int(count)
        map_report = dict(
            id=row["beatmap_id"], sha256=row["sha256"], keys=key,
            split=split, group=group, diagnostics=diagnostics,
            training_windows_before_group_budget=len(positive),
            training_windows=0,
        )
        sampling_maps.append(map_report)
        if not len(positive):
            skipped["charts_without_retained_windows"] += 1
            continue

        budget = max(1, group_window_budget // per_group_maps[(key, group)])
        if len(positive) > budget:
            chosen = np.random.default_rng(int(row["beatmap_id"])).choice(
                len(positive), budget, replace=False)
            positive, negative, context = positive[chosen], negative[chosen], context[chosen]
        map_report["training_windows"] = len(positive)
        arrays[(key, split)][0].append(positive)
        arrays[(key, split)][1].append(negative)
        arrays[(key, split)][2].append(context)
        info = statistics[(key, split)]
        info["maps"] += 1
        info["groups"].add(group)
        info["windows"] += len(positive)
        info["ln_windows"] += int(np.count_nonzero(context))
        info["tap_windows"] = info.get("tap_windows", 0) + int(
            len(context) - np.count_nonzero(context)
        )
        manifest.append(dict(
            id=row["beatmap_id"], sha256=row["sha256"], keys=key,
            split=split, group=group, windows=len(positive),
            ln_windows=int(np.count_nonzero(context)),
        ))
        if index % 100 == 0:
            print(json.dumps(dict(prepared=index, eligible=len(selected))), flush=True)

    data = {}
    for key_split, parts in arrays.items():
        if parts[0]:
            data[key_split] = tuple(np.concatenate(part) for part in parts)
    split_info = {}
    for key in range(6, 11):
        for split in ("train", "validation", "test"):
            key_split = (key, split)
            sample = sampling_statistics[key_split]
            sample["max_chord_size_histograms"] = {
                kind: dict(sorted(hist.items(), key=lambda item: int(item[0])))
                for kind, hist in sample["max_chord_size_histograms"].items()
            }
            split_info[f"{key}:{split}"] = dict(
                maps=statistics[key_split]["maps"],
                groups=len(statistics[key_split]["groups"]),
                windows=statistics[key_split]["windows"],
                ln_windows=statistics[key_split]["ln_windows"],
                tap_windows=statistics[key_split].get("tap_windows", 0),
                sampling=sample,
            )
    return data, manifest, dict(
        source_exclusions=excluded, sakura_holdout_groups=len(holdout_groups),
        skipped=dict(skipped), splits=split_info, configs=configs,
        split_seed=split_seed,
        sampling_maps=sampling_maps)


def _mirror_batch(x, keys):
    return mirror_window_features(x, keys)


def _score(model, x, keys, torch):
    mirrored = _mirror_batch(x, keys)
    return (model(torch.from_numpy(x.astype(np.float32)))
            + model(torch.from_numpy(mirrored.astype(np.float32)))) / 2


def evaluate(model, data, split, torch, keys=None):
    result = {}
    model.eval()
    with torch.inference_mode():
        selected_keys = range(6, 11) if keys is None else keys
        for key_count in selected_keys:
            sample = data.get((key_count, split))
            if sample is None:
                continue
            positive, negative, context = sample
            correct = loss = total = 0.0
            by_context = {
                "ln": [0.0, 0.0, 0.0],
                "taps": [0.0, 0.0, 0.0],
            }
            for begin in range(0, len(positive), 512):
                end = min(len(positive), begin + 512)
                pos = positive[begin:end].astype(np.float32)
                neg = negative[begin:end].reshape(
                    -1, positive.shape[1]).astype(np.float32)
                pos_score = _score(model, pos, key_count, torch).numpy()[:, 0]
                neg_score = _score(model, neg, key_count, torch).numpy()[:, 0].reshape(
                    end - begin, negative.shape[1])
                margin = pos_score[:, None] - neg_score
                rows = context[begin:end]
                for name, mask in (("ln", rows), ("taps", ~rows)):
                    selected = margin[mask]
                    if not len(selected):
                        continue
                    bucket = by_context[name]
                    bucket[0] += float(np.count_nonzero(selected > 0) + .5 * np.count_nonzero(selected == 0))
                    bucket[1] += float(np.logaddexp(0, -selected).sum())
                    bucket[2] += selected.size
            for count in by_context.values():
                correct += count[0]
                loss += count[1]
                total += count[2]
            result[str(key_count)] = dict(
                pairs=int(total),
                accuracy=round(correct / total, 4) if total else None,
                loss=round(loss / total, 4) if total else None,
                ln=dict(pairs=int(by_context["ln"][2]),
                        accuracy=round(by_context["ln"][0] / by_context["ln"][2], 4)
                        if by_context["ln"][2] else None,
                        loss=round(by_context["ln"][1] / by_context["ln"][2], 4)
                        if by_context["ln"][2] else None),
                taps=dict(pairs=int(by_context["taps"][2]),
                          accuracy=round(by_context["taps"][0] / by_context["taps"][2], 4)
                          if by_context["taps"][2] else None,
                          loss=round(by_context["taps"][1] / by_context["taps"][2], 4)
                          if by_context["taps"][2] else None),
            )
    return result


def _training_keys(data, requested=None):
    available = [key for key in range(6, 11) if (key, "train") in data]
    keys = available if requested is None else sorted(set(requested))
    if not keys or any(key not in range(6, 11) for key in keys):
        raise ValueError("training keys must contain values from 6 through 10")
    missing = [
        key for key in keys
        if (key, "train") not in data or (key, "validation") not in data
    ]
    if missing:
        raise ValueError(f"Need training and validation groups for keys {missing}")
    return keys


def _manifest_sha256(manifest):
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _validate_initial_model_report(artifact, report, manifest):
    """Only warm-start within the exact same frozen corpus and split."""
    expected_hash = _manifest_sha256(manifest)
    source_manifest = report.get("manifest")
    if (not isinstance(source_manifest, list)
            or _manifest_sha256(source_manifest) != expected_hash
            or report.get("training_manifest_sha256") != expected_hash
            or artifact.get("training_manifest_sha256") != expected_hash):
        raise ValueError(
            "initial model/report must use the same frozen training manifest "
            "and song-group split as this run"
        )
    if (report.get("revision") != artifact.get("revision")
            or report.get("feature_names") != artifact.get("features")
            or report.get("training_keys") != artifact.get("training_keys")):
        raise ValueError("initial model/report schema or training keys do not match")
    inherited_hash = report.get("initial_training_manifest_sha256")
    warm_start_fields = (
        "initial_model", "initial_model_sha256", "initial_report",
        "initial_report_sha256", "initial_training_manifest_sha256",
    )
    if any(
        value.get(field) is not None
        for value in (artifact, report) for field in warm_start_fields
    ):
        if (inherited_hash != expected_hash
                or artifact.get("initial_training_manifest_sha256") != expected_hash):
            raise ValueError(
                "initial warm-start model has no matching ancestor manifest proof"
            )
    return expected_hash


def _load_initial_model_weights(model, path, *, feature_schema, train_keys,
                                manifest, torch, report_path=None):
    source = Path(path)
    raw = source.read_bytes()
    artifact = json.loads(raw.decode("utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("initial model artifact must be a JSON object")
    if artifact.get("revision") != feature_schema:
        raise ValueError("initial model feature schema does not match")
    if artifact.get("features") != list(FEATURE_SCHEMA_NAMES[feature_schema]):
        raise ValueError("initial model feature names do not match")

    initial_keys = artifact.get("training_keys")
    if (not isinstance(initial_keys, list)
            or any(isinstance(key, bool) or not isinstance(key, int)
                   for key in initial_keys)
            or not set(train_keys).issubset(initial_keys)):
        raise ValueError(
            "initial model must include every requested training key"
        )

    source_report_path = (
        Path(report_path) if report_path is not None
        else source.with_name("report.json")
    )
    report_raw = source_report_path.read_bytes()
    source_report = json.loads(report_raw.decode("utf-8"))
    if not isinstance(source_report, dict):
        raise ValueError("initial model report must be a JSON object")
    manifest_hash = _validate_initial_model_report(
        artifact, source_report, manifest)
    source_hash = hashlib.sha256(raw).hexdigest()
    if (source_report.get("model_sha256") is not None
            and source_report["model_sha256"] != source_hash):
        raise ValueError("initial model/report model SHA256 does not match")

    initial_layers = artifact.get("layers")
    model_layers = [
        layer for layer in model if isinstance(layer, torch.nn.Linear)
    ]
    if not isinstance(initial_layers, list) or len(initial_layers) != len(model_layers):
        raise ValueError("initial model layer count does not match")

    prepared = []
    for index, (layer, values) in enumerate(zip(model_layers, initial_layers)):
        if not isinstance(values, dict):
            raise ValueError(f"initial model layer {index} is invalid")
        if "weight" not in values or "bias" not in values:
            raise ValueError(f"initial model layer {index} is incomplete")
        weight = np.asarray(values.get("weight"), dtype=np.float32)
        bias = np.asarray(values.get("bias"), dtype=np.float32)
        if weight.shape != tuple(layer.weight.shape):
            raise ValueError(f"initial model layer {index} weight shape does not match")
        if bias.shape != tuple(layer.bias.shape):
            raise ValueError(f"initial model layer {index} bias shape does not match")
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError(f"initial model layer {index} contains non-finite weights")
        prepared.append((weight, bias))

    with torch.no_grad():
        for layer, (weight, bias) in zip(model_layers, prepared):
            layer.weight.copy_(torch.as_tensor(weight, dtype=layer.weight.dtype))
            layer.bias.copy_(torch.as_tensor(bias, dtype=layer.bias.dtype))
    return dict(
        initial_model=str(source),
        initial_model_sha256=source_hash,
        initial_report=str(source_report_path),
        initial_report_sha256=hashlib.sha256(report_raw).hexdigest(),
        initial_training_manifest_sha256=manifest_hash,
    )


def _evaluation_splits(skip_test_evaluation):
    return (
        ("train", "validation")
        if skip_test_evaluation
        else ("train", "validation", "test")
    )


def train(args):
    import torch

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite research output: {args.output}")
    if args.exclude_test_split and (
        args.feature_schema != LN_PHRASE_CONTEXT_SCHEMA
        or not args.skip_test_evaluation
    ):
        raise ValueError(
            "--exclude-test-split requires the phrase-context schema and "
            "--skip-test-evaluation"
        )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data, manifest, prep_report = prepare(
        args.corpus, args.cache,
        windows_per_chart=args.windows_per_chart,
        negatives=args.negatives,
        group_window_budget=args.group_window_budget,
        max_maps_per_split=args.max_maps_per_split,
        negative_policy=args.negative_policy,
        feature_schema=args.feature_schema,
        exclude_test_split=args.exclude_test_split,
        split_seed=args.split_seed)
    train_keys = _training_keys(data, args.keys)
    feature_schema = args.feature_schema
    feature_names = FEATURE_SCHEMA_NAMES[feature_schema]
    feature_count = FEATURE_SCHEMA_COUNTS[feature_schema]
    pools = {}
    for key in train_keys:
        context = data[(key, "train")][2]
        pools[key] = [pool for pool in (
            np.flatnonzero(context), np.flatnonzero(~context)) if len(pool)]

    model = torch.nn.Sequential(
        torch.nn.Linear(feature_count, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1))
    provenance = {}
    if args.initial_model is not None:
        provenance = _load_initial_model_weights(
            model, args.initial_model,
            feature_schema=feature_schema,
            train_keys=train_keys,
            manifest=manifest,
            torch=torch,
            report_path=args.initial_report,
        )
    elif args.initial_report is not None:
        raise ValueError("--initial-report requires --initial-model")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=.002)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    epochs = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for step in range(args.steps):
            key = train_keys[step % len(train_keys)]
            positive, negative, _ = data[(key, "train")]
            available = pools[key]
            per_pool = max(1, args.batch // len(available))
            indices = np.concatenate([
                rng.choice(pool, per_pool, replace=True) for pool in available])
            neg_index = rng.integers(negative.shape[1], size=len(indices))
            x = positive[indices].astype(np.float32)
            y = negative[indices, neg_index].astype(np.float32)
            mirror = rng.random(len(x)) < .5
            x[mirror] = _mirror_batch(x[mirror], key)
            y[mirror] = _mirror_batch(y[mirror], key)
            x_tensor, y_tensor = torch.from_numpy(x), torch.from_numpy(y)
            x_reflect, y_reflect = torch.from_numpy(_mirror_batch(x, key)), torch.from_numpy(_mirror_batch(y, key))
            positive_score = (model(x_tensor) + model(x_reflect)) / 2
            negative_score = (model(y_tensor) + model(y_reflect)) / 2
            loss = torch.nn.functional.softplus(-(positive_score - negative_score)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        validation = evaluate(model, data, "validation", torch, keys=train_keys)
        losses = [
            report[context]["loss"]
            for report in validation.values()
            for context in ("ln", "taps")
            if report[context]["loss"] is not None
        ]
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        if mean_loss < best_loss:
            best_loss, best_epoch = mean_loss, epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        entry = dict(epoch=epoch + 1, train_loss=round(float(np.mean(train_losses)), 4),
                     validation=validation, seconds=round(time.perf_counter() - started, 1))
        epochs.append(entry)
        print(json.dumps(entry), flush=True)
    if best_state is None:
        raise ValueError("No validation windows; refusing to export an unvalidated model")
    model.load_state_dict(best_state)
    final_metrics = {
        split: evaluate(model, data, split, torch, keys=train_keys)
        for split in _evaluation_splits(args.skip_test_evaluation)
    }
    layers = [
        dict(weight=np.round(layer.weight.detach().numpy(), 7).tolist(),
             bias=np.round(layer.bias.detach().numpy(), 7).tolist())
        for layer in model if isinstance(layer, torch.nn.Linear)
    ]
    probe = rng.normal(size=(16, feature_count)).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(probe))[:, 0].numpy()
    exported = score_network(probe, layers)
    export_error = float(np.max(np.abs(expected - exported)))
    if export_error > 1e-4:
        raise AssertionError(f"Export mismatch: {export_error}")
    training_hash = _manifest_sha256(manifest)
    if feature_schema == LN_PHRASE_CONTEXT_SCHEMA:
        method = (
            "rolling 8-beat LN phrase context with a four-row local prefix; "
            "attempted feasible contiguous 2–4-event-row lane perturbations, "
            "with a single-row fallback when no adjacent eligible rows exist; "
            "integrated redline beat positions; song/set/audio-group split; "
            "LN/tap-balanced batches; mirror-averaged MLP"
        )
    else:
        method = (
            "4-row fixed-time windows; feasible 1–3-row lane perturbations; "
            "song/set/audio-group split; LN/tap-balanced batches; mirror-averaged MLP; "
            "ignore red periods above 10000 ms for local feature timing; "
            f"negative_policy={args.negative_policy}"
        )
    report = dict(
        revision=feature_schema,
        purpose=(
            "LN phrase-context lane ranking under fixed authored events; research only"
            if feature_schema == LN_PHRASE_CONTEXT_SCHEMA
            else "Four-row lane-pattern ranking under fixed authored events; research only"
        ),
        feature_names=list(feature_names),
        training_manifest_sha256=training_hash,
        training_keys=train_keys,
        models=final_metrics,
        prep=prep_report,
        best_epoch=best_epoch,
        history=epochs,
        export_max_error=export_error,
        learning_rate=args.learning_rate,
        test_evaluation_skipped=args.skip_test_evaluation,
        test_split_excluded=args.exclude_test_split,
        split_seed=args.split_seed,
        seed=SEED,
        method=method,
        manifest=manifest,
    )
    report.update(provenance)
    args.output.mkdir(parents=True, exist_ok=False)
    payload = dict(
        revision=feature_schema,
        features=list(feature_names),
        training_keys=train_keys,
        layers=layers,
        training_manifest_sha256=training_hash,
        purpose=report["purpose"],
    )
    if provenance:
        payload["initial_training_manifest_sha256"] = training_hash
    model_raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    report["model_sha256"] = hashlib.sha256(model_raw).hexdigest()
    (args.output / "model.json").write_bytes(model_raw)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), best_epoch=best_epoch,
                          models=final_metrics, export_max_error=export_error)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=Path("test-data/corpus-expanded/snapshot-verified"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--windows-per-chart", type=int, default=DEFAULT_WINDOWS_PER_CHART)
    parser.add_argument("--negatives", type=int, default=DEFAULT_NEGATIVES)
    parser.add_argument("--group-window-budget", type=int,
                        default=DEFAULT_GROUP_WINDOW_BUDGET)
    parser.add_argument("--max-maps-per-split", type=int)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--keys", nargs="+", type=int, choices=range(6, 11),
                        help="Train a key-count specialist (default: 6K through 10K).")
    parser.add_argument("--initial-model", type=Path,
                        help="Warm-start within the same frozen corpus and split.")
    parser.add_argument("--initial-report", type=Path,
                        help="Source training report (default: model's sibling report.json).")
    parser.add_argument("--skip-test-evaluation", action="store_true",
                        help="Leave test metrics out of the final training report.")
    parser.add_argument("--exclude-test-split", action="store_true",
                        help="Do not read test maps (phrase-context experiments only).")
    parser.add_argument("--split-seed", type=int,
                        help="Use a deterministic alternate song-group split.")
    parser.add_argument("--negative-policy", choices=NEGATIVE_POLICIES,
                        default=NEGATIVE_POLICY_RANDOM)
    parser.add_argument("--feature-schema",
                        choices=tuple(FEATURE_SCHEMA_NAMES),
                        default="sequence-window-v1")
    args = parser.parse_args()
    if args.cache is None:
        cache_name = (
            "sequence-window-cache-v7-ln-phrase-context-no-test"
            if args.feature_schema == LN_PHRASE_CONTEXT_SCHEMA
            and args.exclude_test_split
            else "sequence-window-cache-v7-ln-phrase-context"
            if args.feature_schema == LN_PHRASE_CONTEXT_SCHEMA
            else "sequence-window-cache-v5-ln-interactions"
            if args.feature_schema == LN_INTERACTION_SCHEMA
            else
            "sequence-window-cache-v4-ln-transition"
            if args.negative_policy == NEGATIVE_POLICY_LN_TRANSITION
            else "sequence-window-cache-v3"
        )
        args.cache = Path("test-data/corpus-expanded") / cache_name
    if args.output is None:
        output_name = (
            "sequence-prior-ln-phrase-context-v2-no-test"
            if args.feature_schema == LN_PHRASE_CONTEXT_SCHEMA
            and args.exclude_test_split
            else "sequence-prior-ln-phrase-context-v2"
            if args.feature_schema == LN_PHRASE_CONTEXT_SCHEMA
            else "sequence-prior-ln-interactions-v1"
            if args.feature_schema == LN_INTERACTION_SCHEMA
            else
            "sequence-prior-ln-transition-v1"
            if args.negative_policy == NEGATIVE_POLICY_LN_TRANSITION
            else "sequence-prior-v1"
        )
        args.output = Path("test-data") / output_name
    train(args)


if __name__ == "__main__":
    main()
