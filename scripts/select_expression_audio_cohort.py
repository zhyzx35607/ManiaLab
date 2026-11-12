"""Freeze a song-grouped audio/chart cohort for expression research.

Only public Ranked/Approved/Loved native 6-10K charts are considered. Existing
paired-audio groups are excluded from new downloads; the selected JSON can be
passed directly to fetch_paired_maps.py.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_corpus import song_identity
from scripts.train_layout_network import _attach_song_groups
from scripts.train_pattern_prior import split_for


PAIRED_ROOTS = (
    "audio-additional-10k",
    "audio-extension2-10k",
    "audio-choice-pairs",
    "audio-holdout",
    "audio-audit",
    "chinese-ln/audio",
    "mapper-training-extra",
    "user-audio-audit",
)
KEY_ORDER = (10, 9, 8, 7, 6)
CLASS_ORDER = ("ln_majority", "hybrid", "tap_majority")
SPLIT_ORDER = ("train", "validation", "test")
STATUS_ORDER = {1: 0, 4: 1, 2: 2}


def existing_pair_tokens(test_data: Path) -> tuple[dict[str, set], int]:
    tokens = {
        "beatmap_ids": set(),
        "beatmapset_ids": set(),
        "songs": set(),
        "audio_hashes": set(),
    }
    count = 0
    for relative in PAIRED_ROOTS:
        root = test_data / relative
        if not root.exists():
            continue
        for path in sorted(root.rglob("reference.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            count += 1
            if value.get("beatmap_id") is not None:
                tokens["beatmap_ids"].add(int(value["beatmap_id"]))
            if value.get("beatmapset_id") is not None:
                tokens["beatmapset_ids"].add(int(value["beatmapset_id"]))
            if value.get("artist") or value.get("title"):
                tokens["songs"].add(song_identity({
                    "artist": value.get("artist", ""),
                    "title": value.get("title", ""),
                }))
            if value.get("audio_sha256"):
                tokens["audio_hashes"].add(value["audio_sha256"])
    return tokens, count


def _has_existing_pair(rows: list[dict], tokens: dict[str, set]) -> bool:
    for row in rows:
        if int(row.get("beatmap_id", 0)) in tokens["beatmap_ids"]:
            return True
        if int(row.get("beatmapset_id", 0)) in tokens["beatmapset_ids"]:
            return True
        songs = {row.get("song_key", ""), *row.get("related_song_keys", [])}
        if songs & tokens["songs"]:
            return True
        audio = {row.get("audio_hash", ""), *row.get("related_audio_hashes", [])}
        if audio & tokens["audio_hashes"]:
            return True
    return False


def _stable_order(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def select_cohort(
    records: list[dict],
    existing: dict[str, set],
    *,
    seed: str = "mania-expression-audio-v1",
    train_per_cell: int = 3,
    validation_per_cell: int = 1,
    test_per_cell: int = 1,
) -> tuple[list[dict], dict]:
    quotas = {
        "train": train_per_cell,
        "validation": validation_per_cell,
        "test": test_per_cell,
    }
    if any(type(value) is not int or value < 0 for value in quotas.values()):
        raise ValueError("Per-split quotas must be nonnegative integers")

    eligible, seen = [], set()
    for row in records:
        if row.get("approved") not in (1, 2, 4):
            continue
        if not row.get("eligible") or row.get("reference_role") != "expression_and_layout":
            continue
        if row.get("sha256") in seen:
            continue
        seen.add(row["sha256"])
        eligible.append(row)

    grouped = _attach_song_groups(eligible)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in grouped:
        by_group[row["group"]].append(row)

    existing_groups = {
        group for group, rows in by_group.items()
        if _has_existing_pair(rows, existing)
    }
    available = Counter()
    candidates: dict[tuple[int, str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for group, rows in by_group.items():
        if group in existing_groups:
            continue
        split = split_for(group)
        for row in rows:
            key = row.get("keys")
            category = row.get("ln_class")
            if key not in KEY_ORDER or category not in CLASS_ORDER:
                continue
            candidates[(key, category, split)][group].append(row)
    for cell, groups in candidates.items():
        available[cell] = len(groups)

    selected_groups = set()
    selected = []
    selected_counts = Counter()
    for key in KEY_ORDER:
        for category in CLASS_ORDER:
            for split in SPLIT_ORDER:
                cell = (key, category, split)
                quota = quotas[split]
                remaining = quota - selected_counts[cell]
                if remaining <= 0:
                    continue
                groups = sorted(
                    candidates.get(cell, {}),
                    key=lambda group: _stable_order(seed, f"{key}:{category}:{group}"),
                )
                for group in groups:
                    if group in selected_groups:
                        continue
                    options = candidates[cell][group]
                    row = min(
                        options,
                        key=lambda item: (
                            STATUS_ORDER.get(item.get("approved"), 3),
                            _stable_order(seed, f"map:{group}:{item['beatmap_id']}"),
                        ),
                    )
                    chosen = dict(row)
                    chosen["layout_training_split"] = split
                    chosen["selection_cell"] = {
                        "keys": key,
                        "ln_class": category,
                        "split": split,
                    }
                    selected.append(chosen)
                    selected_groups.add(group)
                    selected_counts[cell] += 1
                    remaining -= 1
                    if remaining == 0:
                        break

    cells = []
    for key in KEY_ORDER:
        for category in CLASS_ORDER:
            for split in SPLIT_ORDER:
                cell = (key, category, split)
                target = quotas[split]
                chosen = selected_counts[cell]
                cells.append({
                    "keys": key,
                    "ln_class": category,
                    "split": split,
                    "available_new_groups": available[cell],
                    "target_groups": target,
                    "selected_groups": chosen,
                    "shortfall": max(0, target - chosen),
                })
    report = {
        "eligible_charts": len(eligible),
        "eligible_song_groups": len(by_group),
        "existing_pair_manifests": None,
        "existing_pair_groups_excluded": len(existing_groups),
        "selected_song_groups": len(selected_groups),
        "selected_status_counts": dict(Counter(
            {1: "ranked", 4: "loved", 2: "approved"}.get(
                row.get("approved"), "other"
            )
            for row in selected
        )),
        "quotas_per_cell": quotas,
        "coverage": cells,
        "selected_counts": {
            f"{key}K/{category}/{split}": selected_counts[(key, category, split)]
            for key in KEY_ORDER
            for category in CLASS_ORDER
            for split in SPLIT_ORDER
        },
    }
    return selected, report


def build(corpus: Path, output: Path, *, seed: str, quotas: dict[str, int]) -> dict:
    manifest_path = corpus / "manifest-snapshot.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing verified snapshot manifest: {manifest_path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen cohort: {output}")
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(corpus.parents) < 2:
        raise ValueError("Corpus must be nested under the test-data directory")
    test_data = corpus.parents[1]
    existing, existing_count = existing_pair_tokens(test_data)
    selected, report = select_cohort(
        records,
        existing,
        seed=seed,
        train_per_cell=quotas["train"],
        validation_per_cell=quotas["validation"],
        test_per_cell=quotas["test"],
    )
    for row in selected:
        chart = corpus / "maps" / f"{row['beatmap_id']}.osu"
        if not chart.is_file():
            raise FileNotFoundError(f"Selected chart is missing: {chart}")
        if hashlib.sha256(chart.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Selected chart hash mismatch: {row['beatmap_id']}")
    report["existing_pair_manifests"] = existing_count
    report["corpus_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    report["selection_sha256"] = hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report["seed"] = seed
    report["paired_audio_roots"] = list(PAIRED_ROOTS)
    output.mkdir(parents=True)
    (output / "selection.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("test-data/corpus-expanded/snapshot-verified"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test-data/expression-audio-cohort-v1"),
    )
    parser.add_argument("--seed", default="mania-expression-audio-v1")
    parser.add_argument("--train-per-cell", type=int, default=3)
    parser.add_argument("--validation-per-cell", type=int, default=1)
    parser.add_argument("--test-per-cell", type=int, default=1)
    args = parser.parse_args()
    result = build(
        args.corpus,
        args.output,
        seed=args.seed,
        quotas={
            "train": args.train_per_cell,
            "validation": args.validation_per_cell,
            "test": args.test_per_cell,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
