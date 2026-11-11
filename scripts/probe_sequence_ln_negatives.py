"""Measure LN-transition hard-negative coverage on held-out validation charts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manialab.domain import parse_osu, validate
from scripts.train_layout_network import records_and_groups, split_for
from scripts.train_sequence_prior import (
    NEGATIVE_POLICY_LN_TRANSITION,
    NEGATIVE_POLICY_RANDOM,
    LN_TRANSITION_KINDS,
    SEED,
    chart_samples,
)

DIAGNOSTIC_FIELDS = (
    "candidate_event_windows",
    "sampled_window_attempts",
    "negative_failed_windows",
    "negative_examples_found",
    "negative_search_trials",
    "sampled_ln_windows",
    "retained_ln_windows",
    "ln_transition_search_trials",
    "ln_transition_negative_examples_found",
    "ln_transition_failed_windows",
)


def _new_totals():
    return {
        **{field: 0 for field in DIAGNOSTIC_FIELDS},
        "ln_transition_changed_kinds": {
            kind: 0 for kind in LN_TRANSITION_KINDS
        },
    }


def _add_diagnostics(totals, diagnostics):
    for field in DIAGNOSTIC_FIELDS:
        totals[field] += int(diagnostics.get(field, 0))
    for kind, count in diagnostics.get("ln_transition_changed_kinds", {}).items():
        totals["ln_transition_changed_kinds"][kind] += int(count)


def probe(corpus, *, keys=10, max_groups=None, windows_per_chart=80, negatives=2):
    records, excluded = records_and_groups(corpus)
    selected = [
        row for row in records
        if row["keys"] == keys and split_for(row["group"]) == "validation"
    ]
    groups = sorted({row["group"] for row in selected})
    if max_groups is not None:
        if max_groups <= 0:
            raise ValueError("max_groups must be positive")
        groups = groups[:max_groups]
        selected = [row for row in selected if row["group"] in set(groups)]
    if not selected:
        raise ValueError(f"no {keys}K validation charts found")

    reports = {
        NEGATIVE_POLICY_RANDOM: _new_totals(),
        NEGATIVE_POLICY_LN_TRANSITION: _new_totals(),
    }
    by_group = defaultdict(lambda: {
        policy: _new_totals() for policy in reports
    })
    per_map = []
    for row in sorted(selected, key=lambda item: (item["group"], item["beatmap_id"])):
        path = corpus / "maps" / f"{row['beatmap_id']}.osu"
        chart = parse_osu(path.read_bytes().decode("utf-8-sig"))
        if not validate(chart["notes"], keys)["valid"]:
            raise ValueError(f"Invalid validation chart: {row['beatmap_id']}")

        map_result = {
            "beatmap_id": row["beatmap_id"],
            "group": row["group"],
            "policies": {},
        }
        for policy in reports:
            _, _, _, diagnostics = chart_samples(
                chart["notes"],
                keys,
                SEED + int(row["beatmap_id"]),
                windows_per_chart=windows_per_chart,
                negatives=negatives,
                timing_points=chart["timing"],
                negative_policy=policy,
                return_diagnostics=True,
            )
            _add_diagnostics(reports[policy], diagnostics)
            _add_diagnostics(by_group[row["group"]][policy], diagnostics)
            map_result["policies"][policy] = diagnostics
        per_map.append(map_result)

    return {
        "revision": "sequence-ln-hard-negative-probe-v1",
        "notice": (
            "Validation-only coverage probe. It measures whether feasible "
            "one-row alternatives can change the listed LN-transition counts; "
            "it does not label either layout as musically better."
        ),
        "keys": keys,
        "windows_per_chart": windows_per_chart,
        "negatives_per_window": negatives,
        "eligible_groups": len(groups),
        "eligible_maps": len(selected),
        "source_exclusions": excluded,
        "totals_by_policy": reports,
        "totals_by_group": dict(by_group),
        "maps": per_map,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("test-data/corpus-expanded/snapshot-verified"),
    )
    parser.add_argument("--keys", type=int, default=10, choices=range(6, 11))
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--windows-per-chart", type=int, default=80)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument(
        "--output", type=Path,
        default=Path("test-data/sequence-ln-hard-negative-probe-v1/report.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite probe output: {args.output}")
    report = probe(
        args.corpus,
        keys=args.keys,
        max_groups=args.max_groups,
        windows_per_chart=args.windows_per_chart,
        negatives=args.negatives,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "keys": report["keys"],
        "eligible_groups": report["eligible_groups"],
        "eligible_maps": report["eligible_maps"],
        "totals_by_policy": report["totals_by_policy"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
