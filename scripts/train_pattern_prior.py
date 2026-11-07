"""Train a small pairwise chord preference from audited local mania charts.

The human row is compared with feasible alternative columns under its actual
hold history. This evaluates lane preferences only, never audio correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from manialab.domain import parse_osu
from manialab.layout import chord_choices
from manialab.pattern_prior import FEATURES, features


def split_for(song):
    value = int(hashlib.sha256(song.encode()).hexdigest()[:8], 16) % 10
    return "test" if value == 0 else "validation" if value == 1 else "train"


def samples(parsed, seed, limit=250):
    keys = parsed["keys"]
    rows = defaultdict(list)
    for note in parsed["notes"]:
        rows[note.start].append(note)
    beats = [(float(p[0]), float(p[1])) for line in parsed["timing"] if (p := line.split(",")) and len(p) >= 7 and p[6] == "1" and float(p[1]) > 0]
    beats.sort()
    rng = np.random.default_rng(seed)
    take = set(rng.choice(len(rows), min(limit, len(rows)), replace=False))
    ends, last, history = np.full(keys, -100000.), np.full(keys, -100000.), []
    beat_index, result = 0, []
    for index, (start, row) in enumerate(sorted(rows.items())):
        row.sort(key=lambda n: n.lane)
        while beat_index + 1 < len(beats) and beats[beat_index + 1][0] <= start:
            beat_index += 1
        beat = beats[beat_index][1] if beats else 500
        if not 180 <= beat <= 1500:
            beat = 500
        actual = np.array([n.lane for n in row])
        if index in take and len(row) < keys and len(set(actual)) == len(row):
            choices = chord_choices(keys, len(row))
            choices = choices[~((ends[choices] > start) | (last[choices] == start)).any(axis=1)]
            choices = choices[np.any(choices != actual[None], axis=1)]
            if len(choices):
                alternatives = choices[rng.choice(len(choices), min(4, len(choices)), replace=False)]
                durations = np.array([n.end - n.start for n in row])
                good = features(actual[None], keys, start, ends, last, history, durations, beat)[0]
                bad = features(alternatives, keys, start, ends, last, history, durations, beat)
                result.extend(good - bad)
        for note in row:
            ends[note.lane], last[note.lane] = note.end, start
        history.append((start, sum(1 << n.lane for n in row), len(row)))
        history = history[-2:]
    return np.asarray(result, dtype=np.float32).reshape(-1, len(FEATURES))


def fit(x, regularization=1.):
    mirrored = x.copy()
    mirrored[:, [0, 1]] = mirrored[:, [1, 0]]
    x = np.concatenate((x, mirrored))
    def loss(w):
        margins = x @ w
        objective = float(np.logaddexp(0, -margins).mean()) + regularization * np.dot(w, w) / 2
        gradient = -(x.T @ expit(-margins)) / len(x) + regularization * w
        return objective, gradient
    result = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter":150})
    if not result.success:
        raise ValueError(result.message)
    return result.x


def train(root, output, report_path):
    records = [json.loads(line) for p in sorted(root.glob("manifest-*.jsonl")) for line in p.read_text(encoding="utf-8").splitlines() if line]
    seen, selected = set(), []
    for record in records:
        if not record.get("eligible") or record["sha256"] in seen:
            continue
        seen.add(record["sha256"]); selected.append(record)
    # Union song names, exact audio hashes and set IDs before splitting. Packs
    # are deliberately kept together, even when that reduces training data.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:parent[max(a, b)] = min(a, b)
    for r in selected:
        song = "song:" + r["song_key"]
        union(song, "set:" + str(r["beatmapset_id"]))
        if r.get("audio_hash"):union(song, "audio:" + r["audio_hash"])
    arrays = defaultdict(list)
    statistics = defaultdict(lambda: dict(maps=0, groups=set(), rows=0))
    manifest = []
    per_group = Counter()
    for index, r in enumerate(selected):
        group = find("song:" + r["song_key"])
        split = split_for(group)
        key = r["keys"]
        if per_group[(key, group)] >= 5000:
            continue
        parsed = parse_osu((root / "maps" / f"{r['beatmap_id']}.osu").read_text(encoding="utf-8"))
        x = samples(parsed, int(r["beatmap_id"]))[:max(0, 5000 - per_group[(key, group)])]
        if not len(x):
            continue
        per_group[(key, group)] += len(x)
        arrays[(key, split)].append(x)
        info = statistics[(key, split)]
        info["maps"] += 1;info["groups"].add(group);info["rows"] += len(x)
        manifest.append(dict(id=r["beatmap_id"], sha256=r["sha256"], keys=key, split=split, group=group, pairs=len(x)))
        if index % 100 == 0:print(json.dumps(dict(scanned=index, eligible=len(selected))), flush=True)
    models, reports = {}, {}
    for key in range(6, 11):
        data = {split:np.concatenate(arrays[(key, split)]) if arrays[(key, split)] else np.empty((0,len(FEATURES))) for split in ("train","validation","test")}
        split_info = {s:dict(maps=statistics[(key,s)]["maps"], groups=len(statistics[(key,s)]["groups"]), pairs=len(data[s])) for s in data}
        if len(data["train"]) < 100:
            reports[str(key)] = dict(enabled=False, splits=split_info, reason="insufficient training data")
            continue
        coefficients = fit(data["train"], regularization=.02)
        metrics = {}
        for split, x in data.items():
            margin = x @ coefficients
            metrics[split] = dict(pairs=len(x), accuracy=round(float(np.mean(margin > 0) + .5 * np.mean(margin == 0)),4) if len(x) else None,
                                 loss=round(float(np.logaddexp(0,-margin).mean()),4) if len(x) else None)
        # Validation decides enablement; test is reported once and never used
        # to choose coefficients or whether the model is enabled.
        enabled = split_info["train"]["groups"] >= 8 and split_info["validation"]["groups"] >= 2 and (metrics["validation"]["accuracy"] or 0) >= .60
        models[str(key)] = dict(enabled=enabled, coefficients=np.round(coefficients,7).tolist())
        reports[str(key)] = dict(enabled=enabled, splits=split_info, metrics=metrics)
        print(json.dumps(dict(keys=key, **reports[str(key)])), flush=True)
    fingerprint = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    payload = dict(revision=1, features=list(FEATURES), models=models, training_manifest_sha256=fingerprint,
        purpose="Feasible chord preference only; not audio transcription or human playability certification")
    output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    report_path.write_text(json.dumps(dict(models=reports, manifest=manifest, training_manifest_sha256=fingerprint,
        method="pairwise logistic preference; song/audio/set grouped split; human row vs feasible random alternatives"),indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--output",type=Path,default=Path("assets/pattern-prior.json"))
    parser.add_argument("--report",type=Path,default=Path("test-data/pattern-prior-report.json"))
    args=parser.parse_args();train(args.folder,args.output,args.report)
