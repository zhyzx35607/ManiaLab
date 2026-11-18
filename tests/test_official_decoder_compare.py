from copy import deepcopy
import hashlib
from pathlib import Path
import zipfile

import pytest

from scripts.verify_official_decoder.compare import compare, digest_rows


def fixture_report(tmp_path):
    path = tmp_path / "control.osz"
    chart = "\n".join([
        "osu file format v14", "", "[General]", "AudioFilename:silence.wav", "Mode:3",
        "", "[Metadata]", "Title:Timing control", "Artist:ManiaLab", "Creator:ManiaLab",
        "Version:Timing control 6K", "", "[Difficulty]", "CircleSize:6",
        "", "[TimingPoints]", "0,500,3,2,0,70,1,0", "1750,-50,4,2,0,70,0,0",
        "3000,250,5,2,0,70,1,0", "3750,-200,4,2,0,70,0,0", "",
        "[HitObjects]", "42,192,500,1,0,0:0:0:0:", "128,192,1500,128,0,3250:0:0:0:0:",
        "298,192,3000,1,0,0:0:0:0:", "469,192,4000,128,0,4500:0:0:0:0:", "",
    ])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("timing.osu", chart)
        archive.writestr("silence.wav", b"audio fixture")
    rows = [(500, 500, 0, "tap"), (1500, 3250, 1, "ln"),
            (3000, 3000, 3, "tap"), (4000, 4500, 5, "ln")]
    return {
        "game": "test", "ruleset": "test",
        "archives": [{
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "difficulties": [{
                "entry": "timing.osu", "mode": "mania", "keys": 6, "version": "Timing control 6K",
                "audio": "silence.wav", "objects": 4, "holds": 2,
                "rows_sha256": digest_rows(rows),
                "endpoints_sha256": digest_rows((a, b, kind) for a, b, _, kind in rows),
                "timing": [
                    {"time": 0, "period": 500, "meter": 3},
                    {"time": 3000, "period": 250, "meter": 5},
                ],
                "velocities": [
                    {"time": 1750, "value": 2},
                    {"time": 3000, "value": 1},
                    {"time": 3750, "value": 0.5},
                ],
            }],
        }],
    }


def test_official_decoder_comparison_accepts_matching_control(tmp_path):
    result = compare(fixture_report(tmp_path))
    assert result["totals"] == {
        "difficulties": 1, "objects": 4, "holds": 2, "endpoints": 6,
        "red_lines": 2, "velocity_changes": 3,
    }


@pytest.mark.parametrize(("field", "mutate"), [
    ("keys", lambda row: row.update(keys=7)),
    ("lane signature", lambda row: row.update(rows_sha256="0" * 64)),
    ("endpoint signature", lambda row: row.update(endpoints_sha256="0" * 64)),
    ("meter", lambda row: row["timing"][1].update(meter=4)),
    ("scroll velocity", lambda row: row["velocities"][0].update(value=1.5)),
    ("audio reference", lambda row: row.update(audio="missing.wav")),
])
def test_official_decoder_comparison_rejects_mismatches(tmp_path, field, mutate):
    report = deepcopy(fixture_report(tmp_path))
    mutate(report["archives"][0]["difficulties"][0])
    with pytest.raises(ValueError):
        compare(report)
