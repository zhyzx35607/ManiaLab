"""Run pytest once and write a source/JUnit provenance sidecar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pytest_provenance import (
    ROOT, SOURCE_SCHEMA, source_commit, source_snapshot, sha256_file, utc_now,
)


def run(junitxml: Path, evidence: Path, pytest_args: list[str], *, root: Path = ROOT) -> int:
    root = Path(root).resolve()
    junitxml = Path(junitxml)
    evidence = Path(evidence)
    if not junitxml.is_absolute():
        junitxml = root / junitxml
    if not evidence.is_absolute():
        evidence = root / evidence
    if any(arg == "--junitxml" or arg.startswith("--junitxml=") for arg in pytest_args):
        raise ValueError("pytest args must not override the sidecar-bound JUnit path")
    junitxml.parent.mkdir(parents=True, exist_ok=True)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    before = source_snapshot(root)
    commit = source_commit(root)
    started = utc_now()
    command = [sys.executable, "-m", "pytest", "-q", *pytest_args,
               f"--junitxml={junitxml}"]
    pytest_exit_code = 127
    launch_error = None
    try:
        completed = subprocess.run(command, cwd=root, check=False)
        pytest_exit_code = completed.returncode
    except OSError as exc:
        launch_error = str(exc)
    after = source_snapshot(root)
    sidecar = {
        "schema": SOURCE_SCHEMA,
        "source_commit": commit,
        "argv": command,
        "junit_sha256": sha256_file(junitxml) if junitxml.is_file() else "",
        "source_digest_before": before["digest"],
        "source_digest_after": after["digest"],
        "exit_code": pytest_exit_code,
        "started_at": started,
        "completed_at": utc_now(),
        "file_count": before["files"],
    }
    if launch_error is not None:
        sidecar["launch_error"] = launch_error
    evidence.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8",
    )
    if before["digest"] != after["digest"] or not junitxml.is_file():
        return pytest_exit_code or 1
    return pytest_exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args, pytest_args = parser.parse_known_args()
    return run(args.junitxml, args.evidence, pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
