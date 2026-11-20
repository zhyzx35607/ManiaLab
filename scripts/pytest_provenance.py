"""Create and validate provenance sidecars for a pytest JUnit report."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCHEMA = "manialab-pytest-evidence-v1"
_TEXT_EXTENSIONS = frozenset({
    ".cjs", ".cfg", ".css", ".html", ".ini", ".js", ".json", ".md", ".mjs", ".pem",
    ".osu", ".py", ".toml", ".txt", ".xml", ".yaml", ".yml",
})
_SOURCE_DIRS = (".github", "assets", "docs", "licenses", "manialab", "scripts", "static", "tests", "vendor")
_ROOT_FILES = frozenset({
    ".gitattributes", ".gitignore", "AGENTS.md", "LICENSE", "README.md",
    "THIRD_PARTY_NOTICES.md", "launcher.py", "package.json", "package-lock.json",
    "playwright.config.cjs", "pytest.ini", "requirements.txt", "requirements-dev.txt",
    "requirements-lock.txt", "requirements-deep.txt", "requirements-deep-lock.txt",
    "requirements-banquet-research.txt", "requirements-timbre-trap-research.txt",
    "requirements-tsumugi-research.txt", "requirements-vocal-research.txt",
    "assets/basic-pitch.onnx",
})
_EXCLUDED_FILES = frozenset({"licenses/FFmpeg-build-info.txt"})
_SESSION_STATE = pytest.StashKey[dict]()


def _run_git(root: Path, *arguments: str) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), *arguments], stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise ValueError(f"pytest evidence cannot inspect git sources: {detail}") from exc
    values = output.split(bytes([0])) if bytes([0]) in output else [output]
    return [item.decode("utf-8", "surrogateescape").rstrip("\r\n") for item in values if item]


def _is_source(relative: str) -> bool:
    if relative in _EXCLUDED_FILES:
        return False
    if relative in _ROOT_FILES:
        return True
    path = Path(relative)
    if path.parent == Path(".") and path.name.startswith("requirements") and path.suffix == ".txt":
        return True
    parts = path.parts
    if not parts or path.suffix.lower() not in _TEXT_EXTENSIONS:
        return False
    return parts[0] in _SOURCE_DIRS


def _source_files(root: Path) -> list[tuple[str, Path]]:
    root = Path(root).resolve()
    names = set(_run_git(root, "ls-files", "--cached", "-z"))
    names.update(_run_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    files = []
    for relative in sorted(names):
        relative = Path(relative).as_posix()
        if not _is_source(relative):
            continue
        path = root / Path(relative)
        if path.is_file() and not path.is_symlink():
            files.append((relative, path))
    return files


def source_snapshot(root: Path = ROOT) -> dict[str, object]:
    """Hash tracked and nonignored textual application/test sources only."""
    digest = hashlib.sha256()
    files = _source_files(Path(root))
    for relative, path in files:
        encoded_path = relative.encode("utf-8", "surrogateescape")
        content = path.read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {"digest": digest.hexdigest(), "files": len(files)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit(root: Path = ROOT) -> str:
    commits = _run_git(Path(root).resolve(), "rev-parse", "HEAD")
    if len(commits) != 1 or len(commits[0]) != 40:
        raise ValueError("pytest evidence has no valid source commit")
    return commits[0]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pytest_sessionstart(session) -> None:
    """Capture the checkout before tests when pytest is asked to write JUnit."""
    xmlpath = getattr(session.config.option, "xmlpath", None)
    if not xmlpath:
        return
    junit_path = Path(str(xmlpath)).resolve()
    state = {
        "junit_path": junit_path,
        "evidence_path": junit_path.with_name(junit_path.name + ".provenance.json"),
        "source_commit": "",
        "snapshot_before": {"digest": "", "files": 0},
        "started_at": utc_now(),
        "argv": [sys.executable, *sys.argv],
        "provenance_error": None,
    }
    try:
        state["source_commit"] = source_commit(ROOT)
        state["snapshot_before"] = source_snapshot(ROOT)
    except (OSError, ValueError) as exc:
        state["provenance_error"] = str(exc)
    session.config.stash[_SESSION_STATE] = state


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus) -> None:
    """Write the sidecar after pytest's JUnit plugin has flushed its report."""
    state = session.config.stash.get(_SESSION_STATE, None)
    if state is None:
        yield
        return
    hook_error = None
    try:
        yield
    except Exception as exc:
        # A JUnit plugin/write error must not prevent recording failed evidence.
        hook_error = exc
    junit_path = state["junit_path"]
    evidence_path = state["evidence_path"]
    provenance_error = state.get("provenance_error")
    if hook_error is not None:
        provenance_error = str(hook_error)
    try:
        snapshot_after = source_snapshot(ROOT)
        after_digest = snapshot_after["digest"]
        after_files = snapshot_after["files"]
    except (OSError, ValueError) as exc:
        after_digest, after_files = "", 0
        provenance_error = str(exc)
    document = {
        "schema": SOURCE_SCHEMA,
        "source_commit": state["source_commit"],
        "argv": state["argv"],
        "junit_sha256": sha256_file(junit_path) if junit_path.is_file() else "",
        "source_digest_before": state["snapshot_before"]["digest"],
        "source_digest_after": after_digest,
        "exit_code": int(exitstatus),
        "started_at": state["started_at"],
        "completed_at": utc_now(),
        "file_count": state["snapshot_before"]["files"],
    }
    if provenance_error is not None:
        document["provenance_error"] = provenance_error
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8")
    if (provenance_error is not None or not junit_path.is_file()
            or state["snapshot_before"]["digest"] != after_digest
            or state["snapshot_before"]["files"] != after_files):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def validate_evidence(
    evidence_path: Path,
    junit_path: Path,
    *,
    root: Path = ROOT,
    expected_commit: str | None = None,
) -> dict:
    """Validate a sidecar against its JUnit bytes and current source checkout."""
    evidence_path, junit_path, root = map(Path, (evidence_path, junit_path, root))
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("pytest evidence sidecar is missing or invalid") from exc
    if not isinstance(evidence, dict) or evidence.get("schema") != SOURCE_SCHEMA:
        raise ValueError("pytest evidence schema is unsupported")
    required = {
        "source_commit", "argv", "junit_sha256", "source_digest_before",
        "source_digest_after", "exit_code", "started_at", "completed_at", "file_count",
    }
    if not required.issubset(evidence):
        raise ValueError("pytest evidence is missing required fields")
    if (
        not isinstance(evidence["source_commit"], str)
        or not isinstance(evidence["argv"], list)
        or not all(isinstance(item, str) for item in evidence["argv"])
        or not isinstance(evidence["file_count"], int)
        or not isinstance(evidence["exit_code"], int)
    ):
        raise ValueError("pytest evidence fields have invalid types")
    for field in ("junit_sha256", "source_digest_before", "source_digest_after"):
        value = evidence[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"pytest evidence {field} is invalid")
    if evidence["exit_code"] != 0:
        raise ValueError("pytest evidence records a failed pytest run")
    if not junit_path.is_file() or sha256_file(junit_path) != evidence["junit_sha256"]:
        raise ValueError("pytest evidence JUnit digest does not match the report")
    current_commit = source_commit(root)
    if expected_commit is not None and current_commit != expected_commit:
        raise ValueError("pytest evidence checkout is not the expected commit")
    if evidence["source_commit"] != current_commit:
        raise ValueError("pytest evidence commit differs from the current checkout")
    snapshot = source_snapshot(root)
    if evidence["file_count"] != snapshot["files"]:
        raise ValueError("pytest evidence source file count differs from the current checkout")
    if (
        evidence["source_digest_before"] != evidence["source_digest_after"]
        or evidence["source_digest_before"] != snapshot["digest"]
    ):
        raise ValueError("pytest evidence source digest is unstable or stale")
    try:
        started = datetime.fromisoformat(evidence["started_at"].replace("Z", "+00:00"))
        completed = datetime.fromisoformat(evidence["completed_at"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("pytest evidence timestamps are invalid") from exc
    if completed < started:
        raise ValueError("pytest evidence timestamps are reversed")
    return evidence
