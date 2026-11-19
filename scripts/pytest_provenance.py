"""Create and validate provenance sidecars for a pytest JUnit report."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCHEMA = "manialab-pytest-evidence-v1"
_TEXT_EXTENSIONS = frozenset({
    ".cjs", ".cfg", ".css", ".html", ".ini", ".js", ".json", ".md", ".mjs",
    ".osu", ".py", ".toml", ".txt", ".xml", ".yaml", ".yml",
})
_SOURCE_DIRS = ("docs", "manialab", "scripts", "static", "tests", "vendor", ".github")
_ROOT_FILES = frozenset({
    ".gitattributes", ".gitignore", "AGENTS.md", "LICENSE", "README.md",
    "THIRD_PARTY_NOTICES.md", "launcher.py", "package.json", "package-lock.json",
    "pytest.ini", "requirements-lock.txt", "requirements-deep-lock.txt",
    "assets/arrangement-profiles.json", "assets/deep-models.json",
    "assets/guitar-model.json", "assets/model-manifest.json",
    "assets/pattern-prior.json", "assets/piano-model.json", "assets/basic-pitch.onnx",
    "assets/mapperatorinator-requirements.txt",
})
_EXCLUDED_FILES = frozenset({"licenses/FFmpeg-build-info.txt"})


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
    if relative in _ROOT_FILES or relative in _EXCLUDED_FILES:
        return relative in _ROOT_FILES
    path = Path(relative)
    parts = path.parts
    if not parts or path.suffix.lower() not in _TEXT_EXTENSIONS:
        return False
    if relative in _ROOT_FILES or relative in _EXCLUDED_FILES:
        return relative in _ROOT_FILES
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
