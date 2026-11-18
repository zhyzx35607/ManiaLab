"""Refresh an existing piano blind-review page without decoding source audio."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_piano_onset_blind_review import (
    _write_zip,
    render_review_html,
    review_draft_key,
    sha256,
)
from scripts.check_blind_review_scores import piano_template_rows


def _participant_rows(template_data: bytes) -> list[dict]:
    rows = []
    for source in piano_template_rows(template_data):
        audio_path = PurePosixPath(source["audio_file"])
        audio_start_ms = int(source["audio_start_ms"])
        pitch = int(source["pitch_midi"])
        rows.append({
            "clip_id": source["clip_id"],
            "audio_file": audio_path.as_posix(),
            "audio_start_ms": audio_start_ms,
            "audio_start_label": f"{audio_start_ms / 1000:.3f}s",
            "pitch": pitch,
        })
    return rows


def _validate_source_package(
    source: Path,
    manifest: dict,
    template_data: bytes,
    rows: list[dict],
) -> None:
    archive_path = source.parent / "piano-onset-review.zip"
    expected_archive_sha256 = manifest.get("participant_archive_sha256")
    if (
        not isinstance(expected_archive_sha256, str)
        or len(expected_archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_archive_sha256)
        or not archive_path.is_file()
        or sha256(archive_path) != expected_archive_sha256
    ):
        raise ValueError("Source participant archive does not match its manifest")

    if manifest.get("template_sha256") not in (None, hashlib.sha256(template_data).hexdigest()):
        raise ValueError("Source review template does not match its manifest")
    if manifest.get("samples") != len(rows):
        raise ValueError("Source review sample count does not match its manifest")

    expected_files = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Source review package must not contain symbolic links")
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            expected_files[f"piano-onset-review/{relative}"] = path

    try:
        with ZipFile(archive_path) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)) or set(members) != set(expected_files):
                raise ValueError("Source participant archive file list does not match its directory")
            if archive.testzip() is not None:
                raise ValueError("Source participant archive failed CRC validation")
            for member, path in expected_files.items():
                if archive.read(member) != path.read_bytes():
                    raise ValueError("Source participant archive bytes do not match its directory")
    except BadZipFile as error:
        raise ValueError("Source participant archive is invalid") from error


def refresh_package(source_participant: Path, output: Path) -> dict:
    source = source_participant.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise ValueError("Preserve existing review packages by choosing a new output directory")
    if (
        not output.parent.is_dir()
        or output == source
        or output in source.parents
        or source in output.parents
    ):
        raise ValueError("Choose an existing parent and a separate review output directory")

    source_manifest_path = source.parent / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_audio_sha256 = source_manifest.get("source_sha256", {}).get("audio")
    if (
        not isinstance(source_audio_sha256, str)
        or len(source_audio_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_audio_sha256.lower())
    ):
        raise ValueError("Source review manifest has no valid audio identity")

    template_path = source / "review-template.csv"
    template_data = template_path.read_bytes()
    rows = _participant_rows(template_data)
    _validate_source_package(source, source_manifest, template_data, rows)
    participant = output / "participant"
    participant.mkdir(parents=True)

    clip_sha256_by_id = {}
    for row in rows:
        source_clip = (source / row["audio_file"]).resolve()
        if source not in source_clip.parents or not source_clip.is_file():
            raise ValueError("Piano review template refers to a missing or unsafe clip")
        destination = participant / row["audio_file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_clip, destination)
        clip_sha256_by_id[row["clip_id"]] = sha256(destination)

    (participant / "review-template.csv").write_bytes(template_data)
    old_readme = source / "README.txt"
    if old_readme.is_file():
        shutil.copyfile(old_readme, participant / "README.txt")
    readme_path = participant / "README.txt"
    lock_note = (
        "\n\n网页会尝试把草稿保存在当前浏览器；定期导出 CSV，勿依赖浏览器草稿作为唯一副本。"
        "使用项目 scripts/check_blind_review_scores.py 校验评分并生成解盲前快照，"
        "完成快照后才查看包外解盲表。流程见 docs/human-review-integrity.md。\n"
    )
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if "scripts/check_blind_review_scores.py" not in readme:
        readme_path.write_text(readme + lock_note, encoding="utf-8")
    (participant / "index.html").write_text(
        render_review_html(
            rows,
            draft_key=review_draft_key(rows, source_audio_sha256, clip_sha256_by_id),
        ),
        encoding="utf-8",
    )
    archive_path = output / "piano-onset-review.zip"
    _write_zip(participant, archive_path)

    manifest = {
        "schema": 1,
        "review_type": "piano_onset_blind_review",
        "samples": len(rows),
        "source_sha256": {"audio": source_audio_sha256},
        "template_sha256": hashlib.sha256(template_data).hexdigest(),
        "participant_archive_sha256": sha256(archive_path),
        "participant_draft_autosave": "browser_local_storage",
        "source_audio_decoded": False,
        "model_inference_run": False,
        "deblind_key_copied_or_read": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-participant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(refresh_package(args.source_participant, args.output), indent=2))


if __name__ == "__main__":
    main()
