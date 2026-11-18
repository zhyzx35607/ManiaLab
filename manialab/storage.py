from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import uuid
import zipfile

AUDIO = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus", ".aac"}
ASSETS = AUDIO | {".osu", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMPORT_BYTES = 512 * 1024 * 1024
MAX_IMPORT_ENTRIES = 3000
MAX_IMPORT_CENTRAL_DIRECTORY_BYTES = 32 * 1024 * 1024


def safe_path(root, name):
    name = str(name).replace("\\", "/")
    part = PurePosixPath(name)
    if not name or part.is_absolute() or any(x in ("..", ".") or ":" in x for x in part.parts):
        raise ValueError("文件路径不安全")
    if any(re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)", x, re.I) for x in part.parts):
        raise ValueError("不支持系统保留文件名")
    path = (Path(root) / Path(*part.parts)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("文件路径越界")
    return path


def validate_import_paths(root, entries):
    """Reserve case-insensitive file/directory names before writing uploads."""
    root = Path(root).resolve()
    occupied = {str(path.resolve()).casefold(): path.is_dir() for path in root.rglob("*")}
    targets = []
    for name, is_directory in entries:
        target = safe_path(root, name)
        if target == root:
            raise ValueError("导入文件路径无效")
        for parent in reversed(target.relative_to(root).parents):
            directory = root / parent
            key = str(directory).casefold()
            if key in occupied and not occupied[key]:
                raise ValueError("导入文件与目录路径冲突")
            occupied[key] = True
        key = str(target).casefold()
        if key in occupied and not (is_directory and occupied[key]):
            raise ValueError("导入存在同名文件或文件与目录路径冲突")
        occupied[key] = is_directory
        targets.append(target)
    return targets


def extract_osz(
    stream,
    root,
    *,
    max_uncompressed_bytes=MAX_IMPORT_BYTES,
    max_entries=MAX_IMPORT_ENTRIES,
):
    position = stream.tell()
    try:
        # Match ZipFile's EOCD/ZIP64 interpretation before it reads the full directory.
        endrec = zipfile._EndRecData(stream)
    finally:
        stream.seek(position)
    if endrec is not None:
        if endrec[zipfile._ECD_ENTRIES_TOTAL] > max_entries:
            raise ValueError("导入文件数量超过安全上限")
        if endrec[zipfile._ECD_SIZE] > MAX_IMPORT_CENTRAL_DIRECTORY_BYTES:
            raise ValueError("谱包中央目录超过安全上限")

    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        if len(infos) > max_entries:
            raise ValueError("导入文件数量超过安全上限")
        uncompressed_bytes = sum(info.file_size for info in infos)
        if uncompressed_bytes > max_uncompressed_bytes:
            raise ValueError("导入展开后的总数据量超过 512MB 安全上限")
        # Validate the entire archive before writing even its first member.
        for i in infos:
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("谱包不能包含符号链接")
        targets = validate_import_paths(root, [(i.filename, i.is_dir()) for i in infos])
        for i, target in zip(infos, targets):
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.suffix.lower() not in ASSETS:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(i) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
    return uncompressed_bytes, len(infos)


class Projects:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, project_id):
        if not re.fullmatch(r"[0-9a-f]{32}", project_id):
            raise ValueError("项目 ID 无效")
        return self.root / project_id

    def create(self):
        project_id = uuid.uuid4().hex
        path = self.path(project_id)
        (path / "files").mkdir(parents=True)
        return project_id, path

    def save(self, state):
        path = self.path(state["id"]) / "state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    def load(self, project_id):
        return json.loads((self.path(project_id) / "state.json").read_text(encoding="utf-8"))

    def recent(self):
        paths = sorted(self.root.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]
        result = []
        for path in paths:
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                result.append({"id": state["id"], "title": state["title"], "mode": state["mode"]})
            except (ValueError, KeyError, OSError):
                continue
        return result
