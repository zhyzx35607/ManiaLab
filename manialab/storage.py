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


def extract_osz(stream, root):
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        if len(infos) > 3000 or sum(i.file_size for i in infos) > 512 * 1024 * 1024:
            raise ValueError("谱包解压大小超过 512MB 或文件过多")
        seen = set()
        # Validate the entire archive before writing even its first member.
        for i in infos:
            target = safe_path(root, i.filename)
            if i.is_dir():
                continue
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("谱包不能包含符号链接")
            normalized = str(target).casefold()
            if normalized in seen:
                raise ValueError("谱包存在重复文件名")
            seen.add(normalized)
        for i in infos:
            target = safe_path(root, i.filename)
            if i.is_dir() or target.suffix.lower() not in ASSETS:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(i) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


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
