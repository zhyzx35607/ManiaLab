"""Validate archived source snapshots across Git's LF/CRLF conversion only."""
from __future__ import annotations

import hashlib
from pathlib import Path


def assert_snapshot(recorded, current, root):
    """Return newline-equivalent names; reject additions or any other change.

    Archived hashes remain untouched. Checking current bytes against the hash
    captured at import also rejects edits while a long evaluation is running.
    """
    root = Path(root).resolve()
    if recorded.keys() != current.keys():
        raise ValueError('Source snapshot file list differs')
    equivalent = []
    for name, digest in current.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source snapshot path is outside the project')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Source changed after evaluation started: {name}')
        if recorded[name] == digest:
            continue
        canonical = data.replace(b'\r\n', b'\n')
        choices = {hashlib.sha256(canonical).hexdigest(),
                   hashlib.sha256(canonical.replace(b'\n', b'\r\n')).hexdigest()}
        if recorded[name] not in choices:
            raise ValueError(f'Analysis/source snapshot mismatch: {name}')
        equivalent.append(name)
    return equivalent
