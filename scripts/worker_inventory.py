"""Freeze and verify every file in a freshly built portable worker."""
import hashlib
from pathlib import Path, PurePosixPath


def worker_inventory(directory):
    directory = Path(directory)
    if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
        raise ValueError('Worker directory must be a real directory')
    result = {}
    pending = [directory]
    while pending:
        for path in sorted(pending.pop().iterdir()):
            if path.is_symlink() or path.is_junction():
                raise ValueError('Worker runtime must not contain links: ' + str(path))
            if path.is_dir():
                pending.append(path)
                continue
            if not path.is_file():
                raise ValueError('Worker runtime contains a non-file: ' + str(path))
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            result[path.relative_to(directory).as_posix()] = dict(bytes=path.stat().st_size, sha256=digest)
    if not result:
        raise ValueError('Worker runtime is empty')
    return dict(sorted(result.items()))


def verify_worker_inventory(directory, expected):
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Worker file inventory is missing; rebuild the worker before packaging')
    folded = set()
    for name in expected:
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or path.as_posix() != name
                or '..' in path.parts or '\\' in name or ':' in name
                or name.casefold() in folded):
            raise ValueError('Unsafe or duplicate worker inventory path: ' + name)
        folded.add(name.casefold())
    actual = worker_inventory(directory)
    if set(actual) != set(expected):
        extra = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        raise ValueError(f'Worker file inventory differs: extra={extra}, missing={missing}')
    changed = [name for name in actual if actual[name] != expected[name]]
    if changed:
        raise ValueError('Worker runtime files changed: ' + ', '.join(changed))
    return actual
