"""Check that packaged workers were built from the current local sources."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def worker_sources(root):
    return {name: sha256(Path(root)/name) for name in
            ('scripts/separate_audio.py', 'scripts/worker_lifetime.py')}


def verify_worker(root, directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('worker_sources') != worker_sources(root):
        raise ValueError('Audio worker sources changed or are unrecorded; rebuild scripts/build_audio_worker.py')
    if manifest.get('worker_sha256') != sha256(directory/'ManiaLabAudio/ManiaLabAudio.exe'):
        raise ValueError('Audio worker executable does not match its build manifest')
    return manifest
