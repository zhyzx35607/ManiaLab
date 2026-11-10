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


def verify_instrument_worker(root, directory):
    from manialab.instrument_bundle import MODELS, runtime_identity, verify_model, verify_source
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('worker_sources') != runtime_identity(root):
        raise ValueError('Instrument worker sources changed; rebuild scripts/build_instrument_worker.py')
    executable = directory/'ManiaLabInstrument/ManiaLabInstrument.exe'
    if manifest.get('worker_sha256') != sha256(executable):
        raise ValueError('Instrument worker executable changed')
    if manifest.get('portable_runtime_verified') is not True:
        raise ValueError('Instrument worker requires successful packaged real-audio parity checks')
    if set(manifest.get('verified_models',[])) != set(MODELS):
        raise ValueError('Packaged instrument checks must cover both bundled weights')
    verification_path = directory/'verification.json'
    if not verification_path.is_file() or manifest.get('verification_sha256') != sha256(verification_path):
        raise ValueError('Instrument real-audio verification evidence is missing or changed')
    verification = json.loads(verification_path.read_text(encoding='utf-8'))
    if (verification.get('worker_sha256') != manifest['worker_sha256']
            or verification.get('worker_sources') != manifest['worker_sources']
            or verification.get('models') != MODELS
            or verification.get('portable_runtime_verified') is not True):
        raise ValueError('Instrument verification belongs to another build')
    cases = verification.get('cases',[])
    if (len(cases) != 4 or {(c['model'],c['allow_predicted_releases']) for c in cases}
            != {(model,release) for model in MODELS for release in (False,True)}
            or not all(c.get('raw_evidence_equal') is True and c.get('cache_reused') is True for c in cases)):
        raise ValueError('Instrument verification must cover both models, release modes and cache')
    if verify_source(directory/'source') != manifest['source_files']:
        raise ValueError('Instrument source bundle changed')
    for model in MODELS:
        verify_model(directory/'checkpoints',model)
    return manifest
