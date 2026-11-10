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


def verify_guitar_model(root):
    """Verify an optional graph and its actual upstream license declarations."""
    from manialab.guitar import CHECKPOINT_SHA256, MODEL_SHA256, OUTPUTS, THRESHOLDS
    root = Path(root)
    model = root/'assets/deep/guitar-fl.onnx'
    if not model.exists():
        return None
    manifest = root/'assets/guitar-model.json'
    if not manifest.is_file():
        raise ValueError('Guitar model metadata is missing')
    identity = json.loads(manifest.read_text(encoding='utf-8'))
    expected = dict(filename='guitar-fl.onnx',sha256=MODEL_SHA256,
        upstream_checkpoint_sha256=CHECKPOINT_SHA256,license='MIT',
        model_revision='b7bec65a2b860aca72856b0feef58b5df407b777',
        source_commit='0226e74cbc805660e34bbd6a8fed2083890ebb88',
        sample_rate=16000,input_samples=160000,output_names=list(OUTPUTS),thresholds=THRESHOLDS)
    if any(identity.get(key) != value for key,value in expected.items()):
        raise ValueError('Guitar model identity differs from the verified runtime contract')
    if any(identity.get(key) is not False for key in
           ('default_enabled','predicted_releases_default','normalization')):
        raise ValueError('Guitar model must preserve explicit opt-in and unchanged input gain')
    if not model.is_file() or sha256(model) != MODEL_SHA256:
        raise ValueError('Guitar model checksum differs from the verified export')
    for field, name in (
        ('license','HFGuitar-Model-Card.md'),
        ('source_license_declaration','HFGuitar-Source-Setup.py.txt'),
        ('notice','HFGuitar-NOTICE.txt')):
        path = 'licenses/'+name
        expected_sha = identity.get(field+'_sha256')
        if (identity.get(field+'_file') != path or not isinstance(expected_sha,str)
                or len(expected_sha) != 64 or not (root/path).is_file()
                or sha256(root/path) != expected_sha):
            raise ValueError('Guitar license/attribution evidence is missing or changed: '+name)
    if (identity['license_sha256'] != 'e5717f0ce6570ecc9ad961e0ceaf6ab9c56f061eb497650132da7b4ceee3c2d8'
            or identity['source_license_declaration_sha256'] != '3d6fcb0cace324891df2eac75104d8bedabef9e38ab313eb7ed380b89f10eb7d'):
        raise ValueError('Guitar upstream license declarations differ from the reviewed revisions')
    return identity


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
