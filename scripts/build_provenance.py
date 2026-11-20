"""Check that packaged workers were built from the current local sources."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile
from scripts.worker_inventory import verify_worker_inventory


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


UV_RELEASE = {
    'version': '0.12.1',
    'target': 'x86_64-pc-windows-msvc',
    'archive_name': 'uv-x86_64-pc-windows-msvc.zip',
    'archive_bytes': 19073343,
    'archive_sha256': '8fcb0cb46e1229065e344758980924e569bef5882ef45f46fada8fb24e06b74a',
    'executable_name': 'uv.exe',
    'executable_bytes': 48254464,
    'executable_sha256': 'f537cc65c1791d9d1a022132302b21ecd48cdf0a605a7b345809fbe8af4e807d',
}


def uv_release_identity(release=UV_RELEASE):
    """Return the reviewed upstream archive and extracted executable identity."""
    return {
        'version': release['version'],
        'target': release['target'],
        'archive_url': (
            'https://github.com/astral-sh/uv/releases/download/'
            f"{release['version']}/{release['archive_name']}"
        ),
        'archive_bytes': release['archive_bytes'],
        'archive_sha256': release['archive_sha256'],
        'filename': release['executable_name'],
        'bytes': release['executable_bytes'],
        'sha256': release['executable_sha256'],
    }


def verify_uv_binary(path, *, release=UV_RELEASE, check_version=True):
    """Verify the pinned uv executable bytes and optionally its reported target."""
    path = Path(path)
    identity = uv_release_identity(release)
    if not path.is_file() or path.stat().st_size != identity['bytes']:
        raise ValueError('uv executable size differs from the pinned release identity')
    if sha256(path) != identity['sha256']:
        raise ValueError('uv executable checksum differs from the pinned release identity')
    if check_version:
        result = subprocess.run(
            [str(path), '--version'], check=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        if not output.startswith(f"uv {identity['version']} ") or identity['target'] not in output:
            raise ValueError('uv executable reports an unexpected version or target')
    return identity


def download_uv_binary(destination, *, release=UV_RELEASE, opener=None):
    """Fetch only the reviewed Windows release archive and extract its verified uv.exe."""
    destination = Path(destination)
    identity = uv_release_identity(release)
    open_url = opener or urllib.request.urlopen
    with open_url(identity['archive_url'], timeout=60) as response:
        archive_bytes = response.read(identity['archive_bytes'] + 1)
    if len(archive_bytes) != identity['archive_bytes']:
        raise ValueError('uv release archive size differs from the pinned identity')
    if hashlib.sha256(archive_bytes).hexdigest() != identity['archive_sha256']:
        raise ValueError('uv release archive checksum differs from the pinned identity')
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            matches = [entry for entry in archive.infolist()
                if entry.filename == release['executable_name'] and not entry.is_dir()]
            if len(matches) != 1 or matches[0].file_size != identity['bytes']:
                raise ValueError('uv release archive has an unexpected executable inventory')
            executable_bytes = archive.read(matches[0])
    except zipfile.BadZipFile as exc:
        raise ValueError('uv release archive is not a valid ZIP') from exc
    if hashlib.sha256(executable_bytes).hexdigest() != identity['sha256']:
        raise ValueError('uv executable checksum differs from the pinned release identity')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(executable_bytes)
    return verify_uv_binary(destination, release=release)


def worker_sources(root):
    return {name: sha256(Path(root)/name) for name in
            ('scripts/separate_audio.py', 'scripts/separate_six.py',
             'scripts/worker_lifetime.py')}


def validate_piano_model_metadata(root):
    from manialab.piano import MODEL_BYTES, MODEL_SHA256

    root = Path(root)
    manifest = root/'assets/piano-model.json'
    if not manifest.is_file():
        raise ValueError('Piano model metadata is missing')
    identity = json.loads(manifest.read_text(encoding='utf-8'))
    expected = dict(
        name='High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times',
        authors=['Qiuqiang Kong', 'Bochen Li', 'Xuchen Song', 'Yuan Wan', 'Yuxuan Wang'],
        model_record='https://zenodo.org/records/4034264',
        upstream_checkpoint_url='https://zenodo.org/api/records/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth/content',
        upstream_checkpoint_bytes=171966578,
        upstream_checkpoint_md5='22b961b77c1878239fec963362097045',
        upstream_checkpoint_sha256='c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141',
        source_commit='0226e74cbc805660e34bbd6a8fed2083890ebb88',
        source_repository='https://github.com/qiuqiangkong/piano_transcription_inference',
        source_archive_url='https://codeload.github.com/qiuqiangkong/piano_transcription_inference/zip/0226e74cbc805660e34bbd6a8fed2083890ebb88',
        source_archive_bytes=411788,
        source_archive_sha256='ac7062720a03feb965ffc9e48e6246497c415d90a60780ea98206fb89cbff7e7',
        license='CC-BY-4.0',
        license_file='licenses/PianoRegression-CC-BY-4.0.txt',
        license_sha256='9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94',
        modification='Fixed batch-1, 10-second ONNX graph exported from the verified original weights; no fine-tuning. ManiaLab provides overlap stitching and release provenance.',
        filename='piano-regression.onnx',
        sha256=MODEL_SHA256,
        bytes=MODEL_BYTES,
        sample_rate=16000,
        input_samples=160000,
        export=dict(torch='2.9.1+cpu', onnx='1.19.1', numpy='2.3.5', opset=17),
        intended_source='isolated piano',
        default_enabled=False,
        research_validation='docs/piano-acoustic-research.md',
    )
    if identity != expected:
        raise ValueError('Piano model identity differs from the reviewed release contract')
    license_path = root/identity['license_file']
    if not license_path.is_file() or sha256(license_path) != identity['license_sha256']:
        raise ValueError('Piano model license/attribution evidence is missing or changed')
    return identity


def verify_piano_model(root):
    """Verify the required optional piano graph and its source/license identity."""
    root = Path(root)
    identity = validate_piano_model_metadata(root)
    from manialab.piano import piano_model_path
    model = piano_model_path(root)
    if not model.is_file():
        raise ValueError('Required piano model is missing; run scripts/fetch_piano_model.py')
    if model.stat().st_size != identity['bytes'] or sha256(model) != identity['sha256']:
        raise ValueError('Piano model size or checksum differs from the verified export')
    return identity


def verify_worker(root, directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('worker_sources') != worker_sources(root):
        raise ValueError('Audio worker sources changed or are unrecorded; rebuild scripts/build_audio_worker.py')
    if manifest.get('worker_sha256') != sha256(directory/'ManiaLabAudio/ManiaLabAudio.exe'):
        raise ValueError('Audio worker executable does not match its build manifest')
    verify_worker_inventory(directory/'ManiaLabAudio', manifest.get('worker_files'))
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
    verify_worker_inventory(executable.parent, manifest.get('worker_files'))
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
            or verification.get('worker_files') != manifest['worker_files']
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
