"""Portable callers and source snapshots use the current application contracts."""
import ast
import hashlib
import io
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import wave

import pytest

from manialab.server import create_app
from manialab.stems import separation_identity
from scripts import verify_portable
from scripts.build_provenance import sha256, source_snapshot_paths
from scripts.package_assets import MAPPER_INSTALLER_FILES, capabilities


ROOT = Path(__file__).resolve().parents[1]


def _wave(at_ms):
    rate = 22050
    samples = bytearray(rate * 2 * 2)
    index = round(at_ms * rate / 1000) * 2
    samples[index:index + 2] = b'\x00\x60'
    content = io.BytesIO()
    with wave.open(content, 'wb') as output:
        output.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        output.writeframes(samples)
    return content.getvalue()


def test_non_core_stem_checks_bind_current_analysis_instead_of_the_selected_candidate(tmp_path):
    app = create_app(ROOT, tmp_path)
    try:
        projects = app.extensions['projects']
        pid, folder = projects.create()
        mix = _wave(0)
        (folder / 'files/song.wav').write_bytes(mix)
        voices = ['guitar', 'piano']
        sources = []
        for key, at_ms in [('a' * 64, 1000), ('b' * 64, 1200)]:
            stem = _wave(at_ms)
            source = dict(source='provided', audio_sha256=hashlib.sha256(mix).hexdigest(),
                set_sha256=key, voices=voices, decoded_files={
                    voice: dict(wav_sha256=hashlib.sha256(stem).hexdigest()) for voice in voices})
            for voice in voices:
                file = folder / 'stems-provided' / key / (voice + '.wav')
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(stem)
            sources.append(source)
        projects.save(dict(id=pid, title='Portable source fixture', artist='Fixture', mode='audio',
            audio='song.wav', variants=[dict(mode='audio', notes=[], source_separation=dict(
                identity=separation_identity(sources[0]), separation=sources[0]))], active=0,
            analysis=dict(separation=sources[1])))
        client = app.test_client()
        base = '/api/projects/' + pid
        state = client.get(base).json
        calls = []

        def request(path):
            response = client.get(path)
            assert response.status_code == 200, response.json
            result = response.json if response.is_json else response.data
            calls.append((path, result))
            return result

        verify_portable.verify_analysis_stem_playback(request, base, state, voices)
        assert len(calls) == 4
        for path, result in calls:
            query = parse_qs(urlsplit(path).query)
            assert query['context'] == ['evidence']
            assert query['source_identity'] == [separation_identity(sources[1])]
            if '/stem/' in path:
                assert result == _wave(1200)
            else:
                peak = result['high'].index(max(result['high']))
                time_ms = (result['first_sample'] + peak * result['step_samples']) * 1000 / result['sample_rate']
                assert abs(time_ms - 1200) < 5
    finally:
        app.extensions['executor'].shutdown()


def test_non_core_stem_checks_reject_missing_analysis_identity_before_reading_audio():
    def unexpected_request(path):
        pytest.fail('An unbound playback request was sent')

    with pytest.raises(AssertionError, match='identity'):
        verify_portable.verify_analysis_stem_playback(
            unexpected_request, '/api/projects/fixture', {'analysis': {'separation': {}}}, ['piano'])


def _snapshot(tmp_path, profile):
    source = ast.parse((ROOT / 'scripts/build_windows.py').read_text(encoding='utf-8'))
    function = next(node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == 'source_snapshot')
    # Execute only the pure fingerprint function; importing the builder would run it.
    namespace = dict(ROOT=tmp_path, hashlib=hashlib, enabled=capabilities(profile),
        MAPPER_INSTALLER_FILES=MAPPER_INSTALLER_FILES,
        source_snapshot_paths=source_snapshot_paths, sha256=sha256)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'source_snapshot-only', 'exec'), namespace)
    return namespace['source_snapshot']


@pytest.fixture
def source_tree(tmp_path):
    files = ['launcher.py', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
        'scripts/build_windows.py', 'scripts/package_assets.py', 'scripts/worker_inventory.py',
        'scripts/build_provenance.py', 'manialab/app.py', 'manialab/tsumugi-source.json',
        'assets/pattern-prior.json', 'assets/arrangement-profiles.json',
        'assets/model-manifest.json', 'assets/piano-model.json', 'assets/guitar-model.json',
        'assets/deep-models.json', 'assets/runtime.py', 'static/index.html',
        'vendor/__init__.py', 'vendor/mapperatorinator/__init__.py',
        'vendor/mapperatorinator/event.py', 'vendor/mapperatorinator/template.osu']
    files += ['assets/' + name for name in MAPPER_INSTALLER_FILES if name != 'runtime/uv.exe']
    for name in files:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'synthetic source input')
    return tmp_path


@pytest.mark.parametrize('relative', ['vendor/__init__.py', 'vendor/mapperatorinator/__init__.py',
    'vendor/mapperatorinator/event.py', 'vendor/mapperatorinator/template.osu', 'scripts/build_windows.py'])
def test_source_snapshot_changes_with_bundled_runtime_inputs(source_tree, relative):
    snapshot = _snapshot(source_tree, 'core')
    before = snapshot()
    assert relative in before
    (source_tree / relative).write_bytes(b'changed runtime or build input')
    after = snapshot()
    assert before[relative] != after[relative]


@pytest.mark.parametrize('relative', ['assets/mapperatorinator-source.zip',
    'assets/mapperatorinator-requirements.txt'])
def test_non_core_snapshot_changes_with_the_installer_source_payload(source_tree, relative):
    snapshot = _snapshot(source_tree, 'standard')
    before = snapshot()
    assert relative in before
    (source_tree / relative).write_bytes(b'changed installer source payload')
    assert snapshot()[relative] != before[relative]
    assert relative not in _snapshot(source_tree, 'core')()
