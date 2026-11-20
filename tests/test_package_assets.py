import ast
import argparse
from pathlib import Path
import json
import shutil

import pytest

from scripts.package_assets import (
    BASE_FILES,
    CORE_FILES,
    PUBLIC_PROFILE,
    add_profile_argument,
    capabilities,
    stage_assets,
    start_guide,
    verify_core_model_files,
    verify_core_payload,
)
from scripts.worker_inventory import worker_inventory

ROOT = Path(__file__).resolve().parents[1]


def _stage_license_inputs_function():
    source = ast.parse((ROOT / 'scripts/build_windows.py').read_text(encoding='utf-8'))
    function = next(node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == 'stage_license_inputs')
    namespace = {'Path': Path, 'shutil': shutil}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'license-staging-only', 'exec'), namespace)
    return namespace['stage_license_inputs']


def assets(root):
    for name in BASE_FILES:
        target = root / 'assets' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in ('basic-pitch.onnx', 'model-manifest.json'):
            target.write_bytes((ROOT / 'assets' / name).read_bytes())
        else:
            target.write_bytes(name.encode())
    (root / 'assets/deep/ManiaLabAudio').mkdir()
    (root / 'assets/deep/ManiaLabAudio/ManiaLabAudio.exe').write_bytes(b'worker')
    (root / 'assets/deep/manifest.json').write_text(json.dumps(dict(
        worker_files=worker_inventory(root / 'assets/deep/ManiaLabAudio'))), encoding='utf-8')
    return root / 'assets'


def test_standard_profile_excludes_undeclared_local_assets(tmp_path):
    source = assets(tmp_path)
    (source / 'deep/guitar-fl.onnx').write_bytes(b'guitar')
    (source / 'instruments').mkdir()
    (source / 'instruments/private.wav').write_bytes(b'user audio')
    (source / 'deep/piano-regression-historical.onnx').write_bytes(b'old graph')
    (source / 'unrelated.zip').write_bytes(b'research archive')
    destination = tmp_path / 'staged'
    enabled = stage_assets(tmp_path, destination, 'standard')
    assert enabled == capabilities('standard')
    assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} == set(BASE_FILES) | {'deep/ManiaLabAudio/ManiaLabAudio.exe'}
    guide = start_guide('test', enabled)
    assert '本包未包含：吉他 FL 专用模型、Tsumugi' in guide
    assert '可勾选贝斯' not in guide


def test_core_profile_excludes_weights_and_workers(tmp_path):
    assets(tmp_path)
    destination = tmp_path / 'staged-core'
    enabled = stage_assets(tmp_path, destination, 'core')
    assert enabled == capabilities('core')
    assert enabled == dict(audio=True, separation_four=False, separation_six=False,
                           piano=False, guitar=False, instruments=False,
                           pattern_prior=False, mapperatorinator_install=False)
    assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} == set(CORE_FILES)
    guide = start_guide('test', enabled)
    assert '本包未包含：内置四／六声部分离与节拍模型' in guide
    assert '钢琴专用模型' in guide


def test_core_payload_rejects_models_and_accepts_exact_inventory():
    names = ['ManiaLab/_internal/assets/' + name for name in CORE_FILES]
    verify_core_payload(names)
    with pytest.raises(ValueError, match='model checkpoint'):
        verify_core_payload(names + ['ManiaLab/_internal/assets/deep/htdemucs_6s.th'])
    with pytest.raises(ValueError, match='inventory mismatch'):
        verify_core_payload(names + ['ManiaLab/_internal/assets/unlisted-private.wav'])
    with pytest.raises(ValueError, match='installer or runtime tool'):
        verify_core_payload(names + ['ManiaLab/_internal/uv.exe'])


def test_core_model_must_match_pinned_provenance_and_hash(tmp_path):
    source = assets(tmp_path)
    assert verify_core_model_files(source)['sha256'] == \
        '2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec'
    (source / 'basic-pitch.onnx').write_bytes(b'unexpected model')
    destination = tmp_path / 'staged-core'
    with pytest.raises(ValueError, match='checksum'):
        stage_assets(tmp_path, destination, 'core')
    assert not destination.exists()


def test_extended_profile_requires_components_before_staging(tmp_path):
    source = assets(tmp_path)
    destination = tmp_path / 'staged'
    with pytest.raises(ValueError, match='guitar'):
        stage_assets(tmp_path, destination, 'extended')
    assert not destination.exists()
    (source / 'deep/guitar-fl.onnx').write_bytes(b'guitar')
    with pytest.raises(ValueError, match='instruments'):
        stage_assets(tmp_path, destination, 'extended')
    assert not destination.exists()
    (source / 'instruments').mkdir()
    from manialab.instrument_bundle import MODELS, source_manifest
    names = ['manifest.json', 'verification.json', 'LICENSE-tsumugi']
    names += ['checkpoints/best_model_' + name + '.pth' for name in MODELS]
    names += ['source/' + name for name in source_manifest()['source_files']]
    names += ['ManiaLabInstrument/ManiaLabInstrument.exe']
    for name in names:
        target = source / 'instruments' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'fixture')
    (source / 'instruments/manifest.json').write_text(json.dumps(dict(
        worker_files=worker_inventory(source / 'instruments/ManiaLabInstrument'))), encoding='utf-8')
    enabled = stage_assets(tmp_path, destination, 'extended')
    assert (destination / 'deep/guitar-fl.onnx').is_file()
    assert (destination / 'instruments/manifest.json').is_file()
    assert '本包未包含' not in start_guide('test', enabled)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match='profile'):
        capabilities('automatic')


@pytest.mark.parametrize('profile', ('standard', 'extended'))
def test_build_profile_argument_defaults_public_and_keeps_explicit_internal_profiles(profile):
    parser = argparse.ArgumentParser()
    add_profile_argument(parser)

    assert parser.parse_args([]).profile == PUBLIC_PROFILE == 'core'
    assert parser.parse_args(['--profile', profile]).profile == profile


def test_license_staging_includes_only_runtime_trees_for_enabled_components(tmp_path):
    stage_license_inputs = _stage_license_inputs_function()
    source = tmp_path / 'licenses'
    source.mkdir()
    for name in ('uv-MIT.txt', 'runtime', 'deep-runtime', 'instrument-runtime'):
        path = source / name
        if name.endswith('.txt'):
            path.write_text('tracked license', encoding='utf-8')
        else:
            path.mkdir()
            (path / 'inventory.json').write_text('[]', encoding='utf-8')
    (source / 'runtime-inventory.json').write_text('stale', encoding='utf-8')

    for profile, included in (
        ('core', set()),
        ('standard', {'deep-runtime'}),
        ('extended', {'deep-runtime', 'instrument-runtime'}),
    ):
        destination = tmp_path / f'staged-{profile}'
        stage_license_inputs(source, destination, capabilities(profile))
        staged = {path.name for path in destination.iterdir()}
        assert staged == {'uv-MIT.txt'} | included
        assert not (destination / 'runtime-inventory.json').exists()


def test_worker_private_file_rejected_before_staging(tmp_path):
    source = assets(tmp_path)
    (source / 'deep/ManiaLabAudio/private.wav').write_bytes(b'private recording')
    destination = tmp_path / 'staged'
    with pytest.raises(ValueError, match='extra=.*private.wav'):
        stage_assets(tmp_path, destination, 'standard')
    assert not destination.exists()


def test_worker_added_after_inventory_is_not_copied(tmp_path, monkeypatch):
    source = assets(tmp_path)
    import scripts.package_assets as package
    copy = package.shutil.copy2

    def copy_with_new_file(src, dst):
        (source / 'deep/ManiaLabAudio/private.wav').write_bytes(b'late recording')
        return copy(src, dst)

    monkeypatch.setattr(package.shutil, 'copy2', copy_with_new_file)
    destination = tmp_path / 'staged'
    stage_assets(tmp_path, destination, 'standard')
    assert not (destination / 'deep/ManiaLabAudio/private.wav').exists()
