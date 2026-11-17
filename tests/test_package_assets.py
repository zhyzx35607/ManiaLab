from pathlib import Path
import json

import pytest

from scripts.package_assets import BASE_FILES, capabilities, stage_assets, start_guide
from scripts.worker_inventory import worker_inventory


def assets(root):
    for name in BASE_FILES:
        target = root / 'assets' / name
        target.parent.mkdir(parents=True, exist_ok=True)
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
