from pathlib import Path

import pytest

from scripts.package_assets import BASE_FILES, capabilities, stage_assets, start_guide


def assets(root):
    for name in BASE_FILES:
        target = root / 'assets' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(name.encode())
    (root / 'assets/deep/ManiaLabAudio').mkdir()
    (root / 'assets/deep/ManiaLabAudio/ManiaLabAudio.exe').write_bytes(b'worker')
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
    enabled = stage_assets(tmp_path, destination, 'extended')
    assert (destination / 'deep/guitar-fl.onnx').is_file()
    assert (destination / 'instruments/manifest.json').is_file()
    assert '本包未包含' not in start_guide('test', enabled)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match='profile'):
        capabilities('automatic')
