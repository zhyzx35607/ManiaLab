"""Optional guitar packaging evidence checks without loading or shipping a model."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from manialab import guitar
from scripts.build_provenance import verify_guitar_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package(tmp_path, monkeypatch):
    identity = json.loads((ROOT/'assets/guitar-model.json').read_text(encoding='utf-8'))
    identity['validation'].update(float_inputs='pending', application_chain='pending', portable_runtime='pending')
    model = tmp_path/'assets/deep/guitar-fl.onnx'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'non-neural test graph placeholder')
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    monkeypatch.setattr(guitar, 'MODEL_SHA256', digest)
    identity['sha256'] = digest
    for key in ('license_file','source_license_declaration_file','notice_file'):
        target = tmp_path/identity[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/identity[key], target)
    def save():
        (tmp_path/'assets/guitar-model.json').write_text(json.dumps(identity), encoding='utf-8')
    save()
    return tmp_path, identity, model, save


def test_model_absence_is_supported_without_claiming_it_was_bundled(tmp_path):
    assert verify_guitar_model(tmp_path) is None


def test_optional_identity_retains_pending_validation_and_failed_original_protocol(package):
    root, identity, _, _ = package
    result = verify_guitar_model(root)
    assert result == identity
    assert result['validation']['original_all_window_protocol_passed'] is False
    assert result['validation']['float_inputs'] == 'pending'
    assert result['validation']['application_chain'] == 'pending'
    assert result['validation']['portable_runtime'] == 'pending'
    assert result['default_enabled'] is False and result['predicted_releases_default'] is False


def test_missing_metadata_or_changed_graph_cannot_be_packaged(package):
    root, _, model, save = package
    (root/'assets/guitar-model.json').unlink()
    with pytest.raises(ValueError, match='metadata'):
        verify_guitar_model(root)
    save()
    model.write_bytes(b'other graph')
    with pytest.raises(ValueError, match='checksum'):
        verify_guitar_model(root)


@pytest.mark.parametrize('field,value', [
    ('filename','piano-regression.onnx'), ('license','CC-BY-4.0'),
    ('upstream_checkpoint_sha256','0'*64), ('sample_rate',22050),
    ('source_commit','0'*40), ('model_revision','0'*40),
    ('thresholds',{'onset_threshold':.2,'offset_threshold':.3,'frame_threshold':.1}),
    ('default_enabled',True), ('predicted_releases_default',True),
    ('normalization',True), ('normalization',0),
])
def test_changed_model_contract_is_rejected(package, field, value):
    root, identity, _, save = package
    identity[field] = value
    save()
    with pytest.raises(ValueError, match='identity|opt-in'):
        verify_guitar_model(root)


@pytest.mark.parametrize('field', ['license','source_license_declaration','notice'])
@pytest.mark.parametrize('damage', ['missing','content','path'])
def test_each_license_evidence_file_is_verified(package, field, damage):
    root, identity, _, save = package
    path = root/identity[field+'_file']
    if damage == 'missing':
        path.unlink()
    elif damage == 'content':
        path.write_text('changed notice', encoding='utf-8')
    else:
        identity[field+'_file'] = '../other-license.txt'
        save()
    with pytest.raises(ValueError, match='license/attribution'):
        verify_guitar_model(root)


@pytest.mark.parametrize('field', ['license','source_license_declaration'])
def test_replacing_upstream_declaration_and_its_hash_still_fails(package, field):
    root, identity, _, save = package
    path = root/identity[field+'_file']
    path.write_bytes(b'not the pinned upstream declaration')
    identity[field+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save()
    with pytest.raises(ValueError, match='reviewed revisions'):
        verify_guitar_model(root)


def test_packaged_resource_copy_must_keep_the_same_identity(package):
    root, identity, _, _ = package
    destination = root/'copy/_internal'
    shutil.copytree(root/'assets', destination/'assets')
    shutil.copytree(root/'licenses', destination/'licenses')
    assert verify_guitar_model(root) == verify_guitar_model(destination) == identity
    (destination/'licenses/HFGuitar-NOTICE.txt').write_text('lost attribution', encoding='utf-8')
    with pytest.raises(ValueError, match='license/attribution'):
        verify_guitar_model(destination)
