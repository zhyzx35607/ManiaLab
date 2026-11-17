import pytest

from scripts.worker_inventory import verify_worker_inventory, worker_inventory


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path/'worker'
    (root/'_internal').mkdir(parents=True)
    (root/'worker.exe').write_bytes(b'executable')
    (root/'_internal/runtime.dll').write_bytes(b'original dll')
    return root, worker_inventory(root)


@pytest.mark.parametrize('change, message', [
    ('extra', 'extra=.*private.wav'),
    ('missing', 'missing=.*runtime.dll'),
    ('changed', 'files changed.*runtime.dll'),
])
def test_rejects_changed_runtime_tree(runtime, change, message):
    root, expected = runtime
    assert verify_worker_inventory(root, expected) == expected
    if change == 'extra':
        (root/'private.wav').write_bytes(b'private recording')
    elif change == 'missing':
        (root/'_internal/runtime.dll').unlink()
    else:
        (root/'_internal/runtime.dll').write_bytes(b'modified dll')
    with pytest.raises(ValueError, match=message):
        verify_worker_inventory(root, expected)


def test_legacy_worker_requires_new_build(runtime):
    root, _ = runtime
    with pytest.raises(ValueError, match='rebuild'):
        verify_worker_inventory(root, None)


@pytest.mark.parametrize('name', ['../private.wav', '/private.wav', 'D:/private.wav', 'a\\b', 'a//b'])
def test_rejects_unsafe_manifest_paths(runtime, name):
    root, expected = runtime
    expected[name] = expected['worker.exe']
    with pytest.raises(ValueError, match='Unsafe'):
        verify_worker_inventory(root, expected)
