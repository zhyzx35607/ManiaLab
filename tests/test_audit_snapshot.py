import hashlib

import pytest

from scripts.audit_snapshot import assert_snapshot


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('current,archived', [
    (b'value = 1\n', b'value = 1\r\n'),
    (b'value = 1\r\n', b'value = 1\n'),
])
def test_git_newline_conversion_is_explicitly_equivalent(tmp_path, current, archived):
    (tmp_path / 'file.py').write_bytes(current)
    assert assert_snapshot({'file.py': digest(archived)}, {'file.py': digest(current)}, tmp_path) == ['file.py']


def test_actual_source_change_is_not_masked(tmp_path):
    current, archived = b'value = 2\n', b'value = 1\r\n'
    (tmp_path / 'file.py').write_bytes(current)
    with pytest.raises(ValueError, match='snapshot mismatch'):
        assert_snapshot({'file.py': digest(archived)}, {'file.py': digest(current)}, tmp_path)


def test_edit_after_start_and_missing_files_are_rejected(tmp_path):
    initial = {'file.py': digest(b'value = 1\n')}
    (tmp_path / 'file.py').write_bytes(b'value = 2\n')
    with pytest.raises(ValueError, match='after evaluation started'):
        assert_snapshot(initial, initial, tmp_path)
    with pytest.raises(ValueError, match='file list differs'):
        assert_snapshot({}, initial, tmp_path)
