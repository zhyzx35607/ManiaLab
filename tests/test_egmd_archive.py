from io import BytesIO
from types import SimpleNamespace
import zipfile

import pytest

from scripts.egmd_archive import RangeReader, read_entry, safe_name


class Response(BytesIO):
    status = 206


class Opener:
    def __init__(self, data, **changes):
        self.data, self.changes = data, changes
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        begin, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
        assert request.get_header('If-match') == '"fixed"'
        response = Response(self.data[begin:end+1])
        response.headers = {'Content-Range': f'bytes {begin}-{end}/{len(self.data)}',
                            'ETag': '"fixed"', 'x-goog-generation': '123'}
        response.headers.update(self.changes)
        return response


def archive_bytes():
    data = BytesIO()
    with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('root/song.mid', b'example midi bytes')
        archive.writestr('root/song.wav', b'PCM example'*200)
    return data.getvalue()


def reader(opener, cache=None):
    return RangeReader(opener, url='https://example.invalid/data.zip', size=len(opener.data),
                       etag='"fixed"', generation='123', cache=cache)


def test_remote_zip_seek_and_crc_read():
    opener = Opener(archive_bytes())
    with zipfile.ZipFile(reader(opener)) as archive:
        assert read_entry(archive, 'root/song.mid') == b'example midi bytes'
        assert read_entry(archive, 'root/song.wav') == b'PCM example'*200


def test_cache_is_checked_and_does_not_redownload(tmp_path):
    opener = Opener(archive_bytes())
    first = reader(opener, tmp_path)
    expected = first.read(10)
    second = reader(opener, tmp_path)
    assert second.read(10) == expected and len(opener.requests) == 1
    next(tmp_path.glob('*.bin')).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Cached'):
        reader(opener, tmp_path).read(10)


@pytest.mark.parametrize('changes', [{'ETag': 'different'}, {'x-goog-generation': '124'},
                                   {'Content-Range': 'bytes 0-0/1'}, {'Content-Encoding': 'gzip'}])
def test_server_identity_and_range_must_match(changes):
    with pytest.raises(ValueError):
        reader(Opener(archive_bytes(), **changes)).read(10)


@pytest.mark.parametrize('name', ['../escape.wav', '/absolute.wav', 'C:/song.wav', 'a\\b'])
def test_paths_cannot_escape_selected_output(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_no_unbounded_download_or_zip_bomb():
    big = RangeReader(SimpleNamespace(), size=100000000)
    with pytest.raises(ValueError, match='unbounded'):
        big.read()
    with zipfile.ZipFile(BytesIO(archive_bytes())) as archive:
        with pytest.raises(ValueError, match='bounded'):
            read_entry(archive, 'root/song.wav', max_size=10)
