from __future__ import annotations

import io
import zipfile

import pytest

from scripts.fetch_paired_maps import _find_audio_member
from scripts.sayo_charts import RemoteZip


def _archive_bytes():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("audio.mp3", bytes(range(256)) * 32)
        archive.writestr("unrelated.bin", b"x" * (2 * 1024 * 1024))
    return output.getvalue()


def test_remote_zip_reads_only_the_audio_ranges():
    payload = _archive_bytes()
    requests = []

    def fetch(url, headers=None, include_headers=False):
        range_header = headers["Range"]
        if range_header.startswith("bytes=-"):
            size = int(range_header.removeprefix("bytes=-"))
            start, end = max(0, len(payload) - size), len(payload) - 1
        else:
            start, end = map(int, range_header.removeprefix("bytes=").split("-"))
        requests.append((start, end))
        result = (
            payload[start:end + 1],
            {
                "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                "ETag": '"test-etag"',
            },
            url,
        )
        return result if include_headers else result[0]

    remote = RemoteZip(123, fetch, url="https://example.test/large.osz")
    with zipfile.ZipFile(remote) as archive:
        audio = archive.read(_find_audio_member(archive, "audio.mp3"))

    assert audio == bytes(range(256)) * 32
    assert len(requests) > 1
    assert remote.bytes_fetched < len(payload)


def test_audio_lookup_rejects_ambiguous_pack_members():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("first/audio.mp3", b"first")
        archive.writestr("second/audio.mp3", b"second")

    with zipfile.ZipFile(io.BytesIO(output.getvalue())) as archive:
        with pytest.raises(ValueError, match="missing or ambiguous"):
            _find_audio_member(archive, "audio.mp3")
