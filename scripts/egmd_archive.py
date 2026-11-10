"""Inspect fixed E-GMD ZIP entries through bounded, identity-checked ranges.

Does not download the full 96 GB archive or claim its published SHA256 was
verified. ZIP entry CRC and local SHA256 can be checked after selected reads.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

URL = 'https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip'
SIZE = 96422999145
ETAG = '"510af23329b0472a8349d1aaf8fb98dd"'
GENERATION = '1583866322043301'
PUBLISHED_ARCHIVE_SHA256 = '7d9a264fb4c9eabd9fec09d5f8e333192f529b1a1b845d170279a977ac436053'


class RangeReader(io.RawIOBase):
    def __init__(self, opener, *, url=URL, size=SIZE, etag=ETAG, generation=GENERATION, cache=None):
        self.opener = opener
        self.url = url
        self.size = size
        self.etag = etag
        self.generation = generation
        self.cache = Path(cache) if cache else None
        self.position = 0
        self.requests = []
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
            identity = dict(url=url, size=size, etag=etag, generation=generation)
            path = self.cache/'identity.json'
            if path.exists() and json.loads(path.read_text()) != identity:
                raise ValueError('Cache belongs to a different archive generation')
            path.write_text(json.dumps(identity, indent=2))

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek origin')
        value = offset+(0 if whence == 0 else self.position if whence == 1 else self.size)
        if not 0 <= value <= self.size:
            raise ValueError('Read outside archive')
        self.position = value
        return value

    def read(self, size=-1):
        count = self.size-self.position if size is None or size < 0 else min(size, self.size-self.position)
        if not count:
            return b''
        if count > 64*1024*1024:
            raise ValueError('Refusing unbounded archive read; inspect/select small entries')
        start, end = self.position, self.position+count-1
        cache = self.cache/f'{start}-{count}.bin' if self.cache else None
        proof = cache.with_suffix('.json') if cache else None
        if cache and cache.exists() and proof.exists():
            data = cache.read_bytes()
            prior = json.loads(proof.read_text())
            if len(data) != count or hashlib.sha256(data).hexdigest() != prior['sha256']:
                raise ValueError('Cached byte range changed')
        else:
            url = self.url+'?generation='+self.generation
            request = urllib.request.Request(url, headers={
                'Range': f'bytes={start}-{end}', 'If-Match': self.etag,
                'Accept-Encoding': 'identity', 'User-Agent': 'ManiaLab-research'})
            with self.opener.open(request, timeout=45) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                    raise ValueError('Server did not honor the bounded byte range')
                if response.headers.get('ETag') != self.etag:
                    raise ValueError('Archive generation changed')
                if response.headers.get('x-goog-generation') != self.generation:
                    raise ValueError('Storage generation differs')
                if response.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise ValueError('Unexpected transfer encoding')
                data = response.read(count+1)
            if len(data) != count:
                raise ValueError('Wrong byte range length')
            prior = dict(start=start, count=count, sha256=hashlib.sha256(data).hexdigest())
            if cache:
                cache.write_bytes(data)
                proof.write_text(json.dumps(prior, indent=2))
        self.requests.append(prior)
        self.position += count
        return data


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive member name')
    return path


def read_entry(archive, name, max_size=64*1024*1024):
    """Read one caller-selected entry, validating CRC and limiting expansion."""
    safe_name(name)
    entry = archive.getinfo(name)
    if entry.is_dir() or entry.file_size > max_size or entry.compress_size > max_size:
        raise ValueError('Entry is not a bounded file')
    with archive.open(entry) as stream:
        data = stream.read(max_size+1)
    if len(data) != entry.file_size or len(data) > max_size:
        raise ValueError('Wrong uncompressed length')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to((root/'test-data').resolve()):
        raise ValueError('Use a new local index directory')
    output.mkdir(parents=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': args.proxy, 'https': args.proxy}))
    reader = RangeReader(opener, cache=output/'ranges')
    with zipfile.ZipFile(reader) as archive:
        names = [i.filename for i in archive.infolist()]
        if len(names) != len(set(names)):
            raise ValueError('Ambiguous duplicate archive paths')
        records = []
        for entry in archive.infolist():
            safe_name(entry.filename)
            records.append(dict(name=entry.filename, size=entry.file_size, compressed_size=entry.compress_size,
                                header_offset=entry.header_offset, crc=entry.CRC, directory=entry.is_dir()))
    (output/'entries.json').write_text(json.dumps(records, indent=2))
    report = dict(url=URL, size=SIZE, etag=ETAG, generation=GENERATION, entries=len(records),
                  published_archive_sha256=PUBLISHED_ARCHIVE_SHA256, full_archive_sha256_verified=False,
                  audio_or_midi_entries_read=False, downloaded_bytes=sum(r['count'] for r in reader.requests),
                  requests=reader.requests, entries_sha256=hashlib.sha256((output/'entries.json').read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'requests'}), flush=True)


if __name__ == '__main__':
    main()
