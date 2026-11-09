"""Fetch fixed real-piano members from the versioned SMD ZIP, verifying CRCs."""
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.requests = []
        self.etag = None

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if value < 0:
            raise ValueError('Invalid ZIP seek')
        self.position = value
        return value

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        if size <= 0:
            return b''
        if size > 64*1024*1024:
            raise ValueError('Unexpected large ZIP member request')
        start, end = self.position, self.position+size-1
        url = self.url + f'?manialab_range={start}-{end}'
        request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}',
                                                       'User-Agent': 'ManiaLab-research/0.1'})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Server did not return the requested byte range')
            etag = response.headers.get('ETag')
            if self.etag is not None and etag != self.etag:
                raise ValueError('Remote archive changed')
            self.etag = etag
            data = response.read(size+1)
        if len(data) != size:
            raise ValueError('Incomplete ZIP range')
        self.position += size
        self.requests.append(dict(start=start, end=end, sha256=hashlib.sha256(data).hexdigest()))
        return data


def main():
    root = Path('test-data/smd')
    metadata = json.loads((root/'zenodo-metadata.json').read_text(encoding='utf-8'))
    if metadata['id'] != 13753319:
        raise ValueError('Use the frozen v2 release')
    archive = next(f for f in metadata['files'] if f['key'] == 'SMD-piano_v2.zip')
    protocol = Path('docs/piano-acoustic-protocol.md')
    output = root/'selected-v1'
    if (output/'source.json').exists():
        raise ValueError('Preserve completed dataset')
    output.mkdir(parents=True, exist_ok=True)
    # Freeze the roster using metadata only, before any model receives audio.
    members = json.loads((root/'archive-members.json').read_text(encoding='utf-8'))
    chosen = {}
    for item in sorted(members, key=lambda item: item['name']):
        name = item['name']
        if not name.startswith('wav_22050_mono/') or not name.endswith('.wav'):
            continue
        composer = Path(name).name.split('_')[0].replace('Rachmaninov', 'Rachmaninoff')
        chosen.setdefault(composer, Path(name).stem)
    if len(chosen) != 11:
        raise ValueError('Unexpected composer inventory')
    selection = dict(composers=chosen, protocol_sha256=sha(protocol))
    selection_path = output/'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Selection changed during download')
    selection_path.write_text(json.dumps(selection, indent=2), encoding='utf-8')
    source = RangeFile(archive['links']['self'], archive['size'])
    records = []
    try:
        with zipfile.ZipFile(source) as zipped:
            for composer, stem in chosen.items():
                for directory, extension in [('wav_22050_mono', 'wav'), ('midi', 'mid'), ('csv', 'csv')]:
                    name = f'{directory}/{stem}.{extension}'
                    info = zipped.getinfo(name)
                    data = zipped.read(name)  # zipfile verifies the full member CRC.
                    target = output/directory/Path(name).name
                    target.parent.mkdir(exist_ok=True)
                    if target.exists() and target.read_bytes() != data:
                        raise ValueError('Do not overwrite a different dataset member')
                    target.write_bytes(data)
                    row = dict(composer=composer, member=name, size=len(data),
                               crc32=f'{info.CRC:08x}', sha256=sha(target))
                    records.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        (output/'download-ranges.json').write_text(json.dumps(source.requests, indent=2), encoding='utf-8')
    result = dict(record=13753319, archive=archive, full_archive_md5_verified=False,
                  verified_member_crc=True, records=records, selection=selection,
                  license=metadata['metadata']['license'], metadata_sha256=sha(root/'zenodo-metadata.json'),
                  runner_sha256=sha(__file__), etag=source.etag)
    (output/'source.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
