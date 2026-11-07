"""Fetch a small, reproducible local audio audit set from the public research dataset.

Audio and charts remain in the ignored test-data directory. They are not release
assets. Human maps are references for a mapper's choices, not complete labels of
every audible musical event.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_corpus import DATASET, REVISION, BufferedRangeFile, request_bytes


def choose(records, limit):
    selected, songs, authors = [], set(), set()
    possible = [r for r in records if r.get('eligible') and r.get('audio_hash') and
                r.get('sample_key','').isdigit() and .08 <= r['report']['ln_ratio'] < .9]
    # Prefer 10K and author diversity, with deterministic ordering. This is a
    # development audit set, never silently advertised as an unseen test set.
    possible.sort(key=lambda r: (r['keys'] != 10, r['beatmap_id']))
    for diversity in (True, False):
        for r in possible:
            if r['song_key'] in songs or (diversity and r.get('creator') in authors):
                continue
            selected.append(r);songs.add(r['song_key']);authors.add(r.get('creator'))
            if len(selected) == limit:
                return selected
    return selected


def save_reference(record, output, raw, audio, source):
    key = record['sample_key']; folder = output / key
    folder.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(raw.encode()).hexdigest() != record['sha256']:
        raise ValueError(f'Chart revision changed: {key}')
    if not audio.startswith(b'OggS') or len(audio) > 40 * 1024 * 1024:
        raise ValueError(f'Invalid or oversized reference audio: {key}')
    (folder/'audio.opus').write_bytes(audio)
    (folder/'reference.osu').write_text(raw, encoding='utf-8')
    item = {k: record.get(k) for k in ('beatmap_id','beatmapset_id','title','artist','creator','song_key','audio_hash','keys','report')}
    item.update(sample_key=key, source=source, dataset_revision=REVISION,
                downloaded_audio_sha256=hashlib.sha256(audio).hexdigest(), bytes=len(audio),
                purpose='development audio audit; human map is a style reference, not exhaustive sound truth')
    (folder/'reference.json').write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding='utf-8')
    return item


def fetch(record, output):
    key = record['sample_key']; folder = output / key
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'reference.json'
    if manifest.exists() and (folder/'audio.opus').exists():
        return json.loads(manifest.read_text(encoding='utf-8'))
    query = urllib.parse.urlencode(dict(dataset=DATASET, config='compressed', split='train', offset=int(key), length=1))
    response = json.loads(request_bytes('https://datasets-server.huggingface.co/rows?' + query))
    if not response.get('rows'):
        raise ValueError(f'Viewer only indexes part of the dataset; use fetch_archives for {key}')
    row = response['rows'][0]['row']
    if row['__key__'] != key or REVISION not in row['__url__'] or row['json']['audio_hash'] != record['audio_hash']:
        raise ValueError(f'Dataset row/revision/hash mismatch: {key}')
    raw = next(b['content'] for b in row['json']['beatmaps'] if int(b['beatmap_id']) == record['beatmap_id'])
    url = row['opus'][0]['src']
    if urllib.parse.urlparse(url).hostname != 'datasets-server.huggingface.co':
        raise ValueError('Unexpected dataset audio host')
    audio = request_bytes(url)
    return save_reference(record, output, raw, audio, row['__url__'])


def fetch_archives(records, output):
    """Fetch selected audio members outside the viewer's partial index.

    One header scan per pinned archive; all unrelated audio is skipped. No tar
    path is extracted to disk. Only verified selected bytes are saved.
    """
    grouped = defaultdict(dict)
    prefix = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/compressed/'
    for r in records:
        source = r.get('source', '')
        if not source.startswith(prefix) or not source.endswith('.tar'):
            raise ValueError('Archive records must identify the pinned dataset')
        grouped[source][r['sample_key']] = r
    listing = json.loads(request_bytes(f'https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/compressed?recursive=false&expand=false&limit=100'))
    sizes = {Path(r['path']).name: r['size'] for r in listing}
    for source, selected in grouped.items():
        remote = BufferedRangeFile(source, sizes[source.rsplit('/', 1)[1]])
        content, audio, done = {}, {}, set()
        with tarfile.open(fileobj=remote, mode='r:') as archive:
            for member in archive:
                key, suffix = member.name.rsplit('.', 1) if '.' in member.name else ('', '')
                if key not in selected or not member.isfile() or suffix not in ('json', 'opus'):
                    continue
                if member.size > 40 * 1024 * 1024:
                    raise ValueError('Oversized selected dataset member')
                with archive.extractfile(member) as stream:
                    value = stream.read()
                r = selected[key]
                if suffix == 'json':
                    sample = json.loads(value)
                    if sample['audio_hash'] != r['audio_hash']:
                        raise ValueError('Audio identity changed in pinned archive')
                    content[key] = next(b['content'] for b in sample['beatmaps'] if int(b['beatmap_id']) == r['beatmap_id'])
                else:
                    audio[key] = value
                if key in content and key in audio:
                    yield save_reference(r, output, content.pop(key), audio.pop(key), source)
                    done.add(key)
                    if len(done) == len(selected):
                        break
        if done != set(selected):
            raise ValueError(f'Selected members missing from archive: {sorted(set(selected)-done)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=Path('test-data/corpus'))
    parser.add_argument('--output', type=Path, default=Path('test-data/audio-audit'))
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    records = [json.loads(line) for p in args.corpus.glob('manifest-*.jsonl') for line in p.read_text(encoding='utf-8').splitlines() if line]
    selected = choose(records, args.limit)
    missing = []
    def attempt(record):
        try:
            return fetch(record, args.output)
        except ValueError as error:
            if record.get('source','').endswith('.tar'):
                return record
            raise error
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(attempt, selected):
            if 'bytes' not in result:
                missing.append(result)
                continue
            print(json.dumps({k:result[k] for k in ('beatmap_id','keys','title','creator','bytes')}, ensure_ascii=True), flush=True)
    for result in fetch_archives(missing, args.output) if missing else ():
        print(json.dumps({k:result[k] for k in ('beatmap_id','keys','title','creator','bytes')}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
