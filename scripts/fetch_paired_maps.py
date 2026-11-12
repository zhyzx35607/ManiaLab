"""Fetch selected public beatmap audio for local research; never extract an archive.

The input is an explicit corpus manifest list. Chart bytes stay pinned to the
local hash; the public mirror supplies the audio named in that chart. Source
URLs and audio hashes are retained. Small archives also keep their complete
SHA; oversized archives are read by HTTP range and recorded by ETag and size.
Downloaded files are private research inputs and must not be included in a release.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from manialab.storage import AUDIO
from scripts.sayo_charts import RemoteZip


MAX_FULL_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_RANGE_BYTES = 8 * 1024 * 1024


def _find_audio_member(archive, name):
    normalized = name.replace('\\', '/')
    matching = [
        item for item in archive.infolist()
        if not item.is_dir()
        and (item.filename.replace('\\', '/') == normalized
             or item.filename.replace('\\', '/').endswith('/' + normalized))
    ]
    if len(matching) != 1:
        raise ValueError('The pinned chart audio is missing or ambiguous in this pack revision')
    if matching[0].file_size > MAX_AUDIO_BYTES:
        raise ValueError('Reference audio exceeds the 100MB research limit')
    return matching[0]


def _range_request(url, headers=None, include_headers=False):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ManiaLab-local-research"} | (headers or {}),
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read(MAX_RANGE_BYTES + 1)
        if len(content) > MAX_RANGE_BYTES:
            raise ValueError('ZIP range response exceeds the 8MB limit')
        response_headers = dict(response.headers)
        final_url = response.url
    return (content, response_headers, final_url) if include_headers else content


def _read_remote_audio(sid, url, name):
    remote = RemoteZip(sid, _range_request, url=url)
    try:
        with zipfile.ZipFile(remote) as archive:
            audio = archive.read(_find_audio_member(archive, name))
        metadata = {
            "archive_sha256": None,
            "archive_url": remote.url,
            "archive_etag": remote.etag,
            "archive_size_bytes": remote.size,
            "archive_bytes_fetched": remote.bytes_fetched,
            "archive_read_mode": "http-range",
        }
        return audio, metadata
    finally:
        remote.close()


def fetch(record, corpus, output):
    bid, sid = int(record['beatmap_id']), int(record['beatmapset_id'])
    if bid <= 0 or sid <= 0:
        raise ValueError('A public beatmap and set ID are required')
    target = output / str(bid)
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / 'reference.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        audio = target / saved['audio_filename']
        reference = target/'reference.osu'
        if (saved.get('chart_sha256') == record['sha256'] and reference.is_file()
                and hashlib.sha256(reference.read_bytes()).hexdigest() == record['sha256']
                and audio.exists() and hashlib.sha256(audio.read_bytes()).hexdigest() == saved['audio_sha256']):
            return saved
    chart = (corpus / 'maps' / f'{bid}.osu').read_bytes()
    if hashlib.sha256(chart).hexdigest() != record['sha256']:
        raise ValueError(f'Chart revision mismatch: {bid}')
    parsed = parse_osu(chart.decode('utf-8-sig'))
    name = parsed['general']['AudioFilename'].replace('\\', '/')
    extension = Path(name).suffix.lower()
    if extension not in AUDIO:
        raise ValueError('Unsupported audio filename in reference chart')
    url = f'https://dl.sayobot.cn/beatmaps/download/novideo/{sid}'
    archive_path = target / 'original.osz'
    archive_metadata = None
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            audio = archive.read(_find_audio_member(archive, name))
        archive_metadata = {
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_read_mode": "full",
        }
    else:
        remote = None
        try:
            remote = RemoteZip(sid, _range_request, url=url)
        except (OSError, ValueError, zipfile.BadZipFile):
            pass
        if remote is not None and remote.size > MAX_FULL_ARCHIVE_BYTES:
            try:
                with zipfile.ZipFile(remote) as archive:
                    audio = archive.read(_find_audio_member(archive, name))
                archive_metadata = {
                    "archive_sha256": None,
                    "archive_url": remote.url,
                    "archive_etag": remote.etag,
                    "archive_size_bytes": remote.size,
                    "archive_bytes_fetched": remote.bytes_fetched,
                    "archive_read_mode": "http-range",
                }
            finally:
                remote.close()
        else:
            if remote is not None:
                remote.close()
            request = urllib.request.Request(url, headers={'User-Agent': 'ManiaLab-local-research'})
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read(MAX_FULL_ARCHIVE_BYTES + 1)
            if len(content) > MAX_FULL_ARCHIVE_BYTES:
                audio, archive_metadata = _read_remote_audio(sid, url, name)
            else:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    audio = archive.read(_find_audio_member(archive, name))
                archive_path.write_bytes(content)
                archive_metadata = {
                    "archive_sha256": hashlib.sha256(content).hexdigest(),
                    "archive_size_bytes": len(content),
                    "archive_read_mode": "full",
                }
    destination = target / ('audio'+extension)
    destination.write_bytes(audio)
    (target/'reference.osu').write_bytes(chart)
    saved = {key: record.get(key) for key in ('beatmap_id', 'beatmapset_id', 'title', 'artist', 'creator', 'keys', 'group', 'layout_training_split')}
    saved.update(audio_filename=destination.name, chart_sha256=record['sha256'], source=url,
                 audio_sha256=hashlib.sha256(audio).hexdigest(),
                 notice='Public mirror audio paired by the pinned chart filename; not redistributed, not an exhaustive transcription label')
    saved.update(archive_metadata)
    manifest.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', type=Path)
    parser.add_argument('--corpus', type=Path, default=Path('test-data/corpus'))
    parser.add_argument('--output', type=Path, default=Path('test-data/audio-additional-10k'))
    args = parser.parse_args()
    records = json.loads(args.selection.read_text(encoding='utf-8'))
    def attempt(record):
        try:
            return dict(ok=True, reference=fetch(record, args.corpus, args.output))
        except Exception as error:
            return dict(ok=False, beatmap_id=record['beatmap_id'], error=str(error))
    outcomes = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for outcome in pool.map(attempt, records):
            outcomes.append(outcome)
            print(json.dumps(outcome, ensure_ascii=True), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'fetch-report.json').write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding='utf-8')
    if any(not r['ok'] for r in outcomes):
        raise SystemExit('Some paired sources could not be verified; inspect fetch-report.json')
