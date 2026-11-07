"""Fetch selected public beatmap audio for local research; never extract an archive.

The input is an explicit corpus manifest list. Chart bytes stay pinned to the
local hash; the public mirror supplies the audio named in that chart. Source
URLs and both audio/archive hashes are retained. Downloaded files are private
research inputs and must not be included in a release.
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
    if not archive_path.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'ManiaLab-local-research'})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read(100*1024*1024+1)
        if len(content) > 100*1024*1024:
            raise ValueError('Reference pack exceeds the 100MB research limit')
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if not archive.infolist():
                raise ValueError('Empty reference archive')
        archive_path.write_bytes(content)
    with zipfile.ZipFile(archive_path) as archive:
        matching = [item for item in archive.infolist() if not item.is_dir() and
                    (item.filename.replace('\\', '/') == name or item.filename.replace('\\', '/').endswith('/'+name))]
        if len(matching) != 1:
            raise ValueError('The pinned chart audio is missing or ambiguous in this pack revision')
        if matching[0].file_size > 100*1024*1024:
            raise ValueError('Reference audio exceeds the 100MB research limit')
        audio = archive.read(matching[0])
    destination = target / ('audio'+extension)
    destination.write_bytes(audio)
    (target/'reference.osu').write_bytes(chart)
    saved = {key: record.get(key) for key in ('beatmap_id', 'beatmapset_id', 'title', 'artist', 'creator', 'keys', 'group', 'layout_training_split')}
    saved.update(audio_filename=destination.name, chart_sha256=record['sha256'], source=url,
                 audio_sha256=hashlib.sha256(audio).hexdigest(),
                 archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                 notice='Public mirror audio paired by the pinned chart filename; not redistributed, not an exhaustive transcription label')
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
