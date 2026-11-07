"""Publish exact audit provenance without original audio or chart contents."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--folders',nargs='+',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path('docs/audio-sources.json'))
args=parser.parse_args()
rows=[];failures=[]
for group in args.folders:
    fetch=group/'fetch-report.json'
    if fetch.is_file():
        failures.extend(row|{'group':group.name} for row in json.loads(fetch.read_text(encoding='utf-8')) if not row['ok'])
    for path in sorted(group.glob('*/reference.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        audio=path.parent/record.get('audio_filename','audio.opus')
        if not audio.is_file():
            candidates=[p for p in path.parent.iterdir() if p.suffix.lower() in {'.mp3','.ogg','.opus','.wav','.flac'}]
            if len(candidates)!=1:raise ValueError(f'Ambiguous source audio: {path}')
            audio=candidates[0]
        chart=path.parent/'reference.osu'
        with audio.open('rb') as stream:audio_hash=hashlib.file_digest(stream,'sha256').hexdigest()
        chart_hash=hashlib.sha256(chart.read_bytes()).hexdigest()
        expected=record.get('downloaded_audio_sha256',record.get('audio_sha256'))
        if expected and audio_hash!=expected:raise ValueError(f'Audio hash mismatch: {path}')
        if record.get('chart_sha256') and chart_hash!=record['chart_sha256']:raise ValueError(f'Chart hash mismatch: {path}')
        rows.append(dict(group=group.name,beatmap_id=record['beatmap_id'],beatmapset_id=record.get('beatmapset_id'),
            source=record.get('source'),dataset_revision=record.get('dataset_revision'),
            audio_sha256=audio_hash,chart_sha256=chart_hash,audio_extension=audio.suffix,
            layout_training_split=record.get('layout_training_split'),
            layout_membership_note=record.get('layout_membership_note')))
report=dict(notice='Identifiers and hashes only. Original songs, beatmaps, archives and user audio are not redistributed.',
    songs=rows,download_failures=failures)
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Verified {len(rows)} exact pairs; retained {len(failures)} download failures')
