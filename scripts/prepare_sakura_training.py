"""Prepare author-confirmed source labels for real Mapper audio fine-tuning.

This song is development data. Validation is a later section of the SAME
song, with a gap and complete-window filtering; repeated phrases can recur.
No validation result here establishes unseen-song or musical quality.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, sections
from scripts.probe_mapper_adapter import sha

SOURCE_LANES = {'drums': (0, 1, 2, 3), 'piano': (4, 5), 'other': (6, 7, 8, 9)}
SOURCE_TRACKS = {'drums': ('drums',), 'piano': ('piano',), 'other': ('other', 'bass', 'guitar')}
TRAIN_RANGE = (0, 149367)
VALIDATION_RANGE = (188286, 283840)


def source_chart(text, voice):
    """Preserve original timestamps/columns; never invent piano release labels."""
    chart = parse_osu(text)
    if chart['keys'] != 10 or voice not in SOURCE_LANES:
        raise ValueError('This annotation requires the confirmed 10K source columns')
    selected = [n for n in chart['notes'] if n.lane in SOURCE_LANES[voice]]
    if not selected:
        raise ValueError('No annotated objects for source')
    parts = sections(text)
    parts['HitObjects'] = [n.raw for n in selected]
    parts['Editor'] = []
    parts['General'] = [re.sub(r'^AudioFilename:.*', 'AudioFilename: audio.wav', line)
                        for line in parts['General']]
    parts['Metadata'] = [re.sub(r'^Version:.*', 'Version: source-' + voice, line)
                         for line in parts['Metadata']]
    return '\n'.join(['\n'.join(parts.get('', ['osu file format v14']))] +
                     [f'[{name}]\n' + '\n'.join(body) for name, body in parts.items() if name]) + '\n'


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output', type=Path, default=Path('test-data/sakura-benchmark/source-mmrs'))
    args = cli.parse_args()
    import pandas as pd
    import rosu_pp_py
    manifest = json.loads((args.root / 'manifest.json').read_text(encoding='utf-8'))
    if sha(args.root / 'reference.osu') != manifest['charts']['reference']['chart_sha256']:
        raise ValueError('Author reference changed')
    text = (args.root / 'reference.osu').read_text(encoding='utf-8-sig')
    sources = {row['id']: row for row in manifest['audio']}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'manifest.json').exists():
        raise ValueError('Use a fresh dataset directory to preserve the previous experiment')
    rows, metadata = [], defaultdict(list)
    for voice_index, (voice, names) in enumerate(SOURCE_TRACKS.items()):
        tracks = []
        for name in names:
            row = sources[name]
            if sha(args.root / row['path']) != row['sha256']:
                raise ValueError('Source identity changed')
            rate, data = wavfile.read(args.root / row['preview'])
            if rate != 22050 or data.ndim != 1 or len(data) != row['samples']:
                raise ValueError('Prepared source sample clock changed')
            tracks.append(data.astype(np.float32) / 32768)
        if len({len(t) for t in tracks}) != 1:
            raise ValueError('Source lengths differ')
        mixed = sum(tracks)
        gain = min(1., .98 / max(.001, float(abs(mixed).max())))
        decoded = resample_poly(mixed * gain, 320, 441)
        content = source_chart(text, voice)
        for split, bounds in [('train', TRAIN_RANGE), ('validation', VALIDATION_RANGE)]:
            # Synthetic MMRS track IDs are not official beatmap IDs.
            sid = 900000000 + voice_index * 10 + int(split == 'validation')
            folder = args.output / split / 'data' / str(sid)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / 'reference.osu'
            path.write_text(content, encoding='utf-8')
            wavfile.write(folder / 'audio.wav', 16000, np.round(np.clip(decoded, -1, 1) * 32767).astype(np.int16))
            notes = parse_osu(content)['notes']
            stars = rosu_pp_py.Difficulty().calculate(rosu_pp_py.Beatmap(path=str(path))).stars
            row = dict(beatmap_id=sid, original_beatmap_id=5969640, beatmapset_id=2644803,
                group='sakura-grass-development', keys=10, band=voice, source_role=voice, split=split,
                allowed_audio_ranges_ms=[list(bounds)], prepared_chart_sha256=sha(path),
                decoded_audio_sha256=sha(folder / 'audio.wav'), difficulty=stars,
                ln_ratio=sum(n.hold for n in notes) / len(notes), source_tracks=list(names), gain=gain,
                source_hashes={name: sources[name]['sha256'] for name in names},
                reference_sha256=manifest['charts']['reference']['chart_sha256'])
            rows.append(row)
            metadata[split].append(dict(BeatmapSetId=sid, Id=sid, BeatmapSetFolder=str(sid),
                BeatmapFile='reference.osu', AudioFile='audio.wav', ModeInt=3, Ranked=0,
                DifficultyRating=stars, StarRating=[stars] * 7, SubmittedDate=datetime(1970, 1, 1),
                RankedDate=datetime(1970, 1, 1), UserId=-1))
    for split, items in metadata.items():
        pd.DataFrame(items).to_parquet(args.output / split / 'metadata.parquet', index=False)
    report = dict(revision=1, records=rows, ranked_statuses=[0],
        validation_scope='Same-song later sections; repeated phrases may recur. Not unseen-song validation.',
        counts={s: dict(charts=len(items)) for s, items in metadata.items()},
        protocol=['Author-confirmed Nova sources and source-column subsets; vocals excluded.',
            'Original audio clock and reference columns retained. Piano taps are onset/layer labels only.',
            'Entire audio/history/target window must lie within its declared time split.',
            'No song-position, mapper, year, descriptors, bookmarks or Kiai input context.',
            'Whole-source-chart difficulty and LN proportion are declared oracle conditions for this diagnostic.',
            'Source chart is pending, not mislabelled Ranked. No default application activation.'])
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(records=len(rows), validation_scope=report['validation_scope'])), flush=True)


if __name__ == '__main__':
    main()
