"""Verify the user's Sakura Grass reference/finished pair and bundled audio.

Source labels are the mapper's explicitly supplied column interpretation.
Reference taps supervise onset multiplicity, not acoustic release or pitch.
No maps, audio, or event labels are exported to the public repository.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.audio import decode
from manialab.domain import parse_osu, validate
from manialab.structure import reference_marks
from scripts.probe_mapper_adapter import event_similarity


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/sakura-benchmark'))
    args = cli.parse_args()
    root = args.root
    metadata = json.loads((root/'official-pending.json').read_text(encoding='utf-8'))
    official = next(s for s in metadata if s['id'] == 2644803)
    expected = {b['id']: b for b in official['beatmaps']}
    wanted = {5969640: 'reference', 5969639: 'finished'}
    records, charts, audio_files = {}, {}, []
    bundled = root/'bundled'; bundled.mkdir(exist_ok=True)
    with zipfile.ZipFile(root/'original.osz') as pack:
        for info in pack.infolist():
            if info.filename.lower().endswith('.osu'):
                data = pack.read(info)
                chart = parse_osu(data.decode('utf-8-sig'))
                bid = int(chart['metadata']['BeatmapID'])
                if bid not in wanted:
                    continue
                if hashlib.md5(data).hexdigest() != expected[bid]['checksum']:
                    raise ValueError('Mirror chart revision differs from official checksum')
                if chart['keys'] != 10 or chart['metadata']['Creator'] != 'zhyzx35607':
                    raise ValueError('Unexpected mapper or keycount')
                role = wanted[bid]
                (root/(role+'.osu')).write_bytes(data)
                charts[role] = chart
                records[role] = dict(beatmap_id=bid, official_checksum=expected[bid]['checksum'],
                    chart_sha256=digest(data), version=chart['metadata']['Version'],
                    validation=validate(chart['notes'], 10), marks=reference_marks(data.decode('utf-8-sig')))
            elif info.filename.lower().endswith('.mp3'):
                name = info.filename if info.flag_bits & 0x800 else info.filename.encode('cp437').decode('cp932')
                # The mapper confirmed the original CP932-corrupted names and
                # that these are the Nova stems used to author the reference.
                label = name.removeprefix('Sakura Grass').removesuffix('.mp3').lstrip('_') or 'mix'
                key = {'mix':'mix','人声':'vocals','其他':'other','吉他':'guitar',
                       '?斯':'bass','?琴':'piano','鼓点':'drums'}[label]
                data = pack.read(info)
                path = bundled/(key+'.mp3'); path.write_bytes(data)
                work = bundled/key; work.mkdir(exist_ok=True)
                samples = decode(path, work)
                audio_files.append(dict(id=key, archive_name=name, label=label,
                    path=path.relative_to(root).as_posix(), preview=(work/'preview.wav').relative_to(root).as_posix(),
                    sha256=digest(data), samples=len(samples), sample_rate=22050))
    if set(charts) != {'reference', 'finished'}:
        raise ValueError('Both reference and finished maps are required')
    if any(c['general']['AudioFilename'] != 'Sakura Grass.mp3' for c in charts.values()):
        raise ValueError('Unexpected chart audio')
    reference, finished = charts['reference']['notes'], charts['finished']['notes']
    groups = dict(drums=[0,1,2,3], piano=[4,5], other=[6,7,8,9])
    labels = [dict(start=n.start, end=n.end, source=next(s for s, lanes in groups.items() if n.lane in lanes),
        reference_lane=n.lane, hold=n.hold, pitch=None, release_label=n.end if n.hold else None)
        for n in reference]
    # These local labels must never be passed to a blind audio generation probe.
    (root/'source-labels.json').write_text(json.dumps(labels, indent=2), encoding='utf-8')
    duration = round(next(a for a in audio_files if a['id']=='mix')['samples']/22050*1000)
    boundaries = sorted({0, duration, *records['reference']['marks']['bookmarks']})
    segment_rows = []
    for a, b in zip(boundaries, boundaries[1:]):
        local = [n for n in reference if a <= n.start < b]
        row = dict(start_ms=a, end_ms=b, sources={})
        for voice, lanes in groups.items():
            source = [n for n in local if n.lane in lanes]
            onset_counts = Counter(n.start for n in source)
            row['sources'][voice] = dict(objects=len(source), onset_groups=len(onset_counts),
                stacked_groups=sum(c>1 for c in onset_counts.values()),
                holds=sum(n.hold for n in source))
        row['finished'] = dict(objects=sum(a<=n.start<b for n in finished),
            holds=sum(a<=n.start<b and n.hold for n in finished))
        segment_rows.append(row)
    report = dict(beatmapset_id=2644803, source='https://dl.sayobot.cn/beatmaps/download/novideo/2644803',
        archive_sha256=digest((root/'original.osz').read_bytes()), charts=records, audio=audio_files,
        lane_sources={v:[i+1 for i in lanes] for v,lanes in groups.items()},
        annotations=dict(author='user zhyzx35607', vocals_in_reference=False,
            drum_subtypes={'1':'kick','2':'snare','3':'snare','4':'hi-hat',
                           'exceptions':'The mapper notes additional tom and other sounds; column interpretation is predominant, not an exhaustive class label.'},
            piano_multiplicity='Two simultaneous notes represent layered piano events; no MIDI pitch labels.',
            releases='Only reference LN ends have authored duration labels; taps have no acoustic offset truth.',
            bundled_stems='Mapper-confirmed Nova-separated tracks used to author Reference. The proprietary model and separation settings are unknown.'),
        split='development_reference_only; exclude this song and all its difficulties from claims of independent generalization',
        segments=segment_rows,
        reference_to_finished_onset_groups=event_similarity([(t,) for t in sorted({n.start for n in reference})],
            [(t,) for t in sorted({n.start for n in finished})], [30]))
    (root/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(charts={k:r['validation'] for k,r in records.items()},
        audio=[dict(id=r['id'],label=r['label'],duration=r['samples']/22050) for r in audio_files],
        segments=segment_rows), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
