"""Build a local song/stem audition from verified raw Mapper probe outputs.

No human transcription is assumed. This only replays existing predictions;
changing the audible stem does not attribute notes to that instrument.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.audio import decode
from manialab.domain import parse_osu
from scripts.probe_mapper_adapter import sha, write_review


def read_probe(path, audio_hash):
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['context']['audio_sha256'] != audio_hash:
        raise ValueError('Probe input differs from the supplied original song')
    raw = path.parent / result['raw_file']
    if sha(raw) != result['raw_sha256']:
        raise ValueError('Raw model output changed after evaluation')
    start = result['context']['start_ms']
    end = start + result['context']['seconds'] * 1000
    notes = parse_osu(raw.read_text(encoding='utf-8-sig'))['notes']
    result['notes'] = [n.data() for n in notes if n.start < end and n.end >= start]
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--audio', type=Path, required=True)
    cli.add_argument('--stems', type=Path, required=True)
    cli.add_argument('--root', type=Path, default=Path('test-data'))
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    audio_hash = sha(args.audio)
    separation = json.loads((args.stems / 'separation.json').read_text(encoding='utf-8'))
    if separation['audio_sha256'] != audio_hash:
        raise ValueError('Stems belong to a different source song')
    locations = dict(
        default=args.root / 'mapper-stem-probe/mix/result.json',
        baseline=args.root / 'mapper-mix-condition/baseline/mix/result.json',
        adapter=args.root / 'mapper-mix-condition/adapter/mix/result.json')
    results = {name: read_probe(path, audio_hash) for name, path in locations.items()}
    expected = results['baseline']['context']
    for row in results.values():
        for key in ('start_ms', 'seconds', 'bpm', 'offset', 'keys', 'seed', 'ln_ratio', 'source_commit', 'model_revision'):
            if row['context'][key] != expected[key]:
                raise ValueError('Song probes do not share time, checkpoint and chart conditions')
    if results['adapter']['context'].get('conditioning') != expected.get('conditioning'):
        raise ValueError('Baseline and adapter token conditions differ')
    labels = dict(default='原模型 · 默认条件', baseline='原模型 · 训练条件', adapter='实验微调 · 训练条件')
    sources = {name: dict(label=label, path=args.stems / (name + '.wav')) for name, label in
        dict(drums='分离鼓声', bass='分离贝斯', vocals='分离人声', piano='分离钢琴', guitar='分离吉他', other='分离其余伴奏').items()
        if (args.stems / (name + '.wav')).is_file()}
    args.output.mkdir(parents=True, exist_ok=True)
    # Decode the supplied source again so a stale cached preview cannot change
    # the listening reference. All variants retain the original song clock.
    decode(args.audio, args.output)
    write_review(args.output, args.output/'preview.wav', expected['start_ms'],
        expected['start_ms'] + expected['seconds']*1000, results,
        labels=labels, sources=sources, keys=expected['keys'],
        notice='这首歌没有人工转录真值，三张图都是模型预测。原模型默认条件用于检查条件差异；原模型训练条件与微调训练条件才是本轮配对比较。六声部来自公开 Demucs，可能有串音。未经质量验收的微调权重未启用。')
    audit = dict(audio_sha256=audio_hash, no_human_reference=True,
        separation_model_sha256=separation['model_sha256'],
        audible_stems={name: sha(track['path']) for name, track in sources.items()},
        variants={name: {key: row[key] for key in ('context', 'raw_sha256', 'validation')} for name, row in results.items()})
    (args.output/'provenance.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(dict(review=str(args.output/'review.html'), variants=list(results), stems=list(sources))))


if __name__ == '__main__':
    main()
