"""Compare piano onset candidates; the Reference has no piano offset truth."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, note_from
from scripts.probe_mapper_adapter import event_similarity, sha
from scripts.probe_sakura_sources import match_source
from scripts.research_piano_layers import snap_function


def evaluate(text, candidates):
    human = [n for n in parse_osu(text)['notes'] if n.lane in (4, 5)]
    snap = snap_function(text)
    result = {str(shift): match_source(human, candidates, shift) for shift in (0., 22.5)}
    result['known_timing_4_division'] = event_similarity(
        [(t,) for t in sorted({n.start for n in human})],
        [(t,) for t in sorted({snap(n.start-22.5) for n in candidates})], [40])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('test-data/sakura-benchmark'))
    args = parser.parse_args()
    text = (args.root / 'reference.osu').read_text(encoding='utf-8-sig')
    files = dict(basic_pitch='detector-baseline/piano-notes.json',
                 piano_specialist='piano-specialist/sakura-nova-piano-notes.json')
    report = dict(results={}, inputs={}, notice='Author piano onset/layer labels only. No acoustic offset or MIDI-pitch accuracy claim. Known timing is not automatic timing.')
    for name, filename in files.items():
        path = args.root / filename
        notes = [note_from(n) for n in json.loads(path.read_text(encoding='utf-8'))]
        report['results'][name] = evaluate(text, notes)
        report['inputs'][name] = sha(path)
    (args.root / 'piano-specialist/comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
