"""Compare suggested section boundaries with author bookmarks, without fitting.

Bookmarks are one author's phrase plan, not exhaustive structural truth.
Start/end housekeeping segments are excluded by the model's label, not by
looking at the reference. Every other boundary participates in matching.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.structure import reference_marks
from scripts.probe_mapper_adapter import event_similarity,sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    args=cli.parse_args();root=args.root
    chart=root/'reference.osu'
    human=[(t+22.5,) for t in reference_marks(chart.read_text(encoding='utf-8-sig'))['bookmarks']]
    baseline=json.loads((root/'provided-pipeline/analysis.json').read_text(encoding='utf-8'))
    variants={'existing_acoustic_change':dict(boundaries=baseline['structure']['boundaries'][1:-1],
        source_sha256=sha(root/'provided-pipeline/analysis.json'))}
    for name in ('structure-allin1','structure-allin1-ensemble'):
        path=root/name/'report.json'
        if not path.exists():continue
        row=json.loads(path.read_text(encoding='utf-8'));segments=row['segments']
        boundaries=[round(s['start']*1000) for prev,s in zip(segments[:-1],segments[1:])
                    if prev['label']!='start' and s['label']!='end']
        variants[name]=dict(boundaries=boundaries,identity=row['identity'],elapsed_seconds=row['elapsed_seconds'],
            kernel_check=row['kernel_check'],label_counts=dict(Counter(s['label'] for s in segments)))
    for row in variants.values():
        predicted=[(t,) for t in row.pop('boundaries')]
        row['boundary_count']=len(predicted)
        row['comparison']={str(tolerance):event_similarity(human,predicted,[tolerance]) for tolerance in (500,1000,3000)}
    report=dict(protocol=__doc__,reference_sha256=sha(chart),bookmark_count=len(human),audio_shift_ms=22.5,
        variants=variants,enabled_in_application=False,
        notice='Same-song development, not unseen evaluation. Timing is not taken from the structure model. Native NATTEN equivalence and generalization remain to be checked.')
    (root/'structure-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:row['comparison']['1000'] for name,row in variants.items()}),flush=True)


if __name__=='__main__':main()
