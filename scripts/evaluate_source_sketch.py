"""Compare detected source-onset sketch with the confirmed author reference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import event_similarity,sha
from manialab.domain import parse_osu


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--sketch',type=Path,default=Path('test-data/sakura-benchmark/detected-onset-sketch-v1'))
    args=cli.parse_args();root=args.root;folder=args.sketch
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    manifest=read(root/'manifest.json');report=read(folder/'report.json')
    if (sha(root/'reference.osu')!=manifest['charts']['reference']['chart_sha256'] or
        sha(folder/'sketch.osu')!=report['sketch_sha256']):raise ValueError('Frozen source reference or sketch changed')
    reference=parse_osu((root/'reference.osu').read_text(encoding='utf-8-sig'))['notes']
    predicted=parse_osu((folder/'sketch.osu').read_text(encoding='utf-8-sig'))['notes']
    rows={}
    for name,start,end in [('fit',0,162340),('calibration',162340,188286),
        ('later_sections',188286,283840),('probe_excerpt',215853,227853)]:
        rows[name]={}
        for role,lanes in [('drums',range(4)),('piano',range(4,6)),('other',range(6,10))]:
            human=[n for n in reference if n.lane in lanes and start<=n.start<end]
            generated=[n for n in predicted if n.lane in lanes and start<=n.start<end]
            rows[name][role]=dict(onset_groups=event_similarity([(t,) for t in sorted({n.start for n in human})],
                [(t,) for t in sorted({n.start for n in generated})],[40]),
                objects=event_similarity([(n.start,) for n in human],[(n.start,) for n in generated],[40]))
    result=dict(rows=rows,reference_sha256=sha(root/'reference.osu'),sketch_sha256=sha(folder/'sketch.osu'),
        notice='Author Timing supplied. Same-song learned source choices, all taps; no claim about LN endpoints.')
    (folder/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(rows['probe_excerpt']),flush=True)


if __name__=='__main__':main()
