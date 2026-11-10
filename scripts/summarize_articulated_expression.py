"""Publish full expression comparisons including rejected intermediate rules."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--runs',type=Path,nargs='+',required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve reports')
    rows=[]
    for folder in args.runs:
        r=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if sha(folder/'runner-snapshot.py')!=r['identity']['runner_sha256']:
            raise ValueError('Comparison runner snapshot changed')
        for name,digest in r['identity']['snapshots'].items():
            if sha(folder/(Path(name).stem+'-snapshot.py'))!=digest:raise ValueError('Runtime snapshot changed')
        rows.append(dict(name=folder.name,evidence_sha256=sha(folder/'report.json'),**r))
    args.output.write_text(json.dumps(dict(runs=rows,automatic_recommendation_enabled=False,
        optional_manual_mode=True,notice='Original automatic policy retained. Alternatives need music/phrase validation; all run results retained.'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
