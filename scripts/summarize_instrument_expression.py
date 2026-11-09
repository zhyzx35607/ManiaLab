"""Publish aggregate accompaniment results without audio or note-event data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark/instrument-expression-v1'))
    cli.add_argument('--instrument',type=Path,default=Path('test-data/sakura-benchmark/other-float-yourmt3-v1'))
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve published results')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    report=read(args.root/'report.json');source=read(args.instrument/'report.json')
    if (sha(args.instrument/'report.json')!=report['identity']['inference_report_sha256'] or
        sha(args.root/'runner-snapshot.py')!=report['identity']['runner_sha256']):
        raise ValueError('Completed run identity changed')
    result=dict(report,completed_report_sha256=sha(args.root/'report.json'),
        transcription_decoding_errors=source['decoding_errors'],
        transcription_elapsed_seconds=source['elapsed_seconds'],
        summary_runner_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
