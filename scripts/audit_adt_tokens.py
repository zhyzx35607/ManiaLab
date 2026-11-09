"""Independently audit every saved ADT chunk before any musical score."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.percussion_tokens import audit_tokens, decode_complete
from scripts.probe_mapper_adapter import sha


def audit_run(run):
    run = Path(run)
    report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    identity = report['identity']
    if (sha(run/'tokens.npz') != report['tokens_sha256'] or sha(run/'events.json') != report['events_sha256']):
        raise ValueError('Saved ADT inference changed')
    seconds = report['duration_ms'] / 1000
    chunk_seconds = identity['chunk_samples'] / identity['sample_rate']
    expected = int(np.ceil(seconds / chunk_seconds))
    chunks = []
    exact_events = []
    with np.load(run/'tokens.npz', allow_pickle=False) as saved:
        if set(saved.files) != {str(i) for i in range(expected)}:
            raise ValueError('Missing or unexpected drum chunks')
        for index in range(expected):
            row = saved[str(index)]
            if row.ndim != 1 or row.dtype.kind not in 'iu':
                raise ValueError('Invalid saved token array')
            maximum = min(255, int(np.ceil(min(chunk_seconds, seconds-index*chunk_seconds)*100))-1)
            issues = audit_tokens(row, maximum)
            chunks.append(dict(chunk=index, tokens=len(row), max_time_index=maximum, issues=dict(Counter(issues))))
            if not issues:
                exact_events.extend(dict(onset=e['onset']+index*chunk_seconds, custom_pitch=e['custom_pitch'], velocity=e['velocity'])
                                    for e in decode_complete(row))
    totals = Counter()
    for chunk in chunks:
        totals.update(chunk['issues'])
    agreement = None
    if not totals:
        actual = json.loads((run/'events.json').read_text(encoding='utf-8'))
        ordered = lambda rows: sorted(rows, key=lambda e: (e['onset'], e['custom_pitch'], e['velocity']))
        exact_events, actual = ordered(exact_events), ordered(actual)
        agreement = len(exact_events) == len(actual) and all(
            abs(left['onset']-right['onset']) < 1e-6 and
            (left['custom_pitch'], left['velocity']) == (right['custom_pitch'], right['velocity'])
            for left, right in zip(exact_events, actual))
        if not agreement:
            totals['exact_decoder_disagreement'] += 1
    return dict(report_sha256=sha(run/'report.json'), tokens_sha256=report['tokens_sha256'],
        events_sha256=report['events_sha256'], audit_sha256=sha('manialab/percussion_tokens.py'),
        chunks=chunks, issues=dict(totals), exact_decoder_agreement=agreement,
        suitable_for_event_scoring=not totals)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier audit')
    result = audit_run(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key != 'chunks'}), flush=True)


if __name__ == '__main__':
    main()
