"""Publish aggregate score-event checks after verifying frozen local artifacts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'Artifact identity changed: {path.name}')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT / 'test-data/piano-svsep')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous public summaries')
    folder = args.root / 'score-events-v1'
    report = read(folder / 'report.json')
    identity = report['identity']
    for snapshot, key in [('runner-snapshot.py', 'runner_sha256'), ('interface-snapshot.py', 'interface_sha256'),
                          ('protocol-snapshot.md', 'protocol_sha256')]:
        verify(folder / snapshot, identity[key])
    for filename, digest in report['artifacts_sha256'].items():
        if Path(filename).name != filename:
            raise ValueError('Unexpected artifact path')
        verify(folder / filename, digest)
    old = args.root / 'bar-identity-v1'
    verify(old / 'report.json', identity['source_report_sha256'])
    old_report = read(old / 'report.json')
    for filename, digest in old_report['artifacts_sha256'].items():
        if Path(filename).name != filename:
            raise ValueError('Unexpected original artifact path')
        verify(old / filename, digest)
    if report['completed'] != 7 or report['errors'] != 0:
        raise ValueError('Incomplete score-event controls')
    for flag in ['model_loaded', 'training_cache_read', 'training_cache_changed', 'weights_changed',
                 'enabled_in_application', 'musical_acceptance']:
        if report[flag] is not False:
            raise ValueError(f'Unexpected scope: {flag}')
    for case in report['cases']:
        detail = read(folder / (case['name'] + '-detail.json'))
        events, notation = detail['known']['events'], detail['known']['notation']
        if not all(case['checks'].values()) or detail['checks'] != case['checks']:
            raise ValueError('Unpassed control')
        counts = dict(written_count=len(detail['source_written']), tie_count=len(detail['authored_edges']),
                      event_count=len(events), attack_count=sum(v['role'] == 'attack' for v in notation),
                      continuation_count=sum(v['role'] == 'continuation' for v in notation))
        if any(case[field] != count for field, count in counts.items()):
            raise ValueError('Aggregate count differs from local detail')
    result = dict(date='2026-11-09', summarizer_sha256=sha256(__file__),
                  report_sha256=sha256(folder / 'report.json'), identity=identity,
                  verification=dict(local_artifacts=len(report['artifacts_sha256']),
                                    original_artifacts=len(old_report['artifacts_sha256']),
                                    detail_counts_reproduced=True, cases=7,
                                    passed_checks=sum(len(case['checks']) for case in report['cases'])),
                  cases=report['cases'], elapsed_seconds=report['elapsed_seconds'],
                  limitations=report['limitations'] + [
                      'Written staff, voice and measure fields do not establish global melody identity.',
                      'Only summary counts and identities are published; note arrays and event IDs remain local.'],
                  model_loaded=False, training_cache_read=False, training_cache_changed=False,
                  weights_changed=False, enabled_in_application=False, musical_acceptance=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), sha256=sha256(args.output),
                         verification=result['verification'])), flush=True)


if __name__ == '__main__':
    main()
