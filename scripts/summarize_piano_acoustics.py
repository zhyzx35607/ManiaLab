"""Publish aggregate evidence, never source audio or per-note observations."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--evaluation', type=Path, required=True)
    cli.add_argument('--portability', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing reports')
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    evaluated, portable = read(args.evaluation), read(args.portability)
    inputs = dict(evaluation=args.evaluation, portability=args.portability,
                  application=Path('test-data/sakura-benchmark/piano-application-v1/report.json'),
                  api=Path('test-data/piano-onnx/api-check-v1/verification.json'),
                  single_window=Path('test-data/piano-onnx/export-v1/runtime-verification.json'),
                  data_source=Path('test-data/smd/selected-v1/source.json'))
    application, api, window, source = (read(inputs[key]) for key in ('application', 'api', 'single_window', 'data_source'))
    if not portable['passed'] or len(evaluated['rows']) != 11:
        raise ValueError('Incomplete verification')
    result = dict(date='2026-11-09', evidence_sha256={k: sha(v) for k,v in inputs.items()},
        dataset=dict(record=source['record'], license=source['license'],
                     members=len(source['records']), member_crc_verified=source['verified_member_crc'],
                     full_archive_md5_verified=source['full_archive_md5_verified'],
                     manifest_sha256=evaluated['manifest_sha256']),
        models=evaluated['models'], summary=evaluated['summary'],
        rows=[dict(name=row['name'], audio_sha256=row['audio_sha256'], score_seconds=row['score_seconds'],
                   models={key:value['metrics'] for key,value in row['models'].items()}) for row in evaluated['rows']],
        portability=portable, single_window=window,
        application={k:application[k] for k in ('cold_inference','same_sample_array','raw_candidates',
            'application_candidates','application_reference_diagnostic','release_reasons')},
        api={k:api[k] for k in ('runs','reopen_preserved','visual_check','model_sha256')},
        auto_timing={k:application['auto_timing'][k] for k in ('bpm','offset','confidence','stable')},
        default_enabled=False, optional_piano_backend=True, release_published=False,
        notice='Natural-piano evaluation and optional application integration only. Authored layer selection, automatic timing and full LN mapping are not accepted.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(str(args.output))


if __name__ == '__main__':
    main()
