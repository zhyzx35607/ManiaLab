"""Verify the local audio review and publish only aggregate diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous summaries')
    folder = ROOT/'test-data/sakura-benchmark/piano-audio-review-v2'
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    for name, digest in report['artifacts_sha256'].items():
        if Path(name).name != name or sha256(folder/name) != digest:
            raise ValueError('Changed audio review artifact')
    analysis_path = ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json'
    if sha256(analysis_path) != report['identity']['analysis_sha256']:
        raise ValueError('Frozen candidates changed')
    originals = json.loads(analysis_path.read_text(encoding='utf-8'))['candidates']
    cases = []
    for row in report['clips']:
        notes = json.loads((folder/(row['name']+'-notes.json')).read_text(encoding='utf-8'))
        expected = [n for n in originals if n['voice']=='piano' and n['start']<row['end_ms'] and n['end']>row['begin_ms']]
        if notes != expected:
            raise ValueError('Review changed the selected raw candidates')
        previous = folder.parent/'piano-audio-review-v1'
        for mode in ('source', 'synthesis', 'stereo'):
            filename = f"{row['name']}-{mode}.wav"
            if sha256(previous/filename) != sha256(folder/filename):
                raise ValueError('Axis-only revision changed playback audio')
        short_upper = [n for n in notes if n['pitch']>=88 and n['end']-n['start']<=10]
        repeated = [n for n in notes if n['pitch']==69]
        cases.append(dict(**row, upper_register_intervals_at_most_10ms=len(short_upper),
            pitch69_candidates=len(repeated), pitch69_ending_reasons=dict(Counter(n['release_reason'] for n in repeated))))
    result = dict(identity=report['identity'], report_sha256=sha256(folder/'report.json'),
        summarizer_sha256=sha256(__file__), artifacts_verified=len(report['artifacts_sha256']), clips=cases,
        reused_candidates=report['reused_candidates'], raw_candidates_preserved=True,
        axis_revision_audio_identical=True, model_rerun=False, model_listened_to_audio=False,
        human_listening_completed=False, browser_playback_verified=False, musical_acceptance=False,
        limitations=['Short high-pitch events are review candidates, not proven false positives.',
            'A repeated pitch ending at a reattack is not a phrase-end or LN-release label.',
            'Plots and synthesis expose detector output; they do not validate all pitches or releases.',
            'Browser automation initialization failed: failed to write kernel assets (os error 3).'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(artifacts_verified=result['artifacts_verified'], output_sha256=sha256(args.output))), flush=True)


if __name__ == '__main__':
    main()
