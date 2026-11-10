"""Recompute joint-head development decisions and publish aggregate results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.train_piano_onset_head import pooled
from scripts.piano_joint_selection import compare_recording, eligible, select_epoch
from scripts.evaluate_piano_acoustics import metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cache', type=Path, required=True)
    cli.add_argument('--run', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    cache, run = args.cache.resolve(), args.run.resolve()
    source, report = read(cache/'report.json'), read(run/'report.json')
    if args.output.exists():
        raise ValueError('Preserve earlier aggregate')
    if report['identity']['cache_report_sha256'] != sha256(cache/'report.json'):
        raise ValueError('Wrong source cache')
    for key in ('evaluate_piano_acoustics', 'piano_joint_selection', 'train_piano_onset_head'):
        if report['identity']['code'][key] != sha256(ROOT/'scripts'/(key+'.py')):
            raise ValueError('Current selection/scoring differs from recorded implementation')
    for folder, saved in [(cache, source), (run, report)]:
        for name, digest in saved['artifacts_sha256'].items():
            if sha256(folder/name) != digest:
                raise ValueError('Saved evidence changed: '+name)
    baseline = report['epochs'][0]['validation']
    epochs = []
    scope = lambda notes: [n for n in notes if 0 <= n['onset'] < 170]
    for epoch in report['epochs']:
        number = epoch['epoch']
        comparisons, diagnostics = {}, []
        for row in epoch['validation']:
            name = row['id']
            labels = read(cache/(name+'-labels.json'))
            key, sustain = scope(labels['notes']), scope(labels['sustain_notes'])
            before = scope(read(run/(name+'-epoch0-events.json'))['notes'])
            after = scope(read(run/(name+f'-epoch{number}-events.json'))['notes'])
            observed = {kind: {label: metrics(ref, after, strict=strict)
                              for label, strict in [('50ms', False), ('40ms', True)]}
                        for kind, ref in [('key', key), ('sustain', sustain)]}
            if observed != row['metrics']:
                raise ValueError('Stored metrics do not reproduce')
            difference = compare_recording(key, sustain, before, after)
            if difference != read(run/(name+f'-epoch{number}-comparison.json')):
                raise ValueError('Stored per-reference comparison does not reproduce')
            comparisons[name] = difference
            summary = {}
            for label, diff in difference.items():
                summary[label] = {
                    k: len(diff[k]) for k in ('baseline_onsets_not_guaranteed',
                        'baseline_sustain_not_guaranteed', 'sustain_selected_lost',
                        'sustain_selected_gained', 'tail_regressions', 'incomparable_onset_references')}
                summary[label].update(compared_tails=len(diff['tail_changes']),
                    worst_tail_error_increase_ms=max(
                        (t['error_increase_seconds']*1000 for t in diff['tail_changes']), default=None),
                    before_onset_ambiguous_coverage=len(set(diff['onset_before']['possible']) - set(diff['onset_before']['necessary'])),
                    after_onset_ambiguous_coverage=len(set(diff['onset_after']['possible']) - set(diff['onset_after']['necessary'])))
            diagnostics.append(dict(**row, differences=summary))
        decision = eligible(baseline, epoch['validation'], comparisons)
        if decision != epoch['selection']:
            raise ValueError('Stored gate decision does not reproduce')
        epochs.append(dict(epoch=number, heads_sha256=epoch['heads_sha256'],
            training_losses=epoch['losses'], cells_per_head=epoch['cells_per_head'],
            selection=decision, recordings=diagnostics,
            pooled={kind: {label: {field: pooled(epoch['validation'], kind, label, field)
                                  for field in ('onsets', 'intervals')}
                           for label in ('50ms', '40ms')}
                    for kind in ('key', 'sustain')}))
    if select_epoch(report['epochs']) != report['selected_epoch']:
        raise ValueError('Stored selected epoch does not reproduce')
    preflight = read(run/'preflight.json')
    result = dict(identity=report['identity'], report_sha256=sha256(run/'report.json'),
        preflight_sha256=sha256(run/'preflight.json'), summarizer_sha256=sha256(__file__),
        preflight=dict(windows=sum(r['windows'] for r in preflight['records']),
            all_seven_outputs_bit_identical=all(r['seven_outputs_bit_identical'] for r in preflight['records']),
            all_events_identical=all(r['events_identical'] for r in preflight['records']),
            gradient=preflight['gradient'], cells_per_head_per_epoch=preflight['cells_per_head_per_epoch']),
        epochs=epochs, selected_epoch=report['selected_epoch'], frozen_model_unchanged=report['frozen_model_unchanged'],
        training_performed=True, test_accessed=False, application_enabled=False,
        direct_listening_performed=False, musical_acceptance=False,
        interpretation='Development comparison on repeatedly inspected public recordings; '
                       'not unseen-pretraining evidence, audible-tail labels, or chart LN acceptance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(selected_epoch=result['selected_epoch'], epochs=[dict(
        epoch=e['epoch'], selection=e['selection'], differences=[dict(id=r['id'], **r['differences']['50ms'])
        for r in e['recordings']]) for e in epochs])), flush=True)


if __name__ == '__main__':
    main()
