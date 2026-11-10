"""Score raw OaF drum events against physical labels and author selections."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.evaluate_adt_drums import (CLASSES, aggregate, score, load_labels,
                                       author_scores, match_times)
from scripts import oaf_taxonomy


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def event_times(rows):
    return [dict(**e, time_ms=e['onset']*1000) for e in rows]


def verify_progress(folder):
    identity = read(folder/'identity.json')
    report = read(folder/'report.json')
    progress = read(folder/'progress.json')
    if (not report['complete_corpus'] or not report['frozen_model_unchanged']
            or progress != report['records'] or len(progress) != 24
            or sha256(folder/'identity.json') != report['identity_sha256']):
        raise ValueError('Incomplete or changed inference')
    if [r['id'] for r in identity['inputs']] != [r['id'] for r in progress]:
        raise ValueError('Input roster changed')
    for row in progress:
        for name, digest in row['artifacts_sha256'].items():
            if sha256(folder/row['id']/name) != digest:
                raise ValueError('Inference artifact changed')
        if sha256(row['audio']) != row['audio_sha256']:
            raise ValueError('Input audio changed')
    for name, digest in report['artifacts_sha256'].items():
        if sha256(folder/name) != digest:
            raise ValueError('Inference proof changed')
    return identity, progress


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--inference', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    source, output = args.inference.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use new research output')
    identity, progress = verify_progress(source)
    manifest = ROOT/'test-data/physical-drums/dataset-v2/manifest.json'
    if sha256(manifest) != identity['manifest_sha256']:
        raise ValueError('Changed physical-label manifest')
    rows = {r['id']: r for r in read(manifest)['records'] if r['dataset'] == 'MDB'}
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output/'scoring-snapshot.py')
    shutil.copyfile(ROOT/'scripts/evaluate_adt_drums.py', output/'matching-snapshot.py')
    shutil.copyfile(ROOT/'scripts/oaf_taxonomy.py', output/'taxonomy-snapshot.py')
    results = []
    for row in progress:
        predicted = event_times(read(source/row['id']/'events.json'))
        if row['dataset'] == 'MDB':
            truth = load_labels(rows[row['id']])
            if rows[row['id']]['audio_sha256'] != row['audio_sha256']:
                raise ValueError('Physical-label audio mismatch')
            annotation = ROOT/rows[row['id']]['annotation']
            subclass = annotation.parent.parent/'subclass'/annotation.name.replace('_class.txt','_subclass.txt')
            fine = oaf_taxonomy.load_mdb_subclasses(annotation, subclass)
            mapped = oaf_taxonomy.prediction_events(read(source/row['id']/'events.json'))
            result = dict(id=row['id'], previous_split=row['previous_split'],
                annotation_sha256=rows[row['id']]['annotation_sha256'],
                scores={str(t): score(truth, predicted, t) for t in (20, 40, 50)},
                coarse_score_notice='Provisional six-class comparison has known taxonomy mismatches; use eight_hit_scores for compatible labels.',
                subclass_source={k:v for k,v in fine.items() if k!='events'},
                eight_hit_scores={str(t): oaf_taxonomy.score(fine['events'],mapped,t) for t in (20,40,50)},
                inference_summary=row)
            results.append(result)
        else:
            from manialab.domain import parse_osu
            from scripts.research_source_attack_classes import drum_event_counts
            chart = ROOT/'test-data/sakura-benchmark/reference.osu'
            if sha256(chart) != read(chart.parent/'manifest.json')['charts']['reference']['chart_sha256']:
                raise ValueError('Author chart changed')
            truth = [dict(time_ms=t, kind=k) for t, k in drum_event_counts(
                parse_osu(chart.read_text(encoding='utf-8-sig'))['notes'])]
            lane4 = [e['time_ms'] for e in truth if e['kind'] == 'hi-hat']
            result = dict(id='sakura', reference_sha256=sha256(chart),
                scores=author_scores(truth, predicted, row['duration_seconds']*1000),
                lane4_any_class={label: match_times(lane4, [e['time_ms']-shift for e in predicted],40)
                                for label, shift in [('raw',0),('chart_lead_22_5ms',22.5)]},
                inference_summary=row)
        save(output/(row['id']+'.json'), result)
    summaries = {split: {str(t): {kind: aggregate([r['scores'][str(t)][kind] for r in results
                   if split == 'all_development' or r['previous_split'] == split])
                   for kind in (*CLASSES, 'unknown')} for t in (20,40,50)}
                   for split in ('all_development','train','validation','test')}
    compatible = {split: {str(t): {kind: aggregate([r['eight_hit_scores'][str(t)][kind]
        for r in results if split == 'all_development' or r['previous_split'] == split])
        for kind in (*oaf_taxonomy.CLASSES,'unknown')} for t in (20,40,50)}
        for split in ('all_development','train','validation','test')}
    # Existing physical classifier and OaF on exactly the same recordings.
    baseline = ROOT/'test-data/physical-drums/evaluation-solo-negatives-v1'
    old = read(baseline/'report.json')
    if sha256(baseline/'events.json') != old['events_sha256']:
        raise ValueError('Baseline events changed')
    old_manifest = ROOT/'test-data/physical-drums/dataset-solo-negatives-v1/manifest.json'
    if sha256(old_manifest) != old['dataset_sha256']:
        raise ValueError('Baseline label manifest changed')
    old_rows = {r['id']: r for r in read(old_manifest)['records']}
    old_events = read(baseline/'events.json')
    common = []
    for result in results:
        rid = result['id']
        if rid not in old_events:
            continue
        if any(old_rows[rid][k] != rows[rid][k] for k in ('audio_sha256','annotation_sha256','split')):
            raise ValueError('Baseline source mismatch')
        measured = score(load_labels(rows[rid]), old_events[rid], 40)
        prior = next(r for r in old['records'] if r['id'] == rid)
        if any(measured[k]['matched'] != prior['scores']['40'][k]['matched'] for k in ('kick','snare','hi-hat')):
            raise ValueError('Baseline matching not reproduced')
        common.append(dict(id=rid, previous_split=result['previous_split'],
                           baseline=measured, oaf=result['scores']['40']))
    comparison = {split: dict(ids=[r['id'] for r in common if r['previous_split'] == split],
        scores={variant: {kind: aggregate([r[variant][kind] for r in common if r['previous_split'] == split])
                          for kind in ('kick','snare','hi-hat')}
                for variant in ('baseline','oaf')}) for split in ('validation','test')}
    report = dict(complete_corpus=True, mdb_recordings=len(results), summaries=summaries,
        eight_hit_summaries=compatible,
        taxonomy_notice='Post-result fixed subclass compatibility audit: SST joins snare, TMB joins HH, RDB joins bell. China/splash references remain unsupported, not matched or silently dropped. No probability/threshold/model changes.',
        baseline_comparison_40ms=comparison, inference_identity_sha256=sha256(source/'identity.json'),
        inference_report_sha256=sha256(source/'report.json'), source_inference=str(source),
        baseline_report_sha256=sha256(baseline/'report.json'), script_sha256=sha256(__file__),
        direct_listening_performed=False, application_enabled=False,
        notice='All results are development regressions. Author lane4 is a selection, not exhaustive physical hi-hat truth. Coarse 8-hit classes merge several timbres. No LN duration measured.',
        artifacts_sha256={p.name: sha256(p) for p in output.iterdir()})
    save(output/'report.json', report)
    print(json.dumps(compatible['all_development']['40']), flush=True)


if __name__ == '__main__':
    main()
