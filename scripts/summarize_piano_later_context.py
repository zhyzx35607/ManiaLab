"""Audit proved winning competitors in positive-owner and output-owner contexts.

Uses saved validation logits only. This is a pairwise ranking diagnostic, not
a new decoder, counterfactual recovery estimate, or music-quality measurement.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_provenance import sha256
from scripts.piano_successor_decode import select_successor_groups
from scripts.prepare_piano_later_negatives import dataset_folders, MANIFESTS
from scripts.summarize_piano_successor_ranking import load, read, require, verify_artifacts, sum_fields


def compare_contexts(positive, lost, local_maxima, output_maxima):
    """Compare the *same* global winning candidates against the positive group.

    A local tie is not a strict outrank; no claim is made about other candidates
    or whether removing this winning group would recover the positive group.
    """
    positive = np.asarray(positive, dtype=float)
    lost = np.asarray(lost)
    local_maxima = np.asarray(local_maxima, dtype=float)
    output_maxima = np.asarray(output_maxima, dtype=float)
    require(positive.ndim == local_maxima.ndim == output_maxima.ndim == 1 and
            len(positive) > 0 and len(local_maxima) > 0 and
            local_maxima.shape == output_maxima.shape, 'Invalid context arrays')
    require(lost.dtype == bool and lost.shape == positive.shape and lost.any(), 'Invalid lost-positive mask')
    require(all(np.isfinite(v).all() for v in (positive, local_maxima, output_maxima)), 'Nonfinite context scores')
    require((positive[lost] >= 0).all() and np.all(output_maxima == output_maxima.max()) and
            output_maxima.max() > positive.max(), 'Expected a strictly later winning maximum')
    local = float(local_maxima.max()); output = float(output_maxima.max())
    return dict(local_competitor_max=local, output_competitor_max=output,
        positive_max=float(positive.max()), positive_min=float(positive.min()),
        local_outranks_positive_group=bool(local > positive.max()),
        local_ranked_above_lost_members=int((local > positive[lost]).sum()),
        local_ranked_at_or_below_lost_members=int((local <= positive[lost]).sum()),
        output_minus_local_max=output-local, lost_true_edges=int(lost.sum()))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('before-run', 'after-run', 'comparison', 'competition', 'sidecars', 'output'):
        cli.add_argument('--'+name, type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists() and args.output.resolve().is_relative_to((ROOT/'test-data').resolve()),
            'Use a new local research directory')
    paired = read(args.comparison/'report.json'); competition = read(args.competition/'report.json')
    proof = read(args.sidecars/'report.json')
    for folder, report in ((args.comparison, paired), (args.competition, competition), (args.sidecars, proof)):
        verify_artifacts(folder, report)
    require(competition['identity']['comparison_sha256'] == sha256(args.comparison/'report.json') and
            competition['identity']['proof_report_sha256'] == sha256(args.sidecars/'report.json'), 'Different diagnostic inputs')
    runs = {}
    for side in ('before', 'after'):
        folder = getattr(args, side+'_run'); report = read(folder/'report.json')
        verify_artifacts(folder, report)
        require(paired['identity'][side+'_run'] == sha256(folder/'report.json') and
                paired['selected_weights'][side] == report['selected_sha256'], 'Different selected run')
        runs[side] = report
    require(runs['after']['identity']['proof_report_sha256'] == sha256(args.sidecars/'report.json'), 'Different fitted proof')
    folders = dataset_folders(); manifests = {}
    for kind, folder in folders.items():
        require(sha256(folder/'manifest.json') == MANIFESTS[kind] == proof['identity']['dataset_sha256'][kind], 'Changed dataset')
        manifests[kind] = {(r['name'], r['chunk']):r for r in read(folder/'manifest.json')['items'] if r['split']=='validation'}
    order = list(manifests['clean'])
    annotations = {(r['input'], r['name'], r['chunk']):r for r in proof['items'] if r['split']=='validation'}
    contexts = {side:{} for side in runs}; output_owners = {}
    for kind in folders:
        for index, (name, chunk) in enumerate(order):
            annotation = annotations[kind, name, chunk]; row = manifests[kind][name, chunk]
            path = folders[kind]/row['cache']
            require(sha256(path) == row['cache_sha256'] == annotation['cache_sha256'], 'Changed cache identity')
            graph = load(path); sidecar = load(args.sidecars/annotation['file'])
            a, b = graph['potential_edges']; ids = graph['event_ids']
            require(np.array_equal(sidecar['event_ids'], ids) and
                    np.array_equal(sidecar['potential_edges'], graph['potential_edges']), 'Changed annotation identity')
            for i in np.flatnonzero(graph['core_mask'][b]):
                key = kind, name, str(ids[a[i]]), str(ids[b[i]])
                require(key not in output_owners, 'Duplicate output owner')
                output_owners[key] = chunk
            sources = defaultdict(list)
            for i in np.flatnonzero(sidecar['derived_later_mask']):
                sources[str(ids[a[i]])].append(int(i))
            for side, run in runs.items():
                folder = getattr(args, side+'_run')
                saved_graph = load(folder/f'{kind}-graph-{index:03d}.npz')
                require(graph.keys() == saved_graph.keys() and all(np.array_equal(v, saved_graph[k]) for k,v in graph.items()), 'Changed saved graph')
                logits = load(folder/f'epoch-{run["selected_epoch"]}-{kind}-{index:03d}.npz')['voice_logits']
                for source, negatives in sources.items():
                    pos = np.flatnonzero((ids[a] == source) & graph['voice_mask'] & (graph['voice'] > .5))
                    key = kind, name, source
                    require(key not in contexts[side] and len(pos)>0, 'Invalid positive-owner context')
                    contexts[side][key] = dict(chunk=chunk,
                        positives={str(ids[b[i]]):float(logits[i]) for i in pos},
                        negatives={str(ids[b[i]]):float(logits[i]) for i in negatives})
    variants = []; details = []
    for variant in paired['variants']:
        kind = variant['input']; cases = []
        comp = next(v for v in competition['variants'] if v['input']==kind)
        comp_rows = {r['name']:r for r in comp['cases']}
        for row in variant['cases']:
            name = row['name']; data = load(args.comparison/row['paired_file'])
            masks = load(args.competition/comp_rows[name]['proof_file'])
            require(all(np.array_equal(masks[k], data[k]) for k in ('source_ids','target_ids')), 'Changed proof ordering')
            source_indices = defaultdict(list)
            for i, source in enumerate(data['source_ids']): source_indices[str(source)].append(i)
            results = {}
            for side in runs:
                counts = Counter({k:0 for k in ('sources','lost_true_edges','missing_local_sources','missing_local_true_edges',
                    'local_outranking_sources','local_outranking_true_edges','context_inversion_sources','context_inversion_true_edges',
                    'same_output_window_sources','different_output_window_sources','local_ranked_above_lost_members','local_ranked_at_or_below_lost_members')})
                logits = data[side+'_logits']; selected = data[side+'_selected']
                require(np.array_equal(selected, select_successor_groups(data['source_ids'],data['target_ticks'],logits)), 'Changed decoder')
                for source, indices in source_indices.items():
                    indices = np.asarray(indices)
                    pos = indices[data['actual'][indices] & data['score_mask'][indices]]
                    lost = (logits[pos]>=0) & ~selected[pos]
                    if not lost.any(): continue
                    winning = indices[selected[indices]]
                    maxima = winning[logits[winning] == logits[winning].max()]
                    if not masks['proved'][maxima].all(): continue
                    counts['sources'] += 1; counts['lost_true_edges'] += int(lost.sum())
                    context = contexts[side].get((kind,name,source))
                    if context is None or any(str(data['target_ids'][i]) not in context['negatives'] for i in maxima):
                        counts['missing_local_sources'] += 1; counts['missing_local_true_edges'] += int(lost.sum()); continue
                    require(set(context['positives']) == set(map(str,data['target_ids'][pos])), 'Changed complete positive group')
                    require(all(context['positives'][str(data['target_ids'][i])] == logits[i] for i in pos), 'Positive/output context mismatch')
                    local = [context['negatives'][str(data['target_ids'][i])] for i in maxima]
                    result = compare_contexts(logits[pos],lost,local,logits[maxima])
                    label = 'local_outranking' if result['local_outranks_positive_group'] else 'context_inversion'
                    counts[label+'_sources'] += 1; counts[label+'_true_edges'] += int(lost.sum())
                    chunks = [output_owners[kind,name,source,str(data['target_ids'][i])] for i in maxima]
                    counts[('same' if all(c==context['chunk'] for c in chunks) else 'different')+'_output_window_sources'] += 1
                    for k in ('local_ranked_above_lost_members','local_ranked_at_or_below_lost_members'): counts[k] += result[k]
                    details.append(dict(input=kind,name=name,model=side,source=source,positive_owner=context['chunk'],
                        competitor_output_owners=chunks,positive_targets=list(map(str,data['target_ids'][pos])),
                        maximum_targets=list(map(str,data['target_ids'][maxima])),**result))
                require(counts['lost_true_edges'] == comp_rows[name][side]['removed_true_with_proved_maximum_only'], 'Changed diagnostic population')
                results[side] = dict(counts)
            cases.append(dict(name=name,**results))
        variants.append(dict(input=kind,cases=cases,aggregate={s:sum_fields([r[s] for r in cases]) for s in runs}))
    args.output.mkdir(parents=True); shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    (args.output/'details.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    report = dict(identity={k:sha256(getattr(args,k)/'report.json') for k in ('before_run','after_run','comparison','competition','sidecars')},
        runner_sha256=sha256(__file__),variants=variants,model_loaded=False,original_scoring_changed=False,
        test_sources_read=False,musical_acceptance=False,application_enabled=False,
        scope='Post-result validation diagnostic of the same winning candidates in two saved contexts; no counterfactual recovery or whole-local-decoder claim.',
        artifacts_sha256={p.name:sha256(p) for p in args.output.iterdir()})
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([dict(input=v['input'],**v['aggregate']) for v in variants]),flush=True)


if __name__ == '__main__': main()
