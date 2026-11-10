"""Replay frozen Tsumugi V1 stitching and audit endpoint provenance, without inference.

Reads existing evidence only. It never modifies predictions or grants LN eligibility.
The upstream oracle is AST-isolated pure code: no Torch/model/checkpoint import.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType

SOURCE_COMMIT = 'eba1c7d05e4d6a53cc4e8916d742d9623a3fdc38'
PURE_FILES = ('instrument_agnostic_amt/amt/inference/types.py',
              'instrument_agnostic_amt/amt/inference/windowed.py',
              'instrument_agnostic_amt/amt/inference/v1_windowed.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=invalid)


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def integer(value):
    return type(value) is int


def load_official(source, identity):
    """Load only the pinned dataclass and non-neural stitching class/functions."""
    source = Path(source)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == SOURCE_COMMIT == identity['source_commit'], 'Unexpected upstream commit')
    changed = subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--name-only'], text=True).strip()
    require(not changed, 'Upstream source has local modifications')
    hashes = {name: sha(source/name) for name in PURE_FILES}
    require(all(identity['source_files'].get(name) == digest for name, digest in hashes.items()),
            'Recorded upstream source hashes differ')
    module = ModuleType('_manialab_tsumugi_audit_pure')
    sys.modules[module.__name__] = module
    env = module.__dict__
    env.update(dataclass=dataclass, replace=replace, defaultdict=defaultdict, MIN_MIDI_PITCH=21, NUM_PITCHES=88)
    for path, names in ((PURE_FILES[0], {'PredictedNote'}),
                        (PURE_FILES[1], {'WindowNoteStitcher', '_merge_instrument_candidates'})):
        tree = ast.parse((source/path).read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        require({n.name for n in nodes} == names, 'Missing official pure definitions')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        selected = ast.fix_missing_locations(ast.Module(body=[future]+nodes, type_ignores=[]))
        exec(compile(selected, str(source/path), 'exec'), env)
    return env['WindowNoteStitcher'], hashes


def note_order(note):
    return (note['start_sample'], note['instrument_id'], note['pitch'], note['slot_index'], note['end_sample'])


def replay_lineage(tracks, *, merge_gap_samples, merge_onset_samples):
    """Mirror the pinned finalize with source IDs; separately compare to its oracle."""
    all_segments, forced = [], []
    for segments in tracks.values():
        if not segments:
            continue
        for index, entry in enumerate(segments):
            item = dict(note=copy.deepcopy(entry['note']), sources=[entry['source']],
                        onset_sources=[entry['source']], endpoint_sources=[entry['source']], merges=[])
            if index == len(segments)-1:
                if not item['note']['has_offset']:
                    forced.append(entry['source'])
                item['note']['has_offset'] = True
            all_segments.append(item)
    all_segments.sort(key=lambda item: (item['note']['instrument_id'], item['note']['pitch'],
        item['note']['slot_index'], item['note']['start_sample'], item['note']['end_sample']))
    merged = []
    for item in all_segments:
        if not merged:
            merged.append(item)
            continue
        current = merged[-1]
        target, incoming = current['note'], item['note']
        same = all(target[k] == incoming[k] for k in ('instrument_id', 'pitch', 'slot_index'))
        gap = same and incoming['start_sample'] <= target['end_sample']+merge_gap_samples and not incoming['has_onset']
        near = same and abs(incoming['start_sample']-target['start_sample']) <= merge_onset_samples
        if not (gap or near):
            merged.append(item)
            continue
        old_end = target['end_sample']
        current['merges'].append(dict(kind='continuation' if gap else 'near_onset',
            existing_sources=current['sources'][:], incoming_sources=item['sources'][:],
            existing_end_sample=old_end, incoming_end_sample=incoming['end_sample'],
            incoming_has_onset=incoming['has_onset'], existing_has_onset=target['has_onset']))
        if incoming['end_sample'] > old_end:
            current['endpoint_sources'] = item['endpoint_sources'][:]
        elif incoming['end_sample'] == old_end:
            current['endpoint_sources'].extend(item['endpoint_sources'])
        if incoming['start_sample'] == target['start_sample']:
            current['onset_sources'].extend(item['onset_sources'])
        current['sources'].extend(item['sources'])
        target['end_sample'] = max(old_end, incoming['end_sample'])
        target['velocity'] = max(target['velocity'], incoming['velocity'])
        target['instrument_candidates'] = list(dict.fromkeys(target['instrument_candidates']+incoming['instrument_candidates']))
        target['has_onset'] = target['has_onset'] or incoming['has_onset']
        target['has_offset'] = incoming['has_offset'] if gap else target['has_offset'] or incoming['has_offset']
    # Match upstream ordering BEFORE V1 restores the predicted instrument ID.
    kept = sorted([item for item in merged if item['note']['has_offset']], key=lambda item: note_order(item['note']))
    dropped = [item for item in merged if not item['note']['has_offset']]
    for item in kept:
        candidates = item['note']['instrument_candidates']
        item['note']['instrument_id'] = candidates[0] if candidates else 0
    return kept, dropped, forced


def source_endpoint(row, identity):
    """Describe construction clipping and boundary-head support, not acoustic truth."""
    args, note = row['arguments'], row['constructed_note']
    require(note is not None, 'Dropped construction cannot provide an endpoint')
    flag = args['boundary_flag']
    fraction = flag[3] if flag is not None else 1.
    raw_end = args['window_start_frame']+round((args['end_frame']+fraction)*row['hop_length'])
    window_end = min(row['total_audio_samples'], args['window_start_frame']+args['valid_audio_frames'])
    end = note['end_sample']
    at_input = end*identity['original_sample_rate'] >= identity['original_samples']*identity['resampled_sample_rate']
    reasons = []
    if flag is None:
        reasons.append('no_boundary_head_fallback')
    elif not flag[1]:
        reasons.append('boundary_head_has_no_offset')
    if raw_end != end:
        reasons.append('constructor_adjusted_endpoint')
    if end >= window_end:
        reasons.append('at_window_observation_boundary')
    if at_input:
        reasons.append('at_original_input_boundary')
    return dict(raw_predicted_end_sample=raw_end, constructed_end_sample=end,
        window_valid_end_sample=window_end, boundary_head_offset=None if flag is None else flag[1],
        endpoint_adjusted=raw_end != end, at_original_input_boundary=at_input,
        unsupported_reasons=reasons, model_predicted_offset_support=not reasons)


def source_onset(row):
    """A continuation's left edge is not a newly predicted attack.

    Keep the model's positive boundary prediction distinct from fallback flags
    and from an OR propagated from a later segment by the upstream stitcher.
    This establishes model support, not that the attack is acoustically correct.
    """
    args, note = row['arguments'], row['constructed_note']
    require(note is not None, 'Dropped construction cannot provide an onset')
    flag = args['boundary_flag']
    fraction = flag[2] if flag is not None else 0.
    raw_start = args['window_start_frame'] + round((args['begin_frame'] + fraction) * row['hop_length'])
    start, window_start = note['start_sample'], args['window_start_frame']
    reasons = []
    if flag is None:
        reasons.append('no_boundary_head_fallback')
    elif not flag[0]:
        reasons.append('boundary_head_has_no_onset')
    if raw_start != start:
        reasons.append('constructor_adjusted_onset')
    return dict(raw_predicted_start_sample=raw_start, constructed_start_sample=start,
        window_start_sample=window_start, begin_frame=args['begin_frame'], at_window_start=start == window_start,
        boundary_head_onset=None if flag is None else flag[0], onset_adjusted=raw_start != start,
        input_start_active_interval=(flag is not None and flag[0] is False and window_start == 0
                                     and args['begin_frame'] == 0 and raw_start == start),
        unsupported_reasons=reasons, model_predicted_onset_support=not reasons)


def classify_onsets(kept, constructions):
    """Only sources at the final start can support a new application keypress."""
    for item in kept:
        evidence = {str(source): source_onset(constructions[source]) for source in item['onset_sources']}
        supported = [int(source) for source, value in evidence.items() if value['model_predicted_onset_support']]
        item.update(onset_evidence=evidence, model_supported_onset_sources=supported,
                    model_supported_final_onset=bool(supported),
                    input_start_active_interval=any(v['input_start_active_interval'] for v in evidence.values()))


def classify_endpoints(kept, constructions, identity, forced):
    """Keep predicted endpoint support separate from finalizer/merge history."""
    forced = set(forced)
    for index, item in enumerate(kept):
        support = {str(source):source_endpoint(constructions[source],identity) for source in item['endpoint_sources']}
        supported = [int(source) for source,evidence in support.items() if evidence['model_predicted_offset_support']]
        extended = any(m['incoming_end_sample'] > m['existing_end_sample'] for m in item['merges'])
        forced_sources = [source for source in item['sources'] if source in forced]
        if len(supported) == len(item['endpoint_sources']):
            category = 'merge_extended_predicted_offset' if extended else 'boundary_head_predicted_offset'
        elif supported:
            category = 'mixed_endpoint_evidence'
        elif any(source in forced for source in item['endpoint_sources']):
            category = 'forced_finalization_endpoint'
        elif any(e['endpoint_adjusted'] for e in support.values()):
            category = 'constructor_adjusted_endpoint'
        elif all(e['boundary_head_offset'] is None for e in support.values()):
            category = 'fallback_interval_endpoint'
        else:
            category = 'unverified_stitched_endpoint'
        item.update(note_index=index,endpoint_evidence=support,model_supported_endpoint_sources=supported,
            classification=category,forced_offset_sources=forced_sources,merge_extended_endpoint=extended,
            unmodified_boundary_head_endpoint=(len(supported) == len(item['endpoint_sources'])
                                               and not extended and not forced_sources),
            application_ln_eligible=False)


def loss_diagnostics(kept, dropped, constructions):
    """Separate within-window predicted retriggers from cross-window duplicates."""
    repeated = [m for group in kept+dropped for m in group['merges'] if m['kind']=='near_onset'
                and m['incoming_has_onset'] and m['existing_has_onset']]
    same_window = []
    for event in repeated:
        pairs = []
        for left_id in event['existing_sources']:
            left = constructions[left_id]
            for right_id in event['incoming_sources']:
                right = constructions[right_id]
                a, b = left['arguments'], right['arguments']
                na, nb = left['constructed_note'], right['constructed_note']
                if (a['boundary_flag'] is not None and b['boundary_flag'] is not None
                        and a['boundary_flag'][0] is True and b['boundary_flag'][0] is True
                        and a['window_start_frame'] == b['window_start_frame']
                        and a['begin_frame'] != b['begin_frame']
                        and na['pitch'] == nb['pitch'] and left['v1_slot_index'] == right['v1_slot_index']):
                    pairs.append([left_id,right_id])
        if pairs:
            same_window.append(dict(**event,distinct_predicted_onset_source_pairs=pairs))
    filtered = [source for group in dropped for source in group['sources']
                if constructions[source]['constructed_note']['has_onset']]
    early_unclosed = [source for source in filtered if not constructions[source]['constructed_note']['has_offset']]
    return dict(merged_repeated_onset_events=repeated,
        same_window_distinct_boundary_onset_merge_events=same_window,
        filtered_onset_source_indices=filtered,filtered_early_unclosed_onset_source_indices=early_unclosed)


def audit_replay(identity, constructions, windows, raw_notes, stats, official_type):
    config, settings = identity['model_config'], identity['settings']
    require(identity.get('evidence_revision') == 2 and config['semi_crf_version'] == 'v1', 'Require traced V1 evidence revision 2')
    sr, hop, slots = config['sample_rate'], config['hop_length'], config['num_pitch_slots']
    total = identity['resampled_samples']
    require(all(integer(v) and v > 0 for v in (sr, hop, slots, total, identity['original_samples'], identity['original_sample_rate'])),
            'Invalid input clock/slot configuration')
    require(sr == identity['resampled_sample_rate'], 'Model and recorded clocks differ')
    gap = hop if settings['merge_gap_ms'] is None else round(settings['merge_gap_ms']*sr/1000.)
    onset = round(settings['merge_onset_ms']*sr/1000.)
    require(gap >= 0 and onset >= 0, 'Invalid merge settings')
    official = official_type(hop_length=hop, total_audio_frames=total, velocity=100,
                             merge_gap_samples=gap, merge_onset_samples=onset)
    require(len(constructions) == stats['decoded_interval_count'], 'Construction count differs from decoder count')
    require(len(windows) == stats['decoded_window_count'], 'Window snapshots incomplete')
    require(stats['window_count'] == stats['decoded_window_count']+stats['skipped_silent_window_count'], 'Window counts differ')
    window_size = round(settings['window_ms']*sr/1000.)
    stride = round(settings['stride_ms']*sr/1000.)
    require(window_size > 0 and stride > 0, 'Invalid window clock')
    starts = [0] if total <= window_size else list(range(0, max(1, total-window_size+1), stride))
    if total > window_size and starts[-1] != total-window_size:
        starts.append(total-window_size)
    require(len(starts) == stats['window_count'], 'Window plan differs')
    observed_starts = [w['window_start_sample'] for w in windows]
    require(observed_starts == sorted(set(observed_starts)) and set(observed_starts) <= set(starts), 'Unexpected/duplicate window snapshot')
    construction_starts = [row['arguments']['window_start_frame'] for row in constructions]
    require(construction_starts == sorted(construction_starts), 'Construction order changed')
    tracks, index = defaultdict(list), 0
    for snapshot in windows:
        start = snapshot['window_start_sample']
        while index < len(constructions) and constructions[index]['arguments']['window_start_frame'] == start:
            row, args = constructions[index], constructions[index]['arguments']
            require(row['hop_length'] == hop and row['total_audio_samples'] == total, 'Construction clock differs')
            valid = min(total-start, window_size)
            require(args['valid_audio_frames'] == valid and args['valid_model_frames'] == math.ceil(valid/hop), 'Construction valid window clock differs')
            for key in ('pair_id','begin_frame','end_frame','window_start_frame','valid_audio_frames','valid_model_frames'):
                require(integer(args[key]), 'Noninteger construction '+key)
            require(0 <= args['pair_id'] < config['num_instrument_classes']*88, 'Invalid instrument/pitch pair')
            require(all(integer(c) and 0 <= c < config['num_instrument_classes'] for c in args['instrument_candidates']),
                    'Invalid instrument candidates')
            require(0 <= args['begin_frame'] <= args['end_frame'] < args['valid_model_frames'], 'Invalid decoded frame interval')
            flag = args['boundary_flag']
            if flag is not None:
                require(isinstance(flag, (list,tuple)) and len(flag) == 4 and all(type(x) is bool for x in flag[:2])
                    and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0 <= x <= 1 for x in flag[2:]), 'Invalid boundary flags/fractions')
            built = official._build_interval_note(**args)
            expected_note = plain(asdict(built)) if built is not None else None
            require(expected_note == row['constructed_note'], 'Recorded construction differs from official builder')
            if built is None:
                require(row['v1_slot_index'] is None, 'Rejected construction has a slot')
            else:
                slot = row['v1_slot_index']
                require(integer(slot) and 0 <= slot < slots, 'Missing/invalid V1 slot')
                built.slot_index, built.instrument_id = slot, 0
                track = (built.pitch-21)*slots+slot
                official.notes_by_pair[track].append(built)
                tracks[track].append(dict(source=index,note=plain(asdict(built))))
            index += 1
        actual_snapshot = {str(track):[plain(asdict(n)) for n in notes] for track, notes in official.notes_by_pair.items()}
        require(snapshot['tracks'] == actual_snapshot, 'Cumulative window snapshot differs from construction replay')
    require(index == len(constructions), 'Constructions missing from window snapshots')
    kept, dropped, forced = replay_lineage(tracks,merge_gap_samples=gap,merge_onset_samples=onset)
    oracle = official.finalize()
    for note in oracle:
        note.instrument_id = note.instrument_candidates[0] if note.instrument_candidates else 0
    oracle = [plain(asdict(note)) for note in oracle]
    require([item['note'] for item in kept] == oracle, 'Lineage replay differs from official finalize')
    require(oracle == raw_notes, 'Official replay differs from saved raw notes')
    require(len(raw_notes) == stats['note_count'], 'Final note count differs')
    classify_endpoints(kept,constructions,identity,forced)
    classify_onsets(kept,constructions)
    losses = loss_diagnostics(kept,dropped,constructions)
    return dict(notes=kept,filtered_groups=dropped,forced_offset_source_indices=forced,
        **losses,
        counts=dict(constructions=len(constructions),constructed=sum(row['constructed_note'] is not None for row in constructions),
            returned_none=sum(row['constructed_note'] is None for row in constructions),final_notes=len(kept),
            filtered_groups=len(dropped),filtered_onset_candidates=len(losses['filtered_onset_source_indices']),
            filtered_early_unclosed_onset_candidates=len(losses['filtered_early_unclosed_onset_source_indices']),
            merged_repeated_onset_events=len(losses['merged_repeated_onset_events']),
            same_window_distinct_boundary_onset_merges=len(losses['same_window_distinct_boundary_onset_merge_events']),
            forced_offset_flags=len(forced),unmodified_boundary_head_endpoints=sum(n['unmodified_boundary_head_endpoint'] for n in kept),
            model_supported_final_onsets=sum(n['model_supported_final_onset'] for n in kept),
            unsupported_final_onsets=sum(not n['model_supported_final_onset'] for n in kept),
            endpoint_classes=dict(Counter(n['classification'] for n in kept))),
        official_finalize_equal=True,saved_raw_notes_equal=True,window_snapshots_equal=True)


def audit(input_dir, source):
    folder = Path(input_dir)
    report = read_json(folder/'report.json')
    identity = read_json(folder/'identity.json')
    require(identity == report['identity'], 'Report/identity mismatch')
    evidence = {}
    for filename,key in (('notes.json','notes_sha256'),('raw-notes.json','raw_notes_sha256'),
                         ('window-notes.json','window_notes_sha256'),('interval-constructions.json','interval_constructions_sha256')):
        evidence[filename] = sha(folder/filename)
        require(evidence[filename] == report.get(key), 'Evidence hash mismatch: '+filename)
    official, source_hashes = load_official(source,identity)
    result = audit_replay(identity,read_json(folder/'interval-constructions.json'),read_json(folder/'window-notes.json'),
                          read_json(folder/'raw-notes.json'),report['stats'],official)
    require(result['counts']['final_notes'] == report['notes'], 'Report note count differs')
    return dict(revision=1,input=str(folder.resolve()),runner_sha256=sha(__file__),
        identity=identity,report_sha256=sha(folder/'report.json'),identity_sha256=sha(folder/'identity.json'),
        evidence_sha256=evidence,verified_upstream_source_sha256=source_hashes,**result,
        enabled_in_application=False,predictions_modified=False,
        limits=['Boundary support means a model prediction at the saved endpoint, not confirmed acoustic silence.',
                'Finalizer, clipping, fallback and merging evidence can coexist; source indices retain that history.',
                'Repeated-onset merge counts include window duplicates and are not proven lost musical attacks.',
                'The same-window subset requires distinct begin frames and two explicit boundary-head onset predictions; it still is not acoustic ground truth.',
                'Early unclosed-onset counts are construction candidates, not deduplicated musical notes.',
                'No neural inference, checkpoint loading, source-audio reading or LN eligibility change occurs.'])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input',type=Path,required=True)
    cli.add_argument('--source',type=Path,default=Path('test-data/other-instrument-model-survey-2026-11-10/tsumugi'))
    cli.add_argument('--output',type=Path,required=True)
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve earlier audit; choose a new output file')
    result = audit(args.input,args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(result['counts']),flush=True)


if __name__ == '__main__':
    main()
