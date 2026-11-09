"""Trainable, masked local-notation graphs with one node per score event.

This module does not identify a melody across arbitrary bar boundaries. A voice
target means the next *new attack* in a local written staff/voice, after the
source event ends. Tied continuations provide provenance, never extra attacks.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


DTYPE = [('pitch', 'i4'), ('onset_div', 'i8'), ('duration_div', 'i8'),
         ('onset_beat', 'f8'), ('duration_beat', 'f8'), ('ts_beats', 'f8'),
         ('staff', 'i4'), ('voice', 'i4')]
SCHEMA = 'piano-score-event-training-v1'
EDGE_TYPES = ('onset', 'consecutive', 'during', 'rest')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_meter_map(xml: bytes, source: dict) -> dict[tuple[str, int], Fraction]:
    """Read explicit, inherited nominal meter without altering source timing.

    Simple/compound/additive signatures are supported; polymeter, unmetered
    music, staff-specific signatures, and mid-measure changes are rejected.
    A pickup keeps its nominal meter, not its shorter observed bar duration.
    """
    tree = ET.fromstring(xml)
    if tree.tag != 'score-partwise':
        raise ValueError('Expected score-partwise meter source')
    result = {}
    for part in tree.findall('part'):
        part_id, current = part.get('id'), None
        for ordinal, measure in enumerate(part.findall('measure'), 1):
            moved = False
            for child in measure:
                if child.tag in ('note', 'backup', 'forward'):
                    moved = True
                if child.tag != 'attributes':
                    continue
                times = child.findall('time')
                if len(times) > 1:
                    raise ValueError('Multiple/staff-specific time signatures are unsupported')
                for time in times:
                    if moved or time.get('number') is not None:
                        raise ValueError('Mid-measure/staff-specific time signature is unsupported')
                    children = list(time)
                    if not children or len(children) % 2:
                        raise ValueError('Unmetered/interchangeable time signature is unsupported')
                    total = Fraction(0)
                    for index in range(0, len(children), 2):
                        beats, beat_type = children[index:index+2]
                        if beats.tag != 'beats' or beat_type.tag != 'beat-type':
                            raise ValueError('Unsupported time-signature construct')
                        groups = (beats.text or '').split('+')
                        if any(not token.strip().isdigit() or int(token) < 1 for token in groups):
                            raise ValueError('Invalid additive beat count')
                        text = (beat_type.text or '').strip()
                        if not text.isdigit() or int(text) < 1:
                            raise ValueError('Invalid beat type')
                        denominator = int(text)
                        if denominator & (denominator - 1):
                            raise ValueError('Non-binary beat type is unsupported')
                        total += Fraction(4 * sum(map(int, groups)), denominator)
                    current = total
            if current is None:
                raise ValueError('Missing explicit initial time signature')
            result[(part_id, ordinal)] = current
    expected = {(m['part'], m['ordinal']) for m in source['measures']}
    if set(result) != expected:
        raise ValueError('Meter/source measure identities differ')
    shared = defaultdict(set)
    for (_, ordinal), value in result.items():
        shared[ordinal].add(value)
    if any(len(values) != 1 for values in shared.values()):
        raise ValueError('Different nominal meters between parts are unsupported')
    return result


def _edges(rows):
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2).T


def _geometry(events, selected):
    """Full-piece temporal relations restricted to selected event IDs.

    The globally nearest future onset is resolved before subsetting, so a
    window never creates a new rest edge or an edge going into the past.
    """
    by_onset = defaultdict(list)
    for index, event in enumerate(events):
        by_onset[event['start_tick']].append(index)
    clocks = sorted(by_onset)
    local = {index: position for position, index in enumerate(selected)}
    rows = []
    for index in selected:
        event, a = events[index], local[index]
        start, end = event['start_tick'], event['end_tick']
        for other in by_onset[start]:
            if other in local and other != index:
                rows.append((a, local[other], 0))
        if end in by_onset:
            for other in by_onset[end]:
                if other in local:
                    rows.append((a, local[other], 1))
        else:
            future = bisect_right(clocks, end)
            if future < len(clocks):
                for other in by_onset[clocks[future]]:
                    if other in local:
                        rows.append((a, local[other], 3))
        for onset in clocks[bisect_right(clocks, start):bisect_left(clocks, end)]:
            for other in by_onset[onset]:
                if other in local:
                    rows.append((a, local[other], 2))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3).T


def build_piece_items(source: dict, meter: dict, *, piece_id: str,
                      max_sounding_measures: int = 16, core_bars: int = 4,
                      halo_bars: int = 1, observed_events: list[dict] | None = None) -> list[dict]:
    """Build deterministic windows and separately masked notation targets.

    All event intervals remain in the original, lossless score clock. The core
    owns targets by their true attack bar; context/carry nodes cannot duplicate
    their node loss. Unknown/excluded notation masks the entire part/bar for
    successor supervision because its omitted neighbors are not established.
    Optional observed events may only extend ends. Input intervals, context
    membership and temporal edges then use those observations, while all
    supervision continues to use the unchanged source notation.
    """
    if not piece_id or min(max_sounding_measures, core_bars) < 1 or halo_bars < 0:
        raise ValueError('Invalid piece identity or window settings')
    unit = source['ticks_per_quarter']
    if not isinstance(unit, int) or isinstance(unit, bool) or unit <= 0:
        raise ValueError('Expected a positive integral score clock')
    written = {row['note']['id']: row['note'] for row in source['written']}
    events = sorted(source['representation']['events'],
                    key=lambda e: (e['start_tick'], e['pitch'], e['part'], e['id']))
    if not events:
        return []
    if len({event['id'] for event in events}) != len(events):
        raise ValueError('Duplicate score-event identity')
    observed = events
    if observed_events is not None:
        by_id = {event['id']: event for event in observed_events}
        if len(by_id) != len(observed_events) or set(by_id) != {event['id'] for event in events}:
            raise ValueError('Observed events must preserve the complete source identity set')
        observed = []
        for event in events:
            value = by_id[event['id']]
            if {key: v for key, v in value.items() if key != 'end_tick'} != {
                    key: v for key, v in event.items() if key != 'end_tick'}:
                raise ValueError('Observed events may change only end_tick')
            if (isinstance(value['end_tick'], bool) or not isinstance(value['end_tick'], Integral)
                    or value['end_tick'] < event['end_tick']):
                raise ValueError('Observed event ends must be integral and cannot shorten the source')
            observed.append(value)
    bounds = {}
    for measure in source['measures']:
        key = measure['ordinal']
        value = (measure['start_tick'], measure['end_tick'])
        if key in bounds and bounds[key] != value:
            raise ValueError('Source parts have inconsistent measure clocks')
        bounds[key] = value
    first, last, eligible = [], [], []
    staff_keys = []
    for note in written.values():
        key = (note['part'], note['staff'])
        if key not in staff_keys:
            staff_keys.append(key)
    # Part order comes from the saved source order; numeric staff order is
    # explicit notation, never inferred from pitch, voice, or training labels.
    part_order = {part: i for i, part in enumerate(dict.fromkeys(n['part'] for n in written.values()))}
    staff_keys.sort(key=lambda key: (part_order[key[0]], key[1]))
    staff_index = {key: index for index, key in enumerate(staff_keys)}
    bad_bars = set()
    for event in events:
        members = [written[identifier] for identifier in event['written_ids']]
        if not members or members[0]['start_tick'] != event['start_tick'] or members[-1]['end_tick'] != event['end_tick']:
            raise ValueError('Event/written geometry differs')
        if any(a['end_tick'] != b['start_tick'] for a, b in zip(members, members[1:])):
            raise ValueError('Discontinuous score event')
        if not 0 <= event['pitch'] <= 119:
            raise ValueError('Pitch outside the frozen encoder octave-feature range')
        first.append(members[0]); last.append(members[-1])
        valid = bool(event['supervision_eligible']) and all(n['voice'] for n in members)
        eligible.append(valid)
        if not valid:
            bad_bars.update((n['part'], n['measure']) for n in members)
    structural_parts = {}
    for note in written.values():
        match = re.fullmatch(r'part\[(\d+)\]/measure\[(\d+)\]/note\[(\d+)\]', note['id'])
        if match:
            structural_parts[int(match[1])] = note['part']
    for identifier, _ in source['excluded']:
        match = re.fullmatch(r'part\[(\d+)\]/measure\[(\d+)\]/note\[(\d+)\]', identifier)
        if not match or int(match[1]) not in structural_parts:
            raise ValueError('Cannot locate excluded note in a known part/bar')
        bad_bars.add((structural_parts[int(match[1])], int(match[2])))
    local_key = lambda n: (n['part'], n['measure'], n['staff'], n['voice'])
    attacks = defaultdict(list)
    for index, note in enumerate(first):
        if eligible[index] and (note['part'], note['measure']) not in bad_bars:
            attacks[local_key(note)].append(index)
    truth = set()
    for index, end_note in enumerate(last):
        if not eligible[index] or (end_note['part'], end_note['measure']) in bad_bars:
            continue
        following = [other for other in attacks[local_key(end_note)]
                     if other != index and events[other]['start_tick'] >= events[index]['end_tick']]
        if following:
            next_start = min(events[other]['start_tick'] for other in following)
            truth.update((index, other) for other in following if events[other]['start_tick'] == next_start)
    sounding = sorted({note['measure'] for note in written.values()})[:max_sounding_measures]
    if not sounding:
        return []
    # Core windows cover intervening rest bars as well, and every selected
    # attack belongs to exactly one core even if some bars contain no attacks.
    allowed = sorted(m for m in bounds if sounding[0] <= m <= sounding[-1])
    all_bars = sorted(bounds)
    bar_position = {bar: index for index, bar in enumerate(all_bars)}
    items = []
    for chunk, at in enumerate(range(0, len(allowed), core_bars)):
        core = allowed[at:at+core_bars]
        left_i = max(0, bar_position[core[0]]-halo_bars)
        right_i = min(len(all_bars), bar_position[core[-1]]+halo_bars+1)
        context = all_bars[left_i:right_i]
        left, right = bounds[context[0]][0], bounds[context[-1]][1]
        selected = [index for index, event in enumerate(observed)
                    if event['start_tick'] < right and event['end_tick'] > left]
        if not selected:
            # A core and its halo can lie entirely inside a long written rest.
            # There are no event predictions or labels to train in that window.
            continue
        local = {index: position for position, index in enumerate(selected)}
        array = np.zeros(len(selected), dtype=DTYPE)
        owner, carry_in, carry_out, staff_y, staff_mask = [], [], [], [], []
        for position, index in enumerate(selected):
            event = observed[index]
            onset, duration = event['start_tick'], event['end_tick']-event['start_tick']
            nominal = meter[(first[index]['part'], first[index]['measure'])]
            array[position] = (event['pitch'], onset, duration, onset/unit, duration/unit,
                               float(nominal), 1, 1)
            is_owner = first[index]['measure'] in core
            owner.append(is_owner)
            carry_in.append(onset < left)
            carry_out.append(event['end_tick'] > right)
            member_staff = {(written[n]['part'], written[n]['staff']) for n in event['written_ids']}
            known_staff = len(staff_keys) == 2 and len(member_staff) == 1 and eligible[index]
            staff_y.append(staff_index[next(iter(member_staff))] if known_staff else 0)
            staff_mask.append(known_staff and is_owner)
        potential, voice_y, voice_mask = [], [], []
        chord = []
        for a, index in enumerate(selected):
            for b, other in enumerate(selected):
                if array['onset_div'][a] == array['onset_div'][b] and a != b:
                    chord.append((a, b))
                if array['onset_div'][a] >= array['onset_div'][b]:
                    continue
                potential.append((a, b))
                origin, target = last[index], first[other]
                same_local_bar = (origin['part'], origin['measure']) == (target['part'], target['measure'])
                masked = (eligible[index] and eligible[other] and owner[b] and same_local_bar
                          and (target['part'], target['measure']) not in bad_bars)
                voice_mask.append(masked)
                voice_y.append((index, other) in truth if masked else False)
        open_targets = [(events[a]['id'], events[b]['id']) for a, b in sorted(truth)
                        if first[b]['measure'] in core and (a not in local or b not in local)]
        payload = dict(notes=array, event_ids=np.asarray([piece_id+':'+events[i]['id'] for i in selected]),
            observed_edges=_geometry(observed, selected), potential_edges=_edges(potential),
            chord_edges=_edges(chord), voice=np.asarray(voice_y, dtype=np.float32),
            voice_mask=np.asarray(voice_mask, dtype=bool),
            chord=np.zeros(len(chord), dtype=np.float32), chord_mask=np.zeros(len(chord), dtype=bool),
            staff=np.asarray(staff_y, dtype=np.int64), staff_mask=np.asarray(staff_mask, dtype=bool),
            core_mask=np.asarray(owner, dtype=bool), carry_in=np.asarray(carry_in, dtype=bool),
            carry_out=np.asarray(carry_out, dtype=bool))
        metadata = dict(schema=SCHEMA, piece_id=piece_id, chunk=chunk, core_measures=core,
            context_measures=context, context_ticks=[left, right], ticks_per_quarter=unit,
            selected_sounding_measures=sounding, score_events=len(selected),
            source_events=len(events), bad_part_measures=[list(key) for key in sorted(bad_bars)],
            open_successor_targets=open_targets, nominal_meter_quarters=[str(meter[(first[i]['part'], first[i]['measure'])]) for i in selected],
            written_ids=[events[i]['written_ids'] for i in selected],
            counts=dict(voice_candidates=len(potential), voice_supervised=int(sum(voice_mask)),
                voice_positives=int(sum(voice_y)), chord_supervised=0, staff_supervised=int(sum(staff_mask)),
                carry_in=int(sum(carry_in)), carry_out=int(sum(carry_out)), open_successors=len(open_targets)))
        if observed_events is not None:
            payload['source_end_tick'] = np.asarray([events[i]['end_tick'] for i in selected], dtype=np.int64)
            payload['added_carry'] = np.asarray([events[i]['end_tick'] <= left for i in selected], dtype=bool)
            metadata['input_interval_kind'] = 'extended_observation_source_targets_unchanged'
            metadata['counts']['extended_events'] = sum(observed[i]['end_tick'] > events[i]['end_tick'] for i in selected)
            metadata['counts']['added_carry'] = int(payload['added_carry'].sum())
        items.append(dict(payload=payload, metadata=metadata))
    return items


def load_event_items(folder: Path, split: str) -> list[dict]:
    if split not in ('train', 'validation'):
        raise ValueError('Only train/validation are supported by this research entry')
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unexpected event-training cache schema')
    result = []
    for row in manifest['items']:
        if row['split'] != split:
            continue
        path = folder/row['cache']
        if file_sha256(path) != row['cache_sha256']:
            raise ValueError('Event-training cache changed')
        with np.load(path, allow_pickle=False) as data:
            payload = {name: data[name].copy() for name in data.files if name != 'metadata_json'}
            metadata = json.loads(str(data['metadata_json']))
        result.append(dict(name=row['name'], split=split, chunk=row['chunk'], payload=payload, metadata=metadata))
    return result


def make_event_graph(item: dict, kind: str = 'clean'):
    """Adapt cached geometry to frozen encoder shapes, with target-only masks."""
    if kind != 'clean':
        raise ValueError('Event-duration augmentation has not been specified')
    import torch
    from piano_svsep.utils import get_vocsep_features, HeteroScoreGraph, score_graph_to_pyg
    p = item['payload']
    array = p['notes'].copy()
    array['staff'] = 1
    array['voice'] = 1
    graph = HeteroScoreGraph(get_vocsep_features(array), p['observed_edges'], note_array=array,
                            name=item.get('name', item['metadata']['piece_id']))
    graph.pot_edges = torch.as_tensor(p['potential_edges']).long()
    graph.pot_chord_edges = torch.as_tensor(p['chord_edges']).long()
    data = score_graph_to_pyg(graph)
    # Preserve exact integer geometry and float64 cache values; the frozen
    # decoder concatenates these real-valued fields with float32 embeddings.
    for name in ('onset_beat', 'duration_beat', 'ts_beats'):
        data['note'][name] = data['note'][name].float()
    del data['note'].staff
    del data['note'].voice
    target = {name: torch.as_tensor(p[name]).clone() for name in
              ('voice', 'chord', 'staff', 'voice_mask', 'chord_mask', 'staff_mask')}
    for label, candidates in (('voice', 'potential_edges'), ('chord', 'chord_edges')):
        mask = p[label+'_mask'] & (p[label] > .5)
        target[label+'_edges'] = {tuple(map(int, edge)) for edge in p[candidates][:, mask].T}
    target['counts'] = item['metadata']['counts']
    return data, target
