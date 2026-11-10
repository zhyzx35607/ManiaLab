"""Unlabelled diagnostic windows over frozen, quantized acoustic event IDs.

These coordinates are an artificial musical clock, not corrected audio timing.
No MusicXML identity, source voice, score duration or supervision is invented.
Items are accepted by piano_event_training.make_event_graph unchanged.
"""
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from scripts.piano_event_training import DTYPE, _geometry

FROZEN_SAKURA_SHA256 = '34708342e8265446c2650c365834f10f5a1739ce311cad1ce1df13908076b36e'
FROZEN_SAKURA_EVENTS = 1557
TICKS_PER_QUARTER = 480
QUARTERS_PER_BAR = 4
BAR_TICKS = TICKS_PER_QUARTER * QUARTERS_PER_BAR
CORE_BARS, HALO_BARS = 4, 1
SCHEMA = 'piano-audio-event-graph-v1'
ROW_FIELDS = {'id', 'pitch', 'tick', 'duration'}


def _validated_rows(rows):
    if not isinstance(rows, list):
        raise ValueError('Expected a list of unlabelled diagnostic audio rows')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise ValueError('Audio rows require only id/pitch/tick/duration; no notation labels')
        if not isinstance(row['id'], str) or not row['id'].strip() or row['id'] in seen:
            raise ValueError('Audio event IDs must be nonempty and unique')
        seen.add(row['id'])
        if any(isinstance(row[key], bool) or not isinstance(row[key], Integral)
               for key in ('pitch', 'tick', 'duration')):
            raise ValueError('Audio diagnostic geometry must use exact integers')
        if not 0 <= row['pitch'] <= 119 or row['tick'] < 0 or row['duration'] <= 0:
            raise ValueError('Invalid audio diagnostic geometry')
        if int(row['tick']) + int(row['duration']) > np.iinfo(np.int64).max:
            raise ValueError('Audio diagnostic geometry exceeds the int64 clock')
    return sorted(enumerate(rows), key=lambda pair: (pair[1]['tick'], pair[1]['pitch'], pair[1]['id']))


def load_frozen_sakura_rows(path: Path) -> list[dict]:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != FROZEN_SAKURA_SHA256:
        raise ValueError('Frozen Sakura graph rows changed')
    rows = json.loads(data)
    _validated_rows(rows)
    if len(rows) != FROZEN_SAKURA_EVENTS:
        raise ValueError('Frozen Sakura event count changed')
    return rows


def build_audio_event_items(rows: list[dict], *, piece_id: str) -> list[dict]:
    """Make fixed 4-bar cores with 1-bar context and immutable full intervals.

    All bar indices are zero based, anchored at diagnostic tick zero. Cores
    cover the observed interval extent, including empty/intervening bars.
    Context contains every interval overlapping its half-open tick range.
    Output ownership is not supervision: every target/loss mask stays false.
    """
    if not isinstance(piece_id, str) or not piece_id.strip():
        raise ValueError('Expected a nonempty diagnostic piece identity')
    indexed = _validated_rows(rows)
    if not indexed:
        return []
    ordered = [row for _, row in indexed]
    count = len(ordered)
    starts = np.asarray([row['tick'] for row in ordered], dtype=np.int64)
    durations = np.asarray([row['duration'] for row in ordered], dtype=np.int64)
    ends = starts + durations
    attack_bars = starts // BAR_TICKS
    full_events = [dict(start_tick=int(start), end_tick=int(end)) for start, end in zip(starts, ends)]
    # Resolve rest successors in the complete input exactly once. Window
    # induction cannot invent a replacement target or a backwards rest edge.
    full_edges = _geometry(full_events, list(range(count)))
    total_bars = (int(ends.max()) + BAR_TICKS - 1) // BAR_TICKS
    items = []
    for chunk, first_bar in enumerate(range(0, total_bars, CORE_BARS)):
        last_bar = min(total_bars, first_bar + CORE_BARS)
        context_first = max(0, first_bar-HALO_BARS)
        context_last = min(total_bars, last_bar+HALO_BARS)
        core_left, core_right = first_bar*BAR_TICKS, last_bar*BAR_TICKS
        left, right = context_first*BAR_TICKS, context_last*BAR_TICKS
        selected = np.flatnonzero((starts < right) & (ends > left))
        if not len(selected):
            continue
        local = np.full(count, -1, dtype=np.int64)
        local[selected] = np.arange(len(selected))
        keep = (local[full_edges[0]] >= 0) & (local[full_edges[1]] >= 0)
        observed = full_edges[:, keep].copy()
        observed[:2] = local[observed[:2]]
        array = np.zeros(len(selected), dtype=DTYPE)
        array['pitch'] = [ordered[index]['pitch'] for index in selected]
        array['onset_div'], array['duration_div'] = starts[selected], durations[selected]
        array['onset_beat'] = starts[selected]/TICKS_PER_QUARTER
        array['duration_beat'] = durations[selected]/TICKS_PER_QUARTER
        array['ts_beats'] = QUARTERS_PER_BAR
        array['staff'], array['voice'] = 1, 1
        onsets = starts[selected]
        potential = np.vstack(np.where(onsets[:, None] < onsets[None, :])).astype(np.int64)
        chords = np.vstack(np.where((onsets[:, None] == onsets[None, :])
                                    & ~np.eye(len(selected), dtype=bool))).astype(np.int64)
        owner = (onsets >= core_left) & (onsets < core_right)
        bars = attack_bars[selected]
        voice_owner = owner[potential[1]]
        chord_owner = owner[chords[1]]
        same_bar = bars[potential[0]] == bars[potential[1]]
        carry_in, carry_out = onsets < left, ends[selected] > right
        payload = dict(notes=array,
            event_ids=np.asarray([ordered[index]['id'] for index in selected]),
            source_row_indices=np.asarray([indexed[index][0] for index in selected], dtype=np.int64),
            attack_bar=bars.copy(), observed_edges=observed, potential_edges=potential, chord_edges=chords,
            voice=np.zeros(potential.shape[1], dtype=np.float32), voice_mask=np.zeros(potential.shape[1], dtype=bool),
            chord=np.zeros(chords.shape[1], dtype=np.float32), chord_mask=np.zeros(chords.shape[1], dtype=bool),
            staff=np.zeros(len(selected), dtype=np.int64), staff_mask=np.zeros(len(selected), dtype=bool),
            node_output_owner=owner, voice_output_owner=voice_owner, chord_output_owner=chord_owner,
            voice_same_bar=same_bar, carry_in=carry_in, carry_out=carry_out)
        metadata = dict(schema=SCHEMA, piece_id=piece_id, chunk=chunk,
            coordinate_kind='frozen_manual_quantization_diagnostic', bar_index_base=0,
            ticks_per_quarter=TICKS_PER_QUARTER, quarters_per_bar=QUARTERS_PER_BAR,
            core_bars=list(range(first_bar, last_bar)), context_bars=list(range(context_first, context_last)),
            core_ticks=[core_left, core_right], context_ticks=[left, right], total_bars=total_bars,
            source_events=count, targets_available=False, original_audio_fields_changed=False,
            counts=dict(nodes=len(selected), owned_nodes=int(owner.sum()),
                voice_candidates=potential.shape[1], chord_candidates=chords.shape[1],
                owned_voice=int(voice_owner.sum()), owned_chord=int(chord_owner.sum()),
                owned_same_bar_voice=int((voice_owner & same_bar).sum()),
                voice_supervised=0, voice_positives=0, chord_supervised=0, staff_supervised=0,
                carry_in=int(carry_in.sum()), carry_out=int(carry_out.sum())))
        items.append(dict(name=piece_id, chunk=chunk, metadata=metadata, payload=payload))
    return items
