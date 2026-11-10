"""Read a conservative MusicXML subset without trusting imported tie pointers.

This is a score-supervision adapter, not a performance transcription. Source
paths identify written notes; explicitly paired sound ties identify sustains.
Ambiguous or incomplete declarations never become attack/release labels.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.piano_score_events import WrittenNote, ScoreRepresentation, score_events


class UnsupportedScore(ValueError):
    """An unsupported construct prevents trustworthy source-time recovery."""


@dataclass(frozen=True)
class SourceNote:
    note: WrittenNote
    xml_id: str | None
    measure_label: str
    sound_ties: tuple[str, ...]
    notation_ties: tuple[tuple[str, str | None], ...]
    playback_conditions: tuple[tuple[str, str], ...]
    problems: tuple[str, ...]


@dataclass(frozen=True)
class SourceMeasure:
    part: str
    ordinal: int
    label: str
    start_tick: int
    end_tick: int


@dataclass(frozen=True)
class SourceScore:
    ticks_per_quarter: int
    written: tuple[SourceNote, ...]
    measures: tuple[SourceMeasure, ...]
    ties: tuple[tuple[str, str], ...]
    verified_note_ids: tuple[str, ...]
    issues: tuple[tuple[str, tuple[str, ...]], ...]
    excluded: tuple[tuple[str, str], ...]
    representation: ScoreRepresentation


def _positive(text: str | None, context: str) -> Fraction:
    try:
        value = Fraction(text)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise UnsupportedScore(f'{context}: invalid numeric value {text!r}') from exc
    if value <= 0:
        raise UnsupportedScore(f'{context}: expected positive duration/divisions')
    return value


def _pitch(element: ET.Element) -> int:
    if element.find('unpitched') is not None:
        raise UnsupportedScore('Unpitched notation is outside the piano adapter')
    try:
        step = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[
            element.findtext('pitch/step')]
        octave = int(element.findtext('pitch/octave'))
        alter = Fraction(element.findtext('pitch/alter', '0'))
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        raise UnsupportedScore('Invalid written pitch') from exc
    pitch = (octave + 1) * 12 + step + alter
    if pitch.denominator != 1 or not 0 <= pitch <= 127:
        raise UnsupportedScore('Only integral MIDI pitches in 0..127 are supported')
    return int(pitch)


def _declarations(element: ET.Element):
    sound = tuple(tie.get('type', '') for tie in element.findall('tie'))
    notation = tuple((tie.get('type', ''), tie.get('number'))
                     for tie in element.findall('notations/tied'))
    problems = []
    if len(set(sound)) != len(sound) or not set(sound) <= {'start', 'stop'}:
        problems.append('invalid_sound_tie_declaration')
    if any(tie.get('time-only') is not None for tie in element.findall('tie')):
        problems.append('conditional_tie_requires_repeat_expansion')
    if element.get('time-only') is not None:
        problems.append('conditional_note_requires_repeat_expansion')
    kinds = [kind for kind, _ in notation]
    if len(set(kinds)) != len(kinds) or not set(kinds) <= {'start', 'stop'}:
        problems.append('unsupported_notation_tie_declaration')
    if any(number is not None and (not number.isdecimal() or int(number) < 1)
           for _, number in notation):
        problems.append('invalid_notation_tie_number')
    # Sound ties are authoritative. Visual ties may be absent, but if present
    # their endpoints must agree before this adapter permits supervision.
    if notation and set(kinds) != set(sound):
        problems.append('sound_notation_tie_disagreement')
    if element.get('attack') is not None or element.get('release') is not None:
        problems.append('performance_offset_not_applied')
    conditions = [(key, element.get(key)) for key in ('time-only', 'attack', 'release')
                  if element.get(key) is not None]
    conditions += [(f'tie[{i}].time-only', tie.get('time-only'))
                   for i, tie in enumerate(element.findall('tie'), 1) if tie.get('time-only') is not None]
    return sound, notation, tuple(conditions), tuple(problems)


def _resolve(written: tuple[SourceNote, ...]):
    by_id = {row.note.id: row for row in written}
    starts, stops = defaultdict(list), defaultdict(list)
    all_starts, all_ends = defaultdict(list), defaultdict(list)
    problems = {key: set(row.problems) for key, row in by_id.items()}
    for row in written:
        note = row.note
        all_starts[(note.part, note.pitch, note.start_tick)].append(note.id)
        all_ends[(note.part, note.pitch, note.end_tick)].append(note.id)
        if 'start' in row.sound_ties:
            starts[(note.part, note.pitch, note.end_tick)].append(note.id)
        if 'stop' in row.sound_ties:
            stops[(note.part, note.pitch, note.start_tick)].append(note.id)
    outgoing, incoming = defaultdict(set), defaultdict(set)

    def number(identifier, kind):
        return next((value for tag, value in by_id[identifier].notation_ties if tag == kind), None)

    for key, sources in starts.items():
        for source in sources:
            for target in stops.get(key, ()):
                a, b = number(source, 'start'), number(target, 'stop')
                # An explicit number may disambiguate two simultaneous unisons.
                # Missing/default numbers must not manufacture such evidence.
                if a is not None and b is not None and a != b:
                    continue
                outgoing[source].add(target)
                incoming[target].add(source)
    edges = []
    for identifier, row in by_id.items():
        for kind, neighbors in (('start', outgoing), ('stop', incoming)):
            if kind in row.sound_ties and len(neighbors[identifier]) != 1:
                problems[identifier].add('unmatched_' + kind if not neighbors[identifier]
                                         else 'ambiguous_' + kind)
    for source, targets in outgoing.items():
        if len(targets) == 1:
            target = next(iter(targets))
            if len(incoming[target]) == 1:
                edges.append((source, target))
            else:
                problems[source].add('ambiguous_partner')
        else:
            for target in targets:
                problems[target].add('ambiguous_partner')
    # Unknown declarations at either end mask the whole candidate component,
    # including a locally unambiguous continuation of an ambiguous chain.
    adjacency = defaultdict(set)
    for source, targets in outgoing.items():
        for target in targets:
            adjacency[source].add(target)
            adjacency[target].add(source)
    # A missing declaration does not justify a negative label at its potential
    # partner. An unmatched start facing an unmarked same-pitch note could be
    # a dropped stop, rather than a verified reattack (and vice versa).
    for identifier, row in by_id.items():
        if not problems[identifier]:
            continue
        kinds = set(row.sound_ties) | {kind for kind, _ in row.notation_ties}
        note = row.note
        possible = []
        if 'start' in kinds:
            possible += all_starts[(note.part, note.pitch, note.end_tick)]
        if 'stop' in kinds:
            possible += all_ends[(note.part, note.pitch, note.start_tick)]
        for partner in possible:
            adjacency[identifier].add(partner)
            adjacency[partner].add(identifier)
    pending = [key for key, value in problems.items() if value]
    seen = set(pending)
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                problems[neighbor].add('connected_to_unverified_tie')
                pending.append(neighbor)
    # Never expose speculative edges as resolved ties, even if their events
    # would be masked by score_events downstream.
    edges = tuple(sorted((a, b) for a, b in edges if a not in seen and b not in seen))
    verified = tuple(sorted(set(by_id) - seen))
    issues = tuple((key, tuple(sorted(value))) for key, value in sorted(problems.items()) if value)
    return edges, verified, issues


def read_musicxml(path: str | Path) -> SourceScore:
    return parse_musicxml(Path(path).read_bytes())


def parse_musicxml(data: str | bytes) -> SourceScore:
    """Read written score time; repeats are not expanded into performance time.

    Supports partwise pitched notes, chords, rests, backup/forward, tuplets via
    explicit duration, pickups, and divisions changes at measure boundaries.
    Grace/cue notes are recorded as excluded, never labeled as missing attacks.
    Unsupported clocks or tied excluded notes reject the file, not silently
    guessed timing. IDs are structural source paths, stable for identical XML.
    """
    root = ET.fromstring(data)
    if root.tag != 'score-partwise':
        raise UnsupportedScore('Expected unnamespaced score-partwise MusicXML')
    raw, measures, excluded, part_ids = [], [], [], set()
    for part_index, part in enumerate(root.findall('part'), 1):
        part_id = part.get('id')
        if not part_id or part_id in part_ids:
            raise UnsupportedScore('Expected unique explicit part IDs')
        part_ids.add(part_id)
        position, divisions = Fraction(0), None
        for ordinal, measure in enumerate(part.findall('measure'), 1):
            if measure.get('non-controlling') == 'yes':
                raise UnsupportedScore('Non-controlling measures require separate clock handling')
            start, cursor, high = position, position, position
            anchor = None
            note_index = 0
            label = measure.get('number', '')
            for element in measure:
                tag = element.tag
                if tag == 'attributes':
                    if element.find('transpose') is not None:
                        raise UnsupportedScore('Transposing parts are outside the piano adapter')
                    new = element.findtext('divisions')
                    if new is not None:
                        if high != start or cursor != start:
                            raise UnsupportedScore('Mid-measure divisions changes are unsupported')
                        divisions = _positive(new, 'divisions')
                elif tag in ('backup', 'forward'):
                    if divisions is None:
                        raise UnsupportedScore('Missing divisions before score movement')
                    delta = _positive(element.findtext('duration'), tag) / divisions
                    cursor += delta if tag == 'forward' else -delta
                    if cursor < start:
                        raise UnsupportedScore('Backup escapes its measure')
                    high = max(high, cursor)
                    anchor = None
                elif tag == 'note':
                    note_index += 1
                    identifier = f'part[{part_index}]/measure[{ordinal}]/note[{note_index}]'
                    if divisions is None:
                        raise UnsupportedScore('Missing divisions before note')
                    sound, notation, conditions, problems = _declarations(element)
                    grace = element.find('grace') is not None
                    cue = element.find('cue') is not None
                    rest = element.find('rest') is not None
                    chord = element.find('chord') is not None
                    duration = Fraction(0) if grace else _positive(element.findtext('duration'), identifier) / divisions
                    voice = element.findtext('voice', '')
                    if chord:
                        if anchor is None or anchor[2] != grace or rest or anchor[4]:
                            raise UnsupportedScore('Chord lacks a compatible preceding pitched note')
                        onset = anchor[0]
                        if duration > anchor[1] or voice != anchor[3]:
                            raise UnsupportedScore('Chord duration/voice contradicts its base note')
                    else:
                        onset = cursor
                        anchor = (onset, duration, grace, voice, rest)
                        cursor += duration
                    end = onset + duration
                    high = max(high, cursor, end)
                    if grace or cue or rest:
                        if sound or notation:
                            raise UnsupportedScore('Ties on grace/cue/rest notes require separate handling')
                        if grace or cue:
                            excluded.append((identifier, 'grace' if grace else 'cue'))
                        continue
                    try:
                        staff = int(element.findtext('staff', '1'))
                    except ValueError as exc:
                        raise UnsupportedScore('Non-integral staff') from exc
                    if staff <= 0:
                        raise UnsupportedScore('Staff must be positive')
                    raw.append(dict(id=identifier, part=part_id, pitch=_pitch(element), start=onset,
                        end=end, measure=ordinal, staff=staff, voice=voice,
                        xml_id=element.get('id') or element.get('ID'), measure_label=label,
                        sound_ties=sound, notation_ties=notation,
                        playback_conditions=conditions, problems=problems))
                elif tag not in {'direction', 'barline', 'print', 'sound', 'harmony', 'figured-bass',
                                  'grouping', 'link', 'bookmark'}:
                    raise UnsupportedScore(f'Unsupported measure element: {tag}')
            if high <= start:
                raise UnsupportedScore('Empty measure has no recoverable duration')
            measures.append((part_id, ordinal, label, start, high))
            position = high
    if not part_ids:
        raise UnsupportedScore('No parts found')
    part_bounds = defaultdict(list)
    for part_id, ordinal, label, start, end in measures:
        part_bounds[part_id].append((ordinal, label, start, end))
    if len(part_bounds) != len(part_ids) or len({tuple(rows) for rows in part_bounds.values()}) != 1:
        raise UnsupportedScore('Controlling measure clocks/labels disagree between parts')
    # A single exact score clock, including changes of divisions and tuplets.
    unit = 1
    for value in [v for row in raw for v in (row['start'], row['end'])] + [
            v for row in measures for v in row[3:]]:
        unit = lcm(unit, value.denominator)
    written = tuple(SourceNote(
        WrittenNote(row['id'], row['part'], row['pitch'], int(row['start'] * unit),
            int(row['end'] * unit), row['measure'], row['staff'], row['voice']),
        row['xml_id'], row['measure_label'], row['sound_ties'], row['notation_ties'],
        row['playback_conditions'], row['problems'])
        for row in raw)
    bounds = tuple(SourceMeasure(part, ordinal, label, int(start * unit), int(end * unit))
                   for part, ordinal, label, start, end in measures)
    edges, verified, issues = _resolve(written)
    representation = score_events((row.note for row in written), edges, verified_note_ids=verified)
    return SourceScore(unit, written, bounds, edges, verified, issues, tuple(excluded), representation)
