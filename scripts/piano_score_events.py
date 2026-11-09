"""Keep score sustain events separate from their written note segments.

This research representation accepts explicitly verified MusicXML tie relations.
It does not repair an importer's missing ties, infer melody identity from voice
numbers, or treat a notated end as a measured acoustic/finger release.
"""
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Literal


@dataclass(frozen=True)
class WrittenNote:
    id: str
    part: str
    pitch: int
    start_tick: int
    end_tick: int
    measure: int
    staff: int
    voice: str


@dataclass(frozen=True)
class ScoreEvent:
    id: str
    part: str
    pitch: int
    start_tick: int
    end_tick: int
    written_ids: tuple[str, ...]
    supervision_eligible: bool


@dataclass(frozen=True)
class WrittenView:
    note: WrittenNote
    event_id: str
    role: Literal['attack', 'continuation', 'unknown']


@dataclass(frozen=True)
class ScoreRepresentation:
    events: tuple[ScoreEvent, ...]
    notation: tuple[WrittenView, ...]


def score_events(
    notes: Iterable[WrittenNote],
    ties: Iterable[tuple[str, str]],
    *,
    verified_note_ids: Iterable[str],
) -> ScoreRepresentation:
    """Build a lossless written view and a tie-collapsed score-event view.

    Times must already share one tick unit within each part. Tie edges must have
    been checked against the source declarations, including voice/pitch/time
    ambiguities. ``verified_note_ids`` means that both the presence *and absence*
    of incoming/outgoing ties have been verified. Omitted IDs are unknown, never
    automatic attack labels. An unknown member masks the entire tie chain.

    Voice/staff/measure changes do not break a verified tie. Their values stay on
    the written segments; no single melody/hand label is invented for the event.
    ``attack`` and ``supervision_eligible`` refer to score-event supervision only.
    Parallel written unisons remain separate events; they need not be separate
    audible attacks, and no returned endpoint is an acoustic release label.
    """
    written = tuple(notes)
    by_id = {note.id: note for note in written}
    if len(by_id) != len(written):
        raise ValueError('Written-note IDs must be globally unique')
    for note in written:
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in (note.start_tick, note.end_tick, note.pitch)):
            raise ValueError('Score ticks and MIDI pitch must be finite integers')
        if not isinstance(note.id, str) or not isinstance(note.part, str):
            raise ValueError('Written-note ID and part must be strings')
        if not note.id or not note.part or note.end_tick <= note.start_tick:
            raise ValueError('Expected a named part and positive written duration')
        if not 0 <= note.pitch <= 127:
            raise ValueError('Expected MIDI pitch in 0..127')
    verified = set(verified_note_ids)
    if not verified <= by_id.keys():
        raise ValueError('Verified IDs must refer to written notes')
    successor, predecessor = {}, {}
    for source_id, target_id in ties:
        if source_id not in by_id or target_id not in by_id:
            raise ValueError('Tie endpoint is missing')
        source, target = by_id[source_id], by_id[target_id]
        if source_id == target_id:
            raise ValueError('A note cannot tie to itself')
        if source.part != target.part or source.pitch != target.pitch:
            raise ValueError('A tie must retain part and pitch')
        if source.end_tick != target.start_tick:
            raise ValueError('Tie segments must meet exactly in score time')
        if source_id in successor or target_id in predecessor:
            raise ValueError('Duplicate or branching tie relation')
        successor[source_id], predecessor[target_id] = target_id, source_id
    order = lambda note: (note.part, note.start_tick, note.pitch, note.id)
    events, views, visited = [], {}, set()
    for root in sorted((note for note in written if note.id not in predecessor), key=order):
        members = []
        current = root.id
        while current is not None:
            if current in visited:
                raise ValueError('Tie graph must contain disjoint acyclic chains')
            visited.add(current)
            members.append(current)
            current = successor.get(current)
        eligible = all(identifier in verified for identifier in members)
        event = ScoreEvent(root.id, root.part, root.pitch, root.start_tick,
            by_id[members[-1]].end_tick, tuple(members), eligible)
        events.append(event)
        for i, identifier in enumerate(members):
            role = ('attack' if i == 0 else 'continuation') if eligible else 'unknown'
            views[identifier] = WrittenView(by_id[identifier], event.id, role)
    if visited != set(by_id):
        raise ValueError('Tie graph contains an unresolved cycle')
    if len(events) != len(written) - len(successor):
        raise ValueError('Tie graph did not preserve all written segments')
    return ScoreRepresentation(tuple(events), tuple(views[note.id] for note in sorted(written, key=order)))
