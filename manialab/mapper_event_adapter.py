"""Convert ManiaLab note-like records to Mapperatorinator's event stream."""
from vendor.mapperatorinator.event import Event, EventType


def upstream_events(notes):
    """Bridge to Mapperatorinator's event vocabulary; pair LN endpoints by lane."""
    events = []
    for note in notes:
        events.append((note.start, note.lane,
                       EventType.HOLD_NOTE if note.hold else EventType.CIRCLE))
        if note.hold:
            events.append((note.end, note.lane, EventType.HOLD_NOTE_END))
    result = []
    for time_ms, lane, kind in sorted(
            events, key=lambda event: (event[0], event[2] != EventType.HOLD_NOTE_END, event[1])):
        result.extend((
            Event(kind),
            Event(EventType.TIME_SHIFT, time_ms),
            Event(EventType.MANIA_COLUMN, lane),
        ))
    return result
