"""Fixed MDB subclass comparison with Magenta's E-GMD eight-hit ontology.

These are dataset labels, not a universal GM interpretation. In particular,
E-GMD pitch52 is Crash2_Edge; it does not establish training support for MDB's
splash or china cymbals. Unknown items stay visible and never match each other.
"""
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path

from scripts.evaluate_adt_drums import match_times

CLASSES = ('kick', 'snare', 'hi-hat', 'tom', 'ride', 'bell', 'crash', 'clave')
PITCH_CLASSES = {36: 'kick', 38: 'snare', 46: 'hi-hat', 48: 'tom',
                 51: 'ride', 53: 'bell', 49: 'crash', 75: 'clave'}
SUBCLASS_CLASSES = {
    'KD': 'kick',
    'SD': 'snare', 'SDB': 'snare', 'SDD': 'snare', 'SDF': 'snare',
    'SDG': 'snare', 'SDNS': 'snare', 'SST': 'snare',
    'CHH': 'hi-hat', 'OHH': 'hi-hat', 'PHH': 'hi-hat', 'TMB': 'hi-hat',
    'HIT': 'tom', 'MHT': 'tom', 'HFT': 'tom', 'LFT': 'tom',
    'RDC': 'ride', 'RDB': 'bell', 'CRC': 'crash',
    'CHC': 'unknown', 'SPC': 'unknown',
}
SUBCLASS_PARENT = {
    'KD': 'KD',
    'SD': 'SD', 'SDB': 'SD', 'SDD': 'SD', 'SDF': 'SD', 'SDG': 'SD', 'SDNS': 'SD',
    'CHH': 'HH', 'OHH': 'HH', 'PHH': 'HH',
    'HIT': 'TT', 'MHT': 'TT', 'HFT': 'TT', 'LFT': 'TT',
    'RDC': 'CY', 'RDB': 'CY', 'CRC': 'CY', 'CHC': 'CY', 'SPC': 'CY',
    'SST': 'OT', 'TMB': 'OT',
}
MAX_TIMESTAMP_ROUNDING_SECONDS = Decimal('0.0000005')


def _annotations(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f'Invalid annotation at {path}:{line_number}')
        try:
            timestamp = Decimal(parts[0])
        except InvalidOperation as error:
            raise ValueError(f'Invalid timestamp at {path}:{line_number}') from error
        if not timestamp.is_finite() or timestamp < 0:
            raise ValueError(f'Invalid timestamp at {path}:{line_number}')
        rows.append((timestamp, parts[1], line_number))
    return rows


def load_mdb_subclasses(class_path, subclass_path):
    """Join official rows without nearest-time reassignment or deduplication.

    Rockabilly stores subclass times with nine decimals and class times with
    six. At most half a microsecond of decimal rounding is accepted. Output
    retains the class timestamp used by existing regression scores, plus the
    original subclass timestamp and both source line numbers.
    """
    class_path, subclass_path = Path(class_path), Path(subclass_path)
    coarse, fine = _annotations(class_path), _annotations(subclass_path)
    if len(coarse) != len(fine):
        raise ValueError('Class/subclass row counts differ')
    events = []
    max_delta = Decimal(0)
    for (ct, parent, cline), (st, subclass, sline) in zip(coarse, fine):
        if subclass not in SUBCLASS_CLASSES:
            raise ValueError('Unreviewed MDB subclass: ' + subclass)
        if SUBCLASS_PARENT[subclass] != parent:
            raise ValueError(f'Class/subclass category mismatch at lines {cline}/{sline}')
        delta = abs(ct - st)
        if delta > MAX_TIMESTAMP_ROUNDING_SECONDS:
            raise ValueError(f'Class/subclass onset mismatch at lines {cline}/{sline}')
        max_delta = max(max_delta, delta)
        events.append(dict(time_ms=float(ct) * 1000, kind=SUBCLASS_CLASSES[subclass],
            original_class=parent, subclass=subclass, subclass_time_ms=float(st) * 1000,
            class_line=cline, subclass_line=sline))
    return dict(events=events, class_sha256=hashlib.sha256(class_path.read_bytes()).hexdigest(),
        subclass_sha256=hashlib.sha256(subclass_path.read_bytes()).hexdigest(),
        maximum_timestamp_difference_seconds=float(max_delta),
        unknown_reference_count=sum(e['kind'] == 'unknown' for e in events),
        notice='Fixed eight-hit compatibility mapping; unknown references are unsupported classes, not verified model misses.')


def prediction_events(events):
    """Replace the diagnostic kind using the preserved canonical model pitch."""
    return [dict(event, kind=PITCH_CLASSES.get(event['pitch'], 'unknown'),
                 time_ms=event['onset'] * 1000) for event in events]


def score(reference, predicted, tolerance):
    """Score eight compatible classes; retain unknowns with no allowed match.

    Callers should report eight-class and unknown counts separately. Generic
    unknown predictions cannot establish a correct timbre for unsupported
    reference classes. Neither side is silently discarded from this result.
    """
    known = set(CLASSES) | {'unknown'}
    if any(e['kind'] not in known for e in [*reference, *predicted]):
        raise ValueError('Unmapped event class')
    output = {}
    for kind in (*CLASSES, 'unknown'):
        refs = [e for e in reference if e['kind'] == kind]
        preds = [e for e in predicted if e['kind'] == kind]
        if kind != 'unknown':
            result = match_times([e['time_ms'] for e in refs], [e['time_ms'] for e in preds], tolerance)
        else:
            # Also validates clocks/tolerance, but do not use unknown matches.
            match_times([e['time_ms'] for e in refs], [], tolerance)
            result = match_times([], [e['time_ms'] for e in preds], tolerance)
            result.update(reference=len(refs), missing=len(refs),
                          unmatched_reference=list(range(len(refs))),
                          evaluable=False, unsupported_reference=len(refs),
                          notice='Unknown classes do not match one another; report separately from supported-class quality.')
        result.update(reference_events=refs, predicted_events=preds)
        output[kind] = result
    return output
