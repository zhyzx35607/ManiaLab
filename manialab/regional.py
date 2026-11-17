"""Validate expression dependencies that cross a regional rewrite boundary."""
from .domain import note_from
from .timing import align_candidates, chart_clock_notes


def validate_phrase_scope(analysis, candidates, selection, region, tempo, locked=()):
    """Regional replacements cannot rely on, or duplicate, locked old objects.

    The previous locked object may be a tap, edited hold, or a reused ID from
    another analysis. Even matching phrase flags do not prove that it carries
    the current replacement. Require the user to rewrite the related attacks
    together instead of silently deleting an interior attack. Conversely, a
    newly selected phrase cannot coexist with the internal attack it replaces
    when that old attack is locked, even if an edit moved it away from the
    current analysis position. IDs identify this conflict, not source identity.
    """
    represented = [d for d in selection.get('decisions', [])
        if d.get('reason') in ('represented_by_phrase', 'phrase_not_selected')]
    if not represented:
        return
    identities = {d['id'] for d in represented} | {d['represented_by'] for d in represented}
    raw = [note_from(n) for n in analysis.get('candidates', []) if n['id'] in identities]
    # The ordinary generation path has already checked the timing gate. Use
    # the very same conversion here, including clipping and integer rounding.
    aligned, _ = align_candidates(chart_clock_notes(raw, tempo['audio_shift_ms']), tempo, confirmed=True)
    attacks = {n.id: n.start for n in aligned}
    proposed = {n.id: n.start for n in candidates}
    locked_ids = {n.id for n in locked}
    begin, end = region
    for decision in represented:
        start = attacks.get(decision['id'])
        anchor = proposed.get(decision['represented_by'], attacks.get(decision['represented_by']))
        anchor_inside = anchor is not None and begin <= anchor < end
        missing_anchor = start is not None and begin <= start < end and not anchor_inside
        # Only a selected, noncollapsed phrase represents an internal attack.
        # A rejected proposal must not block an otherwise harmless locked note.
        duplicate_internal = (anchor_inside and decision['reason'] == 'represented_by_phrase'
            and decision['id'] in locked_ids)
        if missing_anchor or duplicate_internal:
            raise ValueError('局部范围切开了乐句替代关系；请扩大重写范围，包含乐句起点及其内部起音，或重新生成整谱')
