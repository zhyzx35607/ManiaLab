"""Keep generation context separate from claims about the selected notes."""
from collections import Counter
from copy import deepcopy
from math import isfinite


def generation_context(selection):
    """Return the source snapshots that still apply when the note list changes."""
    return {field: selection[field] for field in ('structure', 'expression') if field in selection}


def _is_created_key(value):
    return type(value) in (int, float, str) and (type(value) is not float or isfinite(value))


def resolve_generation_origin(candidate, candidates=()):
    """Resolve a candidate's stable source scope, including pre-marker history."""
    by_created = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        created = item.get('created')
        if _is_created_key(created):
            by_created.setdefault(created, []).append(item)
    current = candidate
    visited = set()
    while isinstance(current, dict):
        saved = current.get('generation_origin')
        if isinstance(saved, dict) and saved.get('kind') in (
            'regional_generation', 'full_song_generation'
        ):
            return deepcopy(saved)
        if current.get('region') is not None:
            return dict(kind='regional_generation', region=deepcopy(current['region']))
        selection = current.get('selection')
        if isinstance(selection, dict) and selection.get('scope') == 'regional_output':
            return dict(kind='regional_generation', region=None)
        if isinstance(saved, dict) and saved.get('kind') == 'unknown':
            return deepcopy(saved)
        edits = current.get('edits')
        if edits is None or isinstance(edits, list) and not edits:
            return dict(kind='full_song_generation')
        if not isinstance(edits, list):
            return dict(kind='unknown')
        last_edit = edits[-1]
        parent_created = last_edit.get('parent_created') if isinstance(last_edit, dict) else None
        if parent_created is None:
            return dict(kind='unknown')
        if not _is_created_key(parent_created) or parent_created in visited:
            return dict(kind='unknown')
        visited.add(parent_created)
        parents = by_created.get(parent_created, ())
        if len(parents) != 1:
            return dict(kind='unknown')
        current = parents[0]
    return dict(kind='unknown')


def regional_selection_report(proposal, output, locked):
    """Report actual output; retain the full proposal with an explicit scope.

    The selector ran before regional filtering and knows nothing about locked
    notes from an older analysis or manual edits. Its omitted reasons and solver
    objective cannot describe the merged output. Do not fabricate those missing
    decisions by joining reused IDs across analyses.
    """
    locked_ids = {note.id for note in locked}
    count = len(output)
    held = sum(note.hold for note in output)
    report = dict(scope='regional_output', selected=count,
        locked=sum(note.id in locked_ids for note in output),
        generated=sum(note.id not in locked_ids for note in output),
        actual_ln_ratio=round(held/max(1, count), 4),
        selected_voices=dict(Counter(note.voice for note in output)),
        decisions=[dict(id=note.id, voice=note.voice, decision='selected',
            reason='locked_existing' if note.id in locked_ids else 'regional_regenerated',
            chart_start=note.start, chart_end=note.end) for note in output],
        proposal=dict(deepcopy(proposal), scope='full_song_proposal'),
        message=f'局部定轨完成；整谱实际 LN 比例 {held/max(1, count):.1%}。区域外物件已锁定。')
    report.update(deepcopy(generation_context(proposal)))
    return report
