"""Keep a proposal's selection claims separate from a regionally merged chart."""
from collections import Counter
from copy import deepcopy


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
    # Preserve the generation's own structure snapshot for export bookmarks.
    for field in ('structure', 'expression'):
        if field in proposal:
            report[field] = deepcopy(proposal[field])
    return report
