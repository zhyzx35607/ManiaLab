from manialab.domain import Note
from scripts.check_expression_rhythm import scope_rows


def test_missing_attacks_are_not_counted_as_represented_by_overlapping_ln():
    notes = [Note(0, 1000, id='hold', voice='piano'),
             Note(100, 100, id='omitted', voice='piano'),
             Note(200, 200, id='capacity', voice='piano'),
             Note(300, 300, id='selected', voice='piano')]
    metadata = {n.id:dict(section=0) for n in notes}
    trace = [dict(id=n.id, reason='selected' if n.id in ('hold','selected') else 'omitted',
                  rhythm_policy=dict(keep=n.id!='omitted')) for n in notes]
    rows = scope_rows(notes, [notes[0],notes[-1]], metadata, dict(decisions=trace))
    assert len(rows)==1
    row = rows[0]
    assert row['candidate_attacks']==4 and row['selected_attacks']==2
    assert row['policy_omitted_attacks']==1 and row['allowed_but_unselected_attacks']==1
    assert row['longest_unselected_run']==dict(attacks=2,first=100,last=200)
    assert row['max_selected_gap_ms']==300


def test_chord_count_and_scope_boundaries_do_not_inflate_or_join_omission_runs():
    notes = [Note(100,100,id='low',pitch=48,voice='piano'),
             Note(100,100,id='high',pitch=72,voice='piano'),
             Note(200,200,id='next',voice='piano')]
    metadata = {n.id:dict(section=int(n.id=='next')) for n in notes}
    report = dict(decisions=[dict(id=n.id,reason='group_capacity_or_density') for n in notes])
    rows = scope_rows(notes,[],metadata,report)
    assert len(rows)==2 and all(r['candidate_attacks']==1 for r in rows)
    assert all(r['longest_unselected_run']['attacks']==1 for r in rows)
    assert all(r['max_selected_gap_ms'] is None for r in rows)
