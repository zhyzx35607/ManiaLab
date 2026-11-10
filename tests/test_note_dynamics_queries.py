import copy

import pytest

from scripts.research_tsumugi_velocity import program_hypotheses, validate_queries


def query(**changes):
    return dict(dict(onset=.25, offset=.75, pitch=60, program=40,
                     is_drum=False, program_basis='explicit_assumption'), **changes)


@pytest.mark.parametrize('changes', [dict(program=-1), dict(program=128), dict(pitch=20),
    dict(pitch=109), dict(program_basis=''), dict(offset=1.1), dict(onset=float('nan')),
    dict(onset=.99999998, offset=.99999999)])
def test_rejects_silent_embedding_clamps_and_collapsed_intervals(changes):
    with pytest.raises(ValueError):
        validate_queries([query(**changes)], 1.)


def test_preserves_repeated_chords_and_program_hypotheses():
    queries=[query(),query(program=41),query(pitch=64),query(onset=.75,offset=1.)]
    original=copy.deepcopy(queries)
    validate_queries(queries,1.)
    assert queries==original


def test_enumerates_entire_family_without_claiming_exact_gm():
    event=dict(instrument_class='strings',is_drum=False,program=-1)
    labels={'strings':{'members':[{'program_number':41},{'program_number':40},{'program_number':41}]}}
    hypotheses=program_hypotheses(event,labels,{})
    assert hypotheses==[(40,'assumed_member_of_predicted_broad_class'),
                        (41,'assumed_member_of_predicted_broad_class')]
    assert event['program']==-1
    with pytest.raises(ValueError):
        program_hypotheses(dict(instrument_class='unknown',is_drum=False),labels,{})


def test_non_gm_representative_is_explicit_assumption():
    assert program_hypotheses(dict(instrument_class='vocal',is_drum=False),{}, {'vocal':65})==[
        (65,'upstream_non_gm_representative_assumption')]
