from copy import deepcopy
from dataclasses import replace

import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import select_expression


def select(notes, *, extra_trace=None, trace_overrides=None):
    metadata = {n.id: dict(section=0, role='support') for n in notes}
    trace = {n.id: dict(id=n.id, start=n.evidence_start, end=n.evidence_end,
                       expression_end=n.end, decision='candidate', reason='supported_sustain',
                       mode='sustain') for n in notes}
    trace.update(extra_trace or {})
    for identity, changes in (trace_overrides or {}).items():
        trace[identity].update(changes)
    before = deepcopy(notes)
    chosen, report = select_expression(notes, metadata, trace, parameters(), 500)
    assert notes == before
    return chosen, {row['id']: row for row in report['decisions']}


def candidate(**changes):
    note = Note(1000, 1500, pitch=64, strength=.8, source='pitch', voice='other',
                instrument_id='strings', id='early', evidence_start=920, evidence_end=1505,
                release_reason='note_off')
    return replace(note, **changes)


@pytest.mark.parametrize('end', [1000, 1500])
@pytest.mark.parametrize('reverse', [False, True])
def test_equivalent_duplicates_retain_better_alignment_in_either_order(end, reverse):
    first = candidate(end=end)
    better = replace(first, id='aligned', evidence_start=995, evidence_end=first.evidence_end+5)
    notes = [first, better]
    chosen, trace = select(notes[::-1] if reverse else notes)
    assert chosen == [better]
    assert trace[first.id]['reason'] == 'duplicate_pitch'
    assert trace[better.id]['decision'] == 'selected'


@pytest.mark.parametrize('reverse', [False, True])
def test_equal_scores_have_a_stable_id_tie_break(reverse):
    a = candidate(id='a', evidence_start=990)
    b = replace(a, id='b', evidence_start=1010)
    chosen, _ = select([b, a] if reverse else [a, b])
    assert chosen == [a]


@pytest.mark.parametrize('changes', [
    dict(end=1700), dict(end=1000), dict(strength=.7), dict(lane=2),
    dict(release_reason='same_pitch_reattack'), dict(phrase=True),
    dict(support=('other-attack',)), dict(support_coverage=.8), dict(raw='different sample'),
])
def test_alignment_does_not_replace_a_different_expression_or_chord_rank(changes):
    first = candidate()
    alternative = replace(first, id='closer', evidence_start=995, **changes)
    chosen, _ = select([first, alternative])
    assert chosen == [first]


@pytest.mark.parametrize('changes', [dict(instrument_id='harp'), dict(voice='piano'), dict(pitch=65)])
def test_distinct_sources_and_pitches_are_not_collapsed(changes):
    first = candidate()
    second = replace(first, id='distinct', **changes)
    chosen, _ = select([first, second])
    assert {n.id for n in chosen} == {first.id, second.id}


@pytest.mark.parametrize('changes', [dict(phrase=True), dict(support=('interior',))])
def test_phrase_gestures_keep_their_original_representative(changes):
    first = candidate(**changes)
    second = replace(first, id='closer', evidence_start=995)
    chosen, _ = select([first, second])
    assert chosen == [first]


@pytest.mark.parametrize('anchor', ['early', 'closer'])
def test_referenced_phrase_anchor_is_not_substituted(anchor):
    first = candidate()
    second = replace(first, id='closer', evidence_start=995)
    chosen, trace = select([first, second], extra_trace={'interior': dict(id='interior',
        decision='omitted', reason='represented_by_phrase', represented_by=anchor)})
    assert chosen == [first]
    assert trace['interior']['reason'] == (
        'represented_by_phrase' if anchor == first.id else 'phrase_not_selected')


@pytest.mark.parametrize('changes', [dict(reason='articulated_attack_run'),
                                   dict(mode='onsets'), dict(articulation={'tap': False})])
def test_different_expression_evidence_does_not_get_substituted(changes):
    first = candidate()
    second = replace(first, id='closer', evidence_start=995)
    chosen, _ = select([first, second], trace_overrides={second.id: changes})
    assert chosen == [first]
