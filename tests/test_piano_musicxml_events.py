"""Source declarations, ambiguous ties and exact MusicXML clocks."""
from fractions import Fraction

import pytest

from scripts.piano_musicxml_events import UnsupportedScore, parse_musicxml


def n(identifier, duration=1, ties=(), *, voice='1', staff=1, visual=None, extra='', attrs='', step='C'):
    visual = ties if visual is None else visual
    sound = ''.join(f'<tie type="{kind}"/>' for kind in ties)
    written = ''.join(f'<tied type="{kind}"/>' for kind in visual)
    return (f'<note id="{identifier}" {attrs}>{extra}<pitch><step>{step}</step><octave>4</octave></pitch>'
            f'<duration>{duration}</duration>{sound}<voice>{voice}</voice><staff>{staff}</staff>'
            f'<notations>{written}</notations></note>')


def score(*measures, divisions=1):
    body = ''.join(f'<measure number="{i}">' +
                   (f'<attributes><divisions>{divisions}</divisions></attributes>' if i == 0 else '') +
                   content + '</measure>' for i, content in enumerate(measures))
    return f'<score-partwise><part id="P1">{body}</part></score-partwise>'


def roles(result):
    labels = {view.note.id: view.role for view in result.representation.notation}
    return {row.xml_id: labels[row.note.id] for row in result.written}


def intervals(result):
    return [(Fraction(event.start_tick, result.ticks_per_quarter),
             Fraction(event.end_tick, result.ticks_per_quarter)) for event in result.representation.events]


def test_three_piece_sustain_keeps_one_score_attack_across_staff_voice_and_bar():
    result = parse_musicxml(score(n('a', ties=('start',)),
        n('b', ties=('stop', 'start'), staff=2, voice='7') + n('c', ties=('stop',), staff=2, voice='7')))
    assert roles(result) == {'a': 'attack', 'b': 'continuation', 'c': 'continuation'}
    assert intervals(result) == [(0, 3)]
    assert len(result.ties) == 2
    assert not result.issues
    assert [row.note.voice for row in result.written] == ['1', '7', '7']
    assert [row.measure_label for row in result.written] == ['0', '1', '1']


def test_same_pitch_repeated_notes_stay_separate_without_declarations():
    result = parse_musicxml(score(n('a') + n('b')))
    assert roles(result) == {'a': 'attack', 'b': 'attack'}
    assert not result.ties


def parallel(numbered=False):
    a = n('a', ties=('start',))
    b = n('b', ties=('start',), voice='2', staff=2)
    c = n('c', ties=('stop',))
    d = n('d', ties=('stop',), voice='2', staff=2)
    if numbered:
        a, c = (x.replace('<tied type=', '<tied number="1" type=') for x in (a, c))
        b, d = (x.replace('<tied type=', '<tied number="2" type=') for x in (b, d))
    return score(a + '<backup><duration>1</duration></backup>' + b + d +
                 '<backup><duration>1</duration></backup>' + c)


def test_parallel_unison_voice_numbers_do_not_invent_identity():
    result = parse_musicxml(parallel())
    assert set(roles(result).values()) == {'unknown'}
    assert len(result.issues) == 4
    assert not result.ties and not result.verified_note_ids


def test_explicit_notation_numbers_disambiguate_parallel_sound_ties():
    result = parse_musicxml(parallel(numbered=True))
    assert roles(result) == {'a': 'attack', 'b': 'attack', 'd': 'continuation', 'c': 'continuation'}
    assert len(result.representation.events) == 2 and len(result.ties) == 2
    assert not result.issues


@pytest.mark.parametrize('first,second', [
    (n('a', ties=('start',)), n('b')),
    (n('a'), n('b', ties=('stop',))),
    (n('a', visual=('start',)), n('b', ties=('stop',))),
    (n('a', ties=('start',), visual=('stop',)), n('b', ties=('stop',))),
    (n('a', ties=('start', 'start')), n('b', ties=('stop',))),
    (n('a', ties=('start',)).replace('<tie type=', '<tie time-only="1" type='), n('b', ties=('stop',))),
    (n('a', ties=('start',), attrs='attack="0.1"'), n('b', ties=('stop',))),
])
def test_incomplete_or_disagreeing_ties_are_masked_without_affecting_safe_notes(first, second):
    result = parse_musicxml(score(first + second + n('safe', step='E')))
    assert not result.ties
    assert roles(result)['safe'] == 'attack'
    assert 'unknown' in roles(result).values()


def test_unknown_start_masks_a_later_locally_unambiguous_continuation():
    xml = score(n('a', ties=('stop', 'start')) + n('b', ties=('stop', 'start')) + n('c', ties=('stop',)))
    result = parse_musicxml(xml)
    assert set(roles(result).values()) == {'unknown'}
    assert not result.ties


def test_sound_only_ties_are_supported_without_visual_marks():
    result = parse_musicxml(score(n('a', ties=('start',), visual=()) + n('b', ties=('stop',), visual=())))
    assert intervals(result) == [(0, 2)] and not result.issues


def test_shorter_chord_does_not_shorten_the_next_note_onset():
    result = parse_musicxml(score(n('a', duration=4) + n('b', duration=1, extra='<chord/>', step='E') + n('c')))
    by_id = {row.xml_id: row.note for row in result.written}
    assert (by_id['b'].start_tick, by_id['b'].end_tick) == (0, 1)
    assert by_id['c'].start_tick == 4
    assert result.measures[0].end_tick == 5


def test_exact_tuplets_boundary_divisions_change_pickup_and_forward():
    result = parse_musicxml(score(n('pickup'),
        '<attributes><divisions>5</divisions></attributes>' + n('a') +
        '<forward><duration>1</duration></forward>' + n('b'), divisions=3))
    assert result.ticks_per_quarter == 15
    assert [(row.note.start_tick, row.note.end_tick) for row in result.written] == [(0, 5), (5, 8), (11, 14)]
    assert [(m.start_tick, m.end_tick) for m in result.measures] == [(0, 5), (5, 14)]


def test_grace_and_cue_are_explicit_exclusions_and_do_not_hide_clock_movement():
    result = parse_musicxml(score(n('g', extra='<grace/>') + n('gc', extra='<grace/><chord/>', step='E') +
        n('cue', extra='<cue/>') + n('played')))
    assert len(result.excluded) == 3
    assert roles(result) == {'played': 'attack'}
    assert intervals(result) == [(1, 2)]


def test_structural_ids_survive_missing_or_duplicate_xml_ids_and_separate_parts():
    xml = score(n('same') + n('same')).replace('</score-partwise>',
        '<part id="P2"><measure number="0"><attributes><divisions>1</divisions></attributes>' +
        n('same', duration=2) + '</measure></part></score-partwise>')
    result = parse_musicxml(xml)
    assert len({row.note.id for row in result.written}) == 3
    assert len(result.representation.events) == 3


@pytest.mark.parametrize('content', [
    '<backup><duration>1</duration></backup>' + n('a'),
    n('a') + '<attributes><divisions>3</divisions></attributes>' + n('b'),
    n('a') + n('b', duration=2, extra='<chord/>'),
    n('a', extra='<chord/>'),
    n('a', ties=('start',), extra='<grace/>'),
    n('a', duration=0), n('a', duration=-1), n('a', duration='nan'),
    n('a').replace('<step>C</step>', '<step>C</step><alter>0.5</alter>'),
    '<attributes><transpose><chromatic>2</chromatic></transpose></attributes>' + n('a'),
    '<unknown-clock/>' + n('a'),
])
def test_unsupported_timing_or_notes_fail_explicitly(content):
    with pytest.raises(UnsupportedScore):
        parse_musicxml(score(content))


def test_explicit_conflicting_tie_numbers_are_not_ignored():
    xml = score(n('a', ties=('start',)).replace('<tied type=', '<tied number="1" type=') +
        n('b', ties=('stop',)).replace('<tied type=', '<tied number="2" type='))
    result = parse_musicxml(xml)
    assert roles(result) == {'a': 'unknown', 'b': 'unknown'}
    assert not result.ties


def test_unsupported_let_ring_is_not_an_assumed_release_or_attack():
    result = parse_musicxml(score(n('a', visual=('let-ring',))))
    assert roles(result) == {'a': 'unknown'}


def test_sound_start_and_stop_separated_by_a_gap_are_not_merged():
    result = parse_musicxml(score(n('a', ties=('start',)) +
        '<forward><duration>1</duration></forward>' + n('b', ties=('stop',))))
    assert not result.ties and set(roles(result).values()) == {'unknown'}
