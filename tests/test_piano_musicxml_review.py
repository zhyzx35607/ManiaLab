"""Independent review controls for incomplete source evidence and score clocks."""
import pytest

from scripts.piano_musicxml_events import UnsupportedScore, parse_musicxml


def note(identifier, duration=1, *, ties=(), step='C', attributes=''):
    sound = ''.join(f'<tie type="{kind}"/>' for kind in ties)
    visual = ''.join(f'<tied type="{kind}"/>' for kind in ties)
    return (f'<note id="{identifier}" {attributes}><pitch><step>{step}</step><octave>4</octave></pitch>'
            f'<duration>{duration}</duration>{sound}<voice>1</voice><staff>1</staff>'
            f'<notations>{visual}</notations></note>')


def part(identifier, contents, *, measure_attributes=''):
    return f'<part id="{identifier}">' + ''.join(
        f'<measure number="{index + 1}" {measure_attributes}>' +
        ('<attributes><divisions>1</divisions></attributes>' if index == 0 else '') +
        content + '</measure>' for index, content in enumerate(contents)) + '</part>'


def score(*parts):
    return '<score-partwise>' + ''.join(parts) + '</score-partwise>'


def labels(result):
    roles = {view.note.id: view.role for view in result.representation.notation}
    return {row.xml_id: roles[row.note.id] for row in result.written}


@pytest.mark.parametrize('first_ties,second_ties', [(('start',), ()), ((), ('stop',))])
def test_one_sided_tie_masks_undeclared_same_pitch_boundary_partner(first_ties, second_ties):
    xml = score(part('P1', [note('a', ties=first_ties) + note('b', ties=second_ties) + note('safe', step='E')]))
    result = parse_musicxml(xml)
    assert labels(result) == {'a': 'unknown', 'b': 'unknown', 'safe': 'attack'}
    assert not result.ties


def test_one_sided_tie_masks_all_possible_unison_partners_without_guessing():
    content = (note('a', ties=('start',)) + note('b') +
               '<backup><duration>1</duration></backup>' + note('c') + note('safe', step='E'))
    result = parse_musicxml(score(part('P1', [content])))
    assert labels(result) == {'a': 'unknown', 'b': 'unknown', 'c': 'unknown', 'safe': 'attack'}
    assert not result.ties


def test_unrelated_pitch_at_an_unmatched_boundary_remains_verified():
    result = parse_musicxml(score(part('P1', [note('a', ties=('start',)) + note('different', step='E')])))
    assert labels(result) == {'a': 'unknown', 'different': 'attack'}


def test_repeat_conditional_note_is_not_an_unconditional_attack_label():
    result = parse_musicxml(score(part('P1', [note('conditional', attributes='time-only="2"') + note('safe', step='E')])))
    assert labels(result) == {'conditional': 'unknown', 'safe': 'attack'}


def test_repeat_conditional_note_masks_its_entire_otherwise_resolved_tie_chain():
    content = (note('a', ties=('start',)) +
               note('conditional', ties=('stop', 'start'), attributes='time-only="2"') +
               note('b', ties=('stop',)) + note('safe', step='E'))
    result = parse_musicxml(score(part('P1', [content])))
    assert labels(result) == {'a': 'unknown', 'conditional': 'unknown', 'b': 'unknown', 'safe': 'attack'}
    assert not result.ties


def test_controlling_measure_labels_cannot_silently_start_at_different_times():
    xml = score(part('P1', [note('a'), note('b')]),
                part('P2', [note('c', duration=2), note('d')]))
    with pytest.raises(UnsupportedScore):
        parse_musicxml(xml)


def test_aligned_controlling_parts_preserve_all_independent_attacks():
    result = parse_musicxml(score(part('P1', [note('a'), note('b')]),
                                  part('P2', [note('c'), note('d')])))
    assert labels(result) == {'a': 'attack', 'b': 'attack', 'c': 'attack', 'd': 'attack'}
    assert not result.issues


def test_non_controlling_measures_are_explicitly_unsupported():
    xml = score(part('P1', [note('a')]),
                part('P2', [note('b')], measure_attributes='non-controlling="yes"'))
    with pytest.raises(UnsupportedScore):
        parse_musicxml(xml)
