from dataclasses import replace

from manialab.audio import fuse_attacks
from manialab.domain import Note


def pair():
    return [Note(100, 500, pitch=36, voice='bass', source='pitch', id='first',
                 release_reason='same_pitch_reattack'),
            Note(500, 900, pitch=36, voice='bass', source='pitch', id='next',
                 release_reason='frame_decay')]


def test_earlier_anchored_reattack_also_bounds_previous_interval():
    original = pair()
    notes, _, _ = fuse_attacks(original, [Note(475, 475)])
    assert notes[0] == replace(original[0], end=475)
    assert notes[1] == replace(original[1], start=475)
    assert original[0].end == 500


def test_later_anchored_reattack_cannot_invent_more_sustain():
    original = pair()
    notes, _, _ = fuse_attacks(original, [Note(525, 525)])
    assert notes[0] == original[0] and notes[1].start == 525


def test_independent_decay_or_unknown_release_is_not_changed_by_next_attack():
    for reason in ('frame_decay', 'input_boundary', 'unresolved_or_postprocessed'):
        original = pair()
        original[0] = replace(original[0], release_reason=reason)
        notes, _, _ = fuse_attacks(original, [Note(475, 475)])
        assert notes[0] == original[0] and notes[1].start == 475


def test_other_voice_or_model_instrument_does_not_bound_a_note():
    for field, value in (('voice', 'guitar'), ('instrument_id', 'different:class')):
        original = pair()
        original[1] = replace(original[1], **{field: value})
        notes, _, _ = fuse_attacks(original, [Note(475, 475)])
        assert notes[0] == original[0] and notes[1].start == 475


def test_one_transient_cannot_collapse_multiple_same_pitch_attacks():
    original = [Note(at, at+35, pitch=36, voice='bass', source='pitch', id=str(at))
                for at in (100, 140, 180)]
    notes, unmatched, used = fuse_attacks(original, [Note(140, 140)])
    assert notes == original
    # The same transient already supports these candidates; do not add it again.
    assert unmatched == [] and used == 1


def test_chord_can_share_an_attack_without_same_pitch_collision():
    original = [Note(120, 300, pitch=36), Note(150, 400, pitch=40)]
    notes, _, _ = fuse_attacks(original, [Note(140, 140)])
    assert [n.start for n in notes] == [140, 140]
