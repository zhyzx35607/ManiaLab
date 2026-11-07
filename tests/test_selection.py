from collections import Counter

from manialab.arrange import select_notes
from manialab.domain import Note


def test_selection_can_reserve_room_for_a_later_strong_attack():
    notes = [Note(0, 0, strength=.25, source='transient', id='weak'),
             Note(400, 400, strength=.95, source='transient', id='strong')]
    selected, report = select_notes(notes, {'density': 1})
    assert [n.id for n in selected] == ['strong']
    assert report['attack_optimal'] and report['omitted'] == 1


def test_selection_respects_sliding_density_and_chord_limits():
    notes = [Note(t, t + 600, pitch=48 + pitch, strength=.4 + pitch * .1,
                  source='pitch', id=f'{t}:{pitch}')
             for t in range(0, 5000, 125) for pitch in range(4)]
    original = [n.data() for n in notes]
    selected, report = select_notes(notes, {'density': 7, 'chord_size': 3})
    assert max(Counter(n.start for n in selected).values()) <= 3
    for n in selected:
        assert sum(n.start - 1000 < other.start <= n.start for other in selected) <= 7
    assert [n.data() for n in notes] == original
    assert {n.id for n in selected} <= {n.id for n in notes}


def test_phrase_preference_cannot_invent_sustain_or_leak_into_tap_label():
    notes = [Note(0, 1500, pitch=60, source='pitch', strength=.6, id='v', voice='vocals',
                  phrase=True, support=('v','b'), support_coverage=.95),
             Note(0, 200, pitch=65, source='pitch', strength=.7, id='o', voice='other')]
    selected, _ = select_notes(notes, {'chord_size': 1, 'ln_ratio': 1})
    assert len(selected) == 1 and selected[0].id == 'v' and selected[0].end == 1500
    taps, _ = select_notes(notes, {'chord_size': 1, 'ln_ratio': 0})
    assert len(taps) == 1 and not taps[0].hold and not taps[0].phrase


def test_duplicate_pitch_is_deduplicated_within_voice_only():
    notes = [Note(1000, 2000, pitch=60, source='pitch', voice=v, id=str(i), strength=s)
             for i, (v, s) in enumerate([('vocals',.8),('vocals',.5),('other',.7)])]
    selected, report = select_notes(notes, {'ln_ratio': 1})
    assert {n.id for n in selected} == {'0','2'}
    assert report['omitted'] == 1


def test_selection_does_not_hide_large_attack_errors_by_snapping():
    notes = [Note(1000,1000,source='transient',id='late',strength=.95,evidence_start=1060),
             Note(1125,1125,source='transient',id='near',strength=.8,evidence_start=1127)]
    selected, report = select_notes(notes, {'density':1})
    assert [n.id for n in selected] == ['near']
    assert selected[0].evidence_start == 1127
    assert report['selected_large_attack_shifts'] == 0


def test_distinct_fine_grid_ticks_are_not_misclassified_as_one_chord():
    notes=[Note(1000,1000,source='transient',id='first'),Note(1017,1017,source='transient',id='second')]
    selected,_=select_notes(notes,{'chord_size':1,'density':4})
    assert [n.id for n in selected]==['first','second']


def test_voice_controls_change_selection_without_making_new_notes():
    notes=[Note(1000,1800,pitch=60,source='pitch',voice='vocals',strength=.5,id='v'),
           Note(1000,1000,source='transient',voice='drums',strength=.8,id='d')]
    selected,report=select_notes(notes,{'drums_weight':0,'ln_ratio':1})
    assert [n.id for n in selected]==['v'] and selected[0].end==1800
    assert report['muted_voice_candidates']==1
    selected,_=select_notes(notes,{'vocals_weight':2,'chord_size':1})
    assert [n.id for n in selected]==['v']


def test_source_weights_can_follow_segment_preferences():
    notes=[Note(t,t,source='transient',voice='drums',id=str(t)) for t in (1000,3000)]
    selected,_=select_notes(notes,segment_preferences=[dict(start_ms=0,end_ms=2000,preferences={'drums_weight':0})])
    assert [n.start for n in selected]==[3000]
