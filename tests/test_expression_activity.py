import copy

import pytest

from manialab.arrange import parameters
from manialab.domain import Note, UNOBSERVED_RELEASES
from manialab.expression import expression_evidence, propose_expression, validate_expression
from manialab.expression_activity import section_activity
from tests.test_structure_expression import analysis


def note(start, end, voice, identity, **kwargs):
    return Note(start, end, source='pitch', voice=voice, id=identity, **kwargs)


def test_long_background_does_not_claim_priority_over_short_foreground_attacks():
    rows = [note(0, 12000, 'other', 'pad')]
    rows += [note(t, t+500, 'piano', str(t)) for t in (1000, 2000, 7000, 8000)]
    a = analysis(rows)
    a['structure']['segments'] = [dict(start_ms=0, end_ms=6000), dict(start_ms=6000, end_ms=12000)]
    original = copy.deepcopy(a)
    p = parameters()
    plan = propose_expression(a, p)
    assert all(s['main_voice'] is None for s in plan['sections'])
    assert all(s['voices']['other']['role'] == 'support' for s in plan['sections'])
    assert plan['sections'][1]['voices']['other']['activity']['carried_candidate_count'] == 1
    result, _, _ = expression_evidence(a, plan, p)
    assert [(n.id, n.start, n.end) for n in result] == [(n.id, n.start, n.end) for n in rows]
    assert a == original


def test_a_short_instrument_fill_survives_a_long_section_and_repeated_sections():
    rows = [note(t, t+200, 'guitar', str(t)) for t in (1000, 61000)]
    a = analysis(rows, length=120000)
    a['structure']['segments'] = [dict(start_ms=0, end_ms=60000), dict(start_ms=60000, end_ms=120000)]
    p = parameters()
    plan = propose_expression(a, p)
    result, _, _ = expression_evidence(a, plan, p)
    assert [n.id for n in result] == ['1000', '61000']
    assert [s['voices']['guitar']['mode'] for s in plan['sections']] == ['articulation', 'articulation']
    assert [s['main_voice'] for s in plan['sections']] == ['guitar', 'guitar']


@pytest.mark.parametrize('reason', sorted(UNOBSERVED_RELEASES))
def test_unobserved_tail_does_not_keep_a_source_alive_in_a_later_solo(reason):
    rows = [note(0, 12000, 'other', 'unknown', release_reason=reason),
            note(7000, 8500, 'bass', 'solo')]
    a = analysis(rows)
    a['structure']['segments'] = [dict(start_ms=0, end_ms=6000), dict(start_ms=6000, end_ms=12000)]
    p = parameters()
    plan = propose_expression(a, p)
    assert plan['sections'][1]['main_voice'] == 'bass'
    assert plan['sections'][1]['voices']['other']['mode'] == 'omit'
    assert plan['sections'][0]['voices']['other']['activity']['uncertain_tails'] == 1
    result, _, _ = expression_evidence(a, plan, p)
    assert [(n.id, n.hold) for n in result] == [('unknown', False), ('solo', True)]


def test_activity_is_exact_clipped_and_counts_chords_without_chaining():
    rows = [note(0, 2500, 'other', 'pad', instrument_id='strings'),
            *[note(t, 2300, 'other', str(t), instrument_id='strings') for t in (1990, 2015, 2040)]]
    result = section_activity(rows, 1980, 2080)
    assert result['local_candidate_count'] == 3 and result['local_attack_groups'] == 2
    assert result['carried_candidate_count'] == 1
    assert result['carried_instrument_ids'] == ['strings']
    assert result['observed_interval_coverage'] == result['activity_coverage'] == 1
    assert result['attack_coverage'] == .9


def test_manual_main_voice_and_saved_long_short_expression_remain_authoritative():
    a = analysis([note(1000, 2000, 'other', 'pad'), note(1000, 2000, 'piano', 'piano')])
    p = parameters()
    plan = propose_expression(a, p)
    plan.update(source='manual', revision=2)
    section = plan['sections'][0]
    section['main_voice'] = 'piano'
    section['voices']['piano'].update(role='primary', mode='onsets')
    section.pop('priority_evidence')
    assert validate_expression(plan, a) == plan
    for ratio in (0, .55, 1):
        result, metadata, _ = expression_evidence(a, plan, parameters({'ln_ratio':ratio}))
        assert [(n.id, n.hold) for n in result] == [('pad', True), ('piano', False)]
        assert metadata['piano']['role'] == 'primary'


def test_muting_a_source_and_silence_do_not_create_a_phantom_primary():
    a = analysis([note(1000, 2000, 'piano', 'piano'), note(1000, 2000, 'other', 'other')])
    a['structure']['segments'] = [dict(start_ms=0, end_ms=6000), dict(start_ms=6000, end_ms=12000)]
    plan = propose_expression(a, parameters({'other_weight':0}))
    assert [s['main_voice'] for s in plan['sections']] == ['piano', None]
    assert plan['sections'][0]['voices']['other']['mode'] == 'omit'
    assert all(p['mode'] == 'omit' for p in plan['sections'][1]['voices'].values())


def test_carry_only_category_does_not_create_an_override_that_rewrites_an_earlier_onset():
    a = analysis([note(1000, 9000, 'other', 'strings', instrument_id='strings')])
    a['structure']['segments'] = [dict(start_ms=0, end_ms=6000), dict(start_ms=6000, end_ms=12000)]
    p = parameters()
    plan = propose_expression(a, p)
    assert plan['sections'][1]['voices']['other']['activity']['carried_instrument_ids'] == ['strings']
    assert plan['sections'][1]['instrument_modes'] == {}
    plan['sections'][1]['voices']['other']['mode'] = 'omit'
    result, _, _ = expression_evidence(a, plan, p)
    assert [(n.start, n.end) for n in result] == [(1000, 9000)]


@pytest.mark.parametrize('ratio', [0, .55, .75, 1])
def test_default_expression_depends_on_tendency_not_a_main_voice_slot(ratio):
    notes = [note(t, t+1200, voice, f'{voice}:{t}')
             for voice in ('piano', 'guitar', 'other', 'bass') for t in (1000, 1250, 1500)]
    a = analysis(notes)
    p = parameters({'ln_ratio':ratio})
    plan = propose_expression(a, p)
    assert plan['sections'][0]['main_voice'] is None
    result, _, _ = expression_evidence(a, plan, p)
    for voice in ('piano', 'guitar', 'other', 'bass'):
        holds = [n.hold for n in result if n.voice == voice]
        expected = [False, False, False] if ratio == 0 else [False, False, True] if ratio < .75 and voice in ('piano', 'guitar') else [True, True, True]
        assert holds == expected


def test_automatic_expression_uses_local_sustain_tendency_and_minimum_length():
    rows = [note(1000, 1600, 'piano', 'below-local-minimum'),
            note(5000, 5600, 'piano', 'above-local-minimum'),
            note(9000, 9600, 'piano', 'locally-tap-only')]
    a = analysis(rows)
    a['structure']['segments'] = [dict(start_ms=0, end_ms=4000),
                                  dict(start_ms=4000, end_ms=8000),
                                  dict(start_ms=8000, end_ms=12000)]
    settings = parameters({'ln_ratio': .55, 'min_ln': 240})
    segments = [
        dict(start_ms=0, end_ms=4000, preferences=dict(ln_ratio=.9, min_ln=800)),
        dict(start_ms=4000, end_ms=8000, preferences=dict(ln_ratio=.9, min_ln=500)),
        dict(start_ms=8000, end_ms=12000, preferences=dict(ln_ratio=0, min_ln=100)),
    ]

    plan = propose_expression(a, settings, segment_preferences=segments)
    evidence, _, _ = expression_evidence(a, plan, settings, segment_preferences=segments)

    assert [section['voices']['piano']['mode'] for section in plan['sections']] == [
        'sustain', 'sustain', 'onsets']
    assert [(n.id, n.hold) for n in evidence] == [
        ('below-local-minimum', False),
        ('above-local-minimum', True),
        ('locally-tap-only', False),
    ]


def test_automatic_expression_respects_a_local_voice_mute():
    rows = [note(1000, 1600, 'piano', 'muted'),
            note(7000, 7600, 'piano', 'active')]
    a = analysis(rows)
    a['structure']['segments'] = [dict(start_ms=0, end_ms=6000),
                                  dict(start_ms=6000, end_ms=12000)]
    settings = parameters({'ln_ratio': .9})
    segments = [dict(start_ms=0, end_ms=6000, preferences=dict(piano_weight=0)),
                dict(start_ms=6000, end_ms=12000, preferences=dict(piano_weight=1))]

    plan = propose_expression(a, settings, segment_preferences=segments)
    evidence, _, decisions = expression_evidence(
        a, plan, settings, segment_preferences=segments)

    assert [section['voices']['piano']['mode'] for section in plan['sections']] == [
        'omit', 'sustain']
    assert [(n.id, n.hold) for n in evidence] == [('active', True)]
    assert decisions['muted']['reason'] == 'voice_not_selected'


def test_automatic_vocal_phrases_use_each_sections_minimum_length():
    rows = [note(1000, 1100, 'vocals', 'short-anchor'),
            note(5000, 5100, 'vocals', 'long-anchor')]
    a = analysis(rows)
    a['structure']['segments'] = [dict(start_ms=0, end_ms=4000),
                                  dict(start_ms=4000, end_ms=8000),
                                  dict(start_ms=8000, end_ms=12000)]
    a['phrase_candidates'] = [
        dict(rows[0].data(), end=1700, phrase=True, support=['short-anchor'],
             support_coverage=1., release_reason='vocal_activity_decay'),
        dict(rows[1].data(), end=6000, phrase=True, support=['long-anchor'],
             support_coverage=1., release_reason='vocal_activity_decay'),
    ]
    settings = parameters({'ln_ratio': .9, 'phrase_ln': 1, 'min_ln': 240})
    segments = [
        dict(start_ms=0, end_ms=4000, preferences=dict(min_ln=800)),
        dict(start_ms=4000, end_ms=8000, preferences=dict(min_ln=500)),
    ]

    plan = propose_expression(a, settings, segment_preferences=segments)
    evidence, _, trace = expression_evidence(a, plan, settings, segment_preferences=segments)

    assert [section['voices']['vocals']['mode'] for section in plan['sections']] == [
        'sustain', 'phrase', 'omit']
    assert [(n.id, n.end) for n in evidence] == [('short-anchor', 1000), ('long-anchor', 6000)]
    assert trace['short-anchor']['reason'] == 'short_duration'
