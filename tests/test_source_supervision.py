import pytest

from manialab.domain import Note, parse_osu, sections
from scripts.prepare_sakura_training import source_chart
from scripts.probe_sakura_sources import grouped_onsets
from scripts.research_piano_layers import snap_function
from scripts.research_source_attack_classes import drum_event_counts


CHART = '''osu file format v14
[General]
AudioFilename: example.mp3
Mode: 3
[Editor]
Bookmarks: 1000,2000
[Metadata]
Title: synthetic
Version: Reference
[Difficulty]
CircleSize: 10
[TimingPoints]
100,500,4,2,0,70,1,0
2100,400,4,2,0,70,1,0
[HitObjects]
25,192,1000,1,0,0:0:0:0:
76,192,1500,1,0,0:0:0:0:
128,192,1500,1,0,0:0:0:0:
179,192,2000,1,0,0:0:0:0:
230,192,1000,1,0,0:0:0:0:
281,192,1000,1,0,0:0:0:0:
332,192,1000,128,0,2000:0:0:0:0:
'''


def test_source_labels_preserve_snare_multiplicity_piano_taps_and_other_ln():
    parts = {s: parse_osu(source_chart(CHART, s)) for s in ('drums', 'piano', 'other')}
    drums, piano, other = [parts[s]['notes'] for s in ('drums', 'piano', 'other')]
    assert [(n.start, n.lane) for n in drums] == [(1000, 0), (1500, 1), (1500, 2), (2000, 3)]
    assert [(n.start, n.end, n.lane) for n in piano] == [(1000, 1000, 4), (1000, 1000, 5)]
    assert [(n.start, n.end, n.lane) for n in other] == [(1000, 2000, 6)]
    for voice in parts:
        text = source_chart(CHART, voice)
        assert parts[voice]['timing'] == parse_osu(CHART)['timing']
        assert parts[voice]['general']['AudioFilename'] == 'audio.wav'
        assert not any(line.strip() for line in sections(text)['Editor'])
    counts=drum_event_counts(parse_osu(CHART)['notes'])
    assert counts=={(1000,'kick'):1,(1500,'snare'):2,(2000,'hi-hat'):1}


def test_source_supervision_rejects_unknown_column_semantics():
    with pytest.raises(ValueError):
        source_chart(CHART.replace('CircleSize: 10', 'CircleSize: 7'), 'piano')
    with pytest.raises(ValueError):
        source_chart(CHART, 'vocals')


def test_grouping_does_not_chain_distinct_attacks_or_become_layer_count():
    notes = [Note(t, t) for t in (100, 120, 140, 140)]
    assert grouped_onsets(notes, tolerance=30) == [(110.,), (140.,)]
    assert grouped_onsets(notes, tolerance=0) == [(100.,), (120.,), (140.,)]


def test_known_timing_diagnostic_uses_each_red_line():
    snap = snap_function(CHART)
    assert snap(1229) == 1225
    assert snap(2319) == 2300


def test_expression_sketch_sanitizing_keeps_notes_without_target_structure():
    from scripts.prepare_sakura_expression import prepare_chart
    original=CHART.replace('100,500,4,2,0,70,1,0','100,500,4,2,0,70,1,1')
    prepared=prepare_chart(original,clear_structure=True)
    before,after=parse_osu(original),parse_osu(prepared)
    assert [(n.start,n.end,n.lane) for n in before['notes']]==[(n.start,n.end,n.lane) for n in after['notes']]
    assert '100,500,4,2,0,70,1,0' in prepared
    assert not any(x.strip() for x in sections(prepared)['Editor'])


def test_source_scores_do_not_reward_correct_timing_in_unrelated_columns():
    from scripts.probe_mapper_sources import source_scores
    expected=[Note(1000,1000,4),Note(1000,1000,5)]
    actual=[Note(1000,1000,0),Note(1000,1000,1)]
    result=source_scores(expected,actual,'piano',0,2000)
    assert result['onset_multiplicity']['f1']==1
    assert result['note_column_match']['f1']==0
    assert result['semantic_lane_fraction']==0


def test_expression_study_preserves_ambiguous_sources_and_unmatched_endpoints():
    from scripts.study_sakura_expression import describe
    ref=[Note(1000,1000,0),Note(1000,1000,4),Note(1500,1500,4)]
    finished=[Note(1000,1500,9),Note(2000,2500,8)]
    report=describe(ref,finished)
    assert report['ln_head_candidate_sources']=={'drums+piano':1,'unmatched':1}
    assert report['both_endpoints_near_reference_attacks']==1
    assert report['ln_tails_near_reference_attacks']==1


def test_sketch_codec_keeps_pretrained_event_ids_and_rejects_incompatible_native_gd():
    from assets.mapper_sketch_context import tokenizer_with_sketch
    state=dict(context_sos={'timing':3,'map':5},context_eos={'timing':4,'map':6},
        event_start={'column':100},vocab_size_in=900,vocab_size_out=800)
    updated=tokenizer_with_sketch(state)
    assert updated['context_sos']['gd']==5 and updated['context_eos']['gd']==6
    assert updated['event_start']==state['event_start']
    assert updated['vocab_size_in']==900 and updated['vocab_size_out']==800
    assert 'gd' not in state['context_sos']
    with pytest.raises(ValueError,match='different GD'):
        tokenizer_with_sketch(state|{'context_sos':{'map':5,'gd':10}})
