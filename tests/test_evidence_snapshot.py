"""Keep physical detection, expression and edited chart times distinguishable."""
from copy import deepcopy
from dataclasses import replace
import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.generation import generate_variant
from manialab.timing import align_candidates


def test_collapsed_phrase_does_not_claim_to_cover_interior_notes():
    notes = [Note(1100,1200,id='a',pitch=60,source='pitch',voice='vocals',release_reason='frame_decay'),
             Note(1400,1500,id='b',pitch=64,source='pitch',voice='vocals',release_reason='frame_decay')]
    phrase = replace(notes[0],end=1800,phrase=True,support=('a','b'),support_coverage=1,
                     release_reason='vocal_activity_decay')
    analysis = dict(revision=13,duration=5,candidates=[n.data() for n in notes],
        phrase_candidates=[phrase.data()],timing_attacks=[],
        structure=dict(segments=[dict(start_ms=0,end_ms=5000)]))
    state = dict(analysis=analysis,variants=[],active=-1)
    generate_variant(state,dict(mode='audio',timing_confirmed=True),parameters(dict(phrase_ln=1)),
        dict(bpm=10,offset=0,division=3,audio_shift_ms=0),lambda *_:None,lambda:False)
    variant = state['variants'][-1]
    assert [(n['id'],n['start'],n['end'],n['phrase']) for n in variant['notes']] == [('a',2000,2000,False)]
    decisions = {d['id']:d for d in variant['selection']['decisions']}
    assert decisions['a']['reason'] == 'duration_collapsed_on_grid'
    assert decisions['b']['reason'] == 'phrase_not_selected'
    snapshot = variant['source_evidence']['a']
    assert snapshot['detected']['end'] == 1200
    assert snapshot['expressed']['end'] == 1800 and snapshot['expressed']['phrase']
    assert snapshot['expressed']['release_reason'] == 'vocal_activity_decay'


def test_quantization_clears_only_collapsed_phrase_flag_and_keeps_support():
    phrase = Note(1100,1800,source='pitch',phrase=True,support=('a','b'),release_reason='vocal_activity_decay')
    collapsed,_ = align_candidates([phrase],dict(bpm=10,offset=0,division=3),confirmed=True)
    assert not collapsed[0].phrase and collapsed[0].support == phrase.support
    retained,_ = align_candidates([phrase],dict(bpm=120,offset=0,division=4),confirmed=True)
    assert retained[0].phrase and retained[0].hold
    assert phrase.phrase and phrase.end == 1800


def test_generation_saves_detection_and_expression_before_shift_and_quantization():
    raw = Note(1115,1865,id='tone',source='pitch',voice='piano',release_reason='frame_decay')
    analysis = dict(revision=13,duration=5,candidates=[raw.data()],phrase_candidates=[],timing_attacks=[],
        structure=dict(segments=[dict(start_ms=0,end_ms=5000)]))
    state = dict(analysis=deepcopy(analysis),variants=[],active=-1)
    generate_variant(state,dict(mode='audio',timing_confirmed=True),parameters(dict(ln_ratio=0)),
        dict(bpm=120,offset=93,division=4,audio_shift_ms=22.5),lambda *_:None,lambda:False)
    variant = state['variants'][-1]
    snapshot = variant['source_evidence']['tone']
    assert (snapshot['detected']['start'],snapshot['detected']['end']) == (1115,1865)
    assert (snapshot['expressed']['start'],snapshot['expressed']['end']) == (1115,1115)
    assert snapshot['detected']['release_reason'] == 'frame_decay'
    assert variant['notes'][0]['evidence_end'] == 1093
    state['analysis']['candidates'][0]['end'] = 4900
    assert snapshot['detected']['end'] == 1865
    assert analysis['candidates'][0]['end'] == 1865


def test_locked_region_sources_are_not_replaced_by_a_new_analysis_with_reused_ids():
    from manialab.evidence import capture_source_evidence
    old = {'locked':dict(detected={'start':100,'end':900},expressed={'start':100,'end':900})}
    current = Note(100,400,id='locked',source='pitch')
    fresh = Note(2000,2600,id='fresh',source='pitch')
    legacy = Note(3000,3300,id='legacy',source='pitch')
    analysis = dict(candidates=[n.data() for n in (current,fresh,legacy)])
    result = capture_source_evidence(analysis,[current,fresh,legacy],[current,fresh,legacy],
        locked=[current,legacy],previous=old)
    assert result['locked'] == old['locked'] and result['fresh']['detected']['end'] == 2600
    assert 'legacy' not in result
    result['locked']['detected']['end'] = 500
    assert old['locked']['detected']['end'] == 900


def test_regional_generation_preserves_edit_provenance_only_for_locked_notes():
    from manialab.editing import edit_notes
    raw = [Note(1000,1800,id='locked',source='pitch'), Note(3500,4200,id='rewritten',source='pitch')]
    state = dict(analysis=dict(revision=13,duration=5,candidates=[n.data() for n in raw],
        phrase_candidates=[],timing_attacks=[]),variants=[],active=-1)
    prefs = parameters(dict(ln_ratio=1))
    tempo = dict(bpm=120,offset=0,division=4,audio_shift_ms=0)
    generate_variant(state,dict(mode='audio',timing_confirmed=True),prefs,tempo,lambda *_:None,lambda:False)
    old = state['variants'][-1]
    for identity,end in [('locked',2500),('rewritten',4500)]:
        notes,edit = edit_notes(old,dict(op='update',note_id=identity,values=dict(end=end)),5000)
        old['notes'] = [n.data() for n in notes]
        old.setdefault('edits',[]).append(edit)
    before = deepcopy(old)
    generate_variant(state,dict(mode='audio',timing_confirmed=True,region=[3000,5000]),
        prefs,tempo,lambda *_:None,lambda:False)
    new = state['variants'][-1]
    assert next(n for n in new['notes'] if n['id']=='locked')['end'] == 2500
    assert new['edits'] == [before['edits'][0]]
    assert new['source_evidence']['locked'] == before['source_evidence']['locked']
    assert old == before


def test_regional_generation_rejects_reused_id_that_moved_across_locked_boundary():
    raw = Note(1000,1500,id='piano:p0',source='pitch',voice='piano')
    state = dict(analysis=dict(revision=13,duration=5,candidates=[raw.data()],
        phrase_candidates=[],timing_attacks=[]),variants=[],active=-1)
    prefs = parameters(dict(ln_ratio=1))
    tempo = dict(bpm=120,offset=0,division=4,audio_shift_ms=0)
    generate_variant(state,dict(mode='audio',timing_confirmed=True),prefs,tempo,lambda *_:None,lambda:False)
    previous = deepcopy(state['variants'])
    state['analysis']['candidates'] = [replace(raw,start=3500,end=4000).data()]
    with pytest.raises(ValueError,match='标识.*整谱'):
        generate_variant(state,dict(mode='audio',timing_confirmed=True,region=[3000,5000]),
            prefs,tempo,lambda *_:None,lambda:False)
    assert state['variants'] == previous and state['active'] == 0


def test_regional_generation_rejects_known_audio_source_change():
    raw = [Note(1000,1800,id='locked',source='pitch',voice='piano'),
           Note(3500,4200,id='regional',source='pitch',voice='bass')]
    analysis = dict(revision=13,duration=5,audio_sha256='audio-A',
        candidates=[n.data() for n in raw],phrase_candidates=[],timing_attacks=[])
    state = dict(analysis=analysis,variants=[],active=-1)
    prefs = parameters(dict(ln_ratio=1))
    tempo = dict(bpm=120,offset=0,division=4,audio_shift_ms=0)
    generate_variant(state,dict(mode='audio',timing_confirmed=True),prefs,tempo,
        lambda *_:None,lambda:False)
    assert state['variants'][0]['source_audio_sha256'] == 'audio-A'
    previous = deepcopy(state['variants'])

    analysis['audio_sha256'] = 'audio-B'
    with pytest.raises(ValueError,match='不能混用不同或未知的声轨来源'):
        generate_variant(state,dict(mode='audio',timing_confirmed=True,region=[3000,5000]),
            prefs,tempo,lambda *_:None,lambda:False)

    assert state['variants'] == previous and state['active'] == 0


def test_regional_generation_rejects_unknown_parent_when_current_audio_identity_is_known():
    raw = [Note(1000,1800,id='locked',source='pitch',voice='piano'),
           Note(3500,4200,id='regional',source='pitch',voice='bass')]
    analysis = dict(revision=13,duration=5,candidates=[n.data() for n in raw],
        phrase_candidates=[],timing_attacks=[])
    state = dict(analysis=analysis,variants=[],active=-1)
    prefs = parameters(dict(ln_ratio=1))
    tempo = dict(bpm=120,offset=0,division=4,audio_shift_ms=0)
    generate_variant(state,dict(mode='audio',timing_confirmed=True),prefs,tempo,
        lambda *_:None,lambda:False)
    assert state['variants'][0]['source_audio_sha256'] is None

    analysis['audio_sha256'] = 'new-current-analysis'
    before_rejection = deepcopy(state)
    with pytest.raises(ValueError,match='不能混用不同或未知的声轨来源'):
        generate_variant(state,dict(mode='audio',timing_confirmed=True,region=[3000,5000]),
            prefs,tempo,lambda *_:None,lambda:False)
    assert state == before_rejection
    assert state['variants'] == before_rejection['variants']

    del analysis['audio_sha256']
    generate_variant(state,dict(mode='audio',timing_confirmed=True,region=[3000,5000]),
        prefs,tempo,lambda *_:None,lambda:False)
    assert state['variants'][0]['source_audio_sha256'] is None
    assert state['variants'][1]['source_audio_sha256'] is None


def test_regional_generation_rejects_unknown_current_identity_for_known_parent():
    raw = [Note(1000,1800,id='locked',source='pitch',voice='piano'),
           Note(3500,4200,id='regional',source='pitch',voice='bass')]
    analysis = dict(revision=13,duration=5,audio_sha256='audio-A',
        candidates=[n.data() for n in raw],phrase_candidates=[],timing_attacks=[])
    state = dict(analysis=analysis,variants=[],active=-1)
    prefs = parameters(dict(ln_ratio=1))
    tempo = dict(bpm=120,offset=0,division=4,audio_shift_ms=0)
    generate_variant(state,dict(mode='audio',timing_confirmed=True),prefs,tempo,
        lambda *_:None,lambda:False)
    assert state['variants'][0]['source_audio_sha256'] == 'audio-A'

    del analysis['audio_sha256']
    before_rejection = deepcopy(state)
    with pytest.raises(ValueError,match='不能混用不同或未知的声轨来源'):
        generate_variant(state,dict(mode='audio',timing_confirmed=True,region=[3000,5000]),
            prefs,tempo,lambda *_:None,lambda:False)

    assert state == before_rejection


def test_regional_generation_rejects_changed_separation_source():
    from copy import deepcopy
    from manialab.stems import separation_identity

    raw = Note(1000, 1800, id='piano:source', source='pitch', voice='piano')
    source_a = dict(source='provided', audio_sha256='audio', set_sha256='a'*64,
        voices=['piano'])
    source_b = source_a | {'set_sha256':'b'*64}
    state = dict(analysis=dict(revision=13, duration=5, candidates=[raw.data()],
        phrase_candidates=[], timing_attacks=[], separation=source_a), variants=[], active=-1)
    prefs = parameters(dict(ln_ratio=1))
    tempo = dict(bpm=120, offset=0, division=4, audio_shift_ms=0)
    generate_variant(state, dict(mode='audio', timing_confirmed=True), prefs, tempo,
        lambda *_:None, lambda:False)
    previous = deepcopy(state['variants'])
    state['analysis']['separation'] = source_b

    with pytest.raises(ValueError, match='不能混用不同或未知的声轨来源'):
        generate_variant(state, dict(mode='audio', timing_confirmed=True, region=[3000, 5000]),
            prefs, tempo, lambda *_:None, lambda:False)

    reference = state['variants'][0]['source_separation']
    assert reference['identity'] == separation_identity(source_a)
    assert reference['separation'] == source_a
    assert state['variants'][0]['source_audio_sha256'] == source_a['audio_sha256']
    assert state['variants'] == previous and state['active'] == 0
