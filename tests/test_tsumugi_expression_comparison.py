import copy

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import propose_expression, validate_expression
from scripts.check_tsumugi_expression import comparison_analysis, frozen_expression


def analysis(notes):
    return dict(duration=10.,candidates=[n.data() for n in notes],phrase_candidates=[],
        structure={'segments':[{'start_ms':0,'end_ms':10000,'label':'段1','similarity_group':'S1'}]})


def test_unit_weight_ablation_is_symmetrical_without_changing_source_evidence():
    original = analysis([Note(100,500,id='a',voice='other',strength=.2),
                         Note(200,800,id='b',voice='piano',strength=.6)])
    before = copy.deepcopy(original)
    changed = comparison_analysis(original,['other'],True)
    assert [n['strength'] for n in changed['candidates']] == [1.,.6]
    assert original == before
    for old,new in zip(original['candidates'],changed['candidates']):
        assert {k:v for k,v in old.items() if k!='strength'} == {k:v for k,v in new.items() if k!='strength'}


def test_frozen_voice_modes_use_current_ids_and_explicit_absent_voice_policy():
    baseline = analysis([Note(100,900,id='a',voice='other',strength=.8,instrument_id='old-class')])
    current = analysis([Note(100,900,id='b',voice='other',strength=1.,instrument_id='new-class'),
                        Note(200,700,id='c',voice='guitar',strength=1.,instrument_id='guitar-class')])
    settings = parameters({'keys':10,'ln_ratio':.55})
    reference = propose_expression(baseline,settings)
    before = copy.deepcopy(reference)
    controlled = frozen_expression(current,settings,reference)
    validate_expression(controlled,current)
    section = controlled['sections'][0]
    assert section['voices']['other'] == reference['sections'][0]['voices']['other']
    assert section['voices']['guitar'] == dict(mode='omit',role='support')
    assert section['instrument_modes']['other'] == {'new-class':'inherit'}
    assert reference == before
