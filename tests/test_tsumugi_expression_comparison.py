import copy

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import propose_expression, validate_expression


def analysis(notes):
    return dict(duration=10., candidates=[n.data() for n in notes], phrase_candidates=[],
        structure={'segments': [{'start_ms': 0, 'end_ms': 10000, 'label': '段1',
                                 'similarity_group': 'S1'}]})


def test_expression_plan_uses_current_instrument_ids_and_does_not_mutate_analysis():
    previous = analysis([
        Note(100, 900, id='old', voice='other', strength=.8, instrument_id='old-class'),
    ])
    current = analysis([
        Note(100, 900, id='new', voice='other', strength=1., instrument_id='new-class'),
        Note(200, 700, id='guitar', voice='guitar', strength=1., instrument_id='guitar-class'),
    ])
    settings = parameters({'keys': 10, 'ln_ratio': .55})
    previous_plan = propose_expression(previous, settings)
    original_current = copy.deepcopy(current)

    current_plan = propose_expression(current, settings)
    validate_expression(current_plan, current)

    assert current == original_current
    assert previous_plan['sections'][0]['instrument_modes']['other'] == {'old-class': 'inherit'}
    assert current_plan['sections'][0]['instrument_modes']['other'] == {'new-class': 'inherit'}
    assert current_plan['sections'][0]['instrument_modes']['guitar'] == {'guitar-class': 'inherit'}
    assert current_plan['sections'][0]['voices']['guitar']['role'] == 'support'
