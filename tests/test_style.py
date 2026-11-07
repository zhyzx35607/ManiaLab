import pytest

from manialab.domain import Note
from manialab.style import describe_chart


def test_reference_preferences_keep_input_and_reflect_its_sustains():
    notes = [Note(t,t+500 if i%2 else t,i%10,id=str(i)) for i,t in enumerate(range(0,10000,250))]
    before = [n.data() for n in notes]
    profile = describe_chart(dict(keys=10,notes=notes,metadata={'Title':'Example','Version':'10K'}))
    assert profile['preferences'] == dict(density=4,ln_ratio=.5,min_ln=500,chord_size=1,max_held=1)
    assert profile['ln_duration_ms']['50'] == 500
    assert [n.data() for n in notes] == before
    assert 'keys' not in profile['preferences']


def test_reference_conversion_is_visible_and_placeholder_is_rejected():
    notes = [Note(i*500,i*500,i%6) for i in range(20)]
    profile = describe_chart(dict(keys=6,notes=notes,metadata={'Title':'Inverse Pack'}))
    assert profile['warnings'] and profile['ln_ratio'] == 0
    with pytest.raises(ValueError,match='20'):
        describe_chart(dict(keys=6,notes=notes[:3],metadata={}))
