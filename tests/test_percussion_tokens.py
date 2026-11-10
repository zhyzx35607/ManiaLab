import numpy as np
import pytest
from manialab.percussion_tokens import legal_next,decode_complete,audit_tokens


def test_exact_types_and_physical_time_bounds():
    assert legal_next([2])[4] and legal_next([2])[259] and not legal_next([2])[260]
    assert legal_next([2],max_time_index=9)[13] and not legal_next([2],max_time_index=9)[14]
    assert np.flatnonzero(legal_next([2,10])).tolist()==list(range(335,361))
    assert np.flatnonzero(legal_next([2,10,342])).tolist()==list(range(400,528))


def test_simultaneous_different_drums_and_real_reattacks_remain_legal():
    prefix=[2,10,336,500]
    assert legal_next(prefix)[10] and legal_next(prefix)[11] and not legal_next(prefix)[9]
    assert not legal_next(prefix+[10])[336] and legal_next(prefix+[10])[342]
    assert legal_next(prefix+[11])[336]
    assert decode_complete(prefix+[10,342,490,11,336,480,3])==[
        dict(onset=.06,custom_pitch=36,velocity=100),dict(onset=.06,custom_pitch=42,velocity=90),
        dict(onset=.07,custom_pitch=36,velocity=80)]


def test_end_and_corrupt_triples():
    assert np.flatnonzero(legal_next([2,0])).tolist()==[3]
    assert decode_complete([2,0,3])==[]
    with pytest.raises(ValueError):decode_complete([2,10,336,500,20,3])
    with pytest.raises(ValueError):legal_next([2,10,336,500,10,336,500])
    with pytest.raises(ValueError):legal_next([2,10,336,500,9])


def test_audit_does_not_hide_duplicates_or_events_after_eos():
    assert audit_tokens([2,10,336,500,10,336,510,3])==['duplicate_percussion_event']
    assert audit_tokens([2,10,336,500,9,342,510,3])==['decreasing_timestamp']
    assert audit_tokens([2,0,3,10,336,500])==['events_after_eos']
    assert audit_tokens([2,4,336,500,20,342,510,3],max_time_index=9)==['time_outside_chunk']
    assert 'duplicate_percussion_event' in audit_tokens([2,10,336,500,10,336,510])
    with pytest.raises(ValueError):decode_complete([2,10,336,500,3,10,342,490])
