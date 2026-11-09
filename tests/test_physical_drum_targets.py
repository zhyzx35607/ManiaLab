import numpy as np
import pytest
from scripts.physical_drum_targets import known_targets


def test_solo_recording_does_not_invent_positive_or_negative_onsets():
    targets=np.asarray([[0,-1,0],[0,-1,0]],np.float32)
    values,mask=known_targets(targets)
    np.testing.assert_array_equal(values,np.zeros_like(values))
    assert mask.tolist()==[[True,False,True],[True,False,True]]
    complete=np.array([[1,.25,0],[0,1,1]],np.float32)
    values,mask=known_targets(complete)
    np.testing.assert_array_equal(values,complete)
    assert mask.all()


def test_ambiguous_partial_targets_fail_explicitly():
    for invalid in ([[0,-.5,0]],[[0,np.nan,0]],[[0,2,0]]):
        with pytest.raises(ValueError):known_targets(invalid)
