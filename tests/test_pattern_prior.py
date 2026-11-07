import numpy as np
import pytest

from manialab.pattern_prior import NETWORK_FEATURES, mirror_features, network_features, score_network


@pytest.mark.parametrize('keys',range(6,11))
def test_network_features_keep_lane_state_under_left_right_mirroring(keys):
    choices=np.array([[0,keys-2],[1,keys-1]])
    ends=np.arange(keys)*170+800
    last=ends-200
    history=[(400,3,2),(800,9,2)]
    reverse_mask=lambda mask:sum(1<<(keys-1-i) for i in range(keys) if mask&(1<<i))
    durations=np.array([0,1400])
    x=network_features(choices,keys,1000,ends,last,history,durations,500)
    mirrored=network_features(keys-1-choices[:,::-1],keys,1000,ends[::-1],last[::-1],
        [(t,reverse_mask(m),s) for t,m,s in history],durations[::-1],500)
    assert x.shape==(2,len(NETWORK_FEATURES)) and np.isfinite(x).all()
    assert np.allclose(mirror_features(x,keys),mirrored)
    assert np.array_equal(mirror_features(mirror_features(x,keys),keys),x)


def test_exported_network_scoring_matches_known_nonlinear_function():
    layers=[{'weight':[[1.,-1.],[0.,2.]],'bias':[0.,-1.]},
            {'weight':[[2.,3.]],'bias':[.5]}]
    result=score_network(np.array([[2.,1.],[-1.,0.]]),layers)
    assert np.allclose(result,[5.5,.5])
