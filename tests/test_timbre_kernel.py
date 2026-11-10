import numpy as np
import pytest
from manialab.timbre_kernel import kernel,fit,predict


def test_disconnected_timbre_clusters_can_share_a_class():
    x=np.asarray([[-1,-1],[-1,1],[1,-1],[1,1]],float);y=np.asarray([0,1,1,0])
    model=fit(x,y,.5)
    assert np.array_equal(predict(x,model)>.5,y.astype(bool))
    assert np.all(predict(x,model)[y==1]>.8)
    assert np.all(predict(x,model)[y==0]<.2)
    np.testing.assert_allclose(predict(x[::-1],model),predict(x,model)[::-1])


def test_kernel_refuses_nonfinite_or_missing_classes():
    with pytest.raises(ValueError):kernel([[np.nan]],[[1]],1)
    with pytest.raises(ValueError):kernel([[1]],[[1]],0)
    with pytest.raises(ValueError):fit([[1],[2]],[0,0],1)
