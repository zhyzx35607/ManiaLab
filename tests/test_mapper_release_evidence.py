import numpy as np
import pytest

from manialab.domain import Note
from manialab.mapper_release_evidence import EmbeddingEvidence, MapperReleaseContext
from scripts.extract_mapper_embeddings import window_positions


def test_retention_has_no_duplicate_or_missing_positions():
    for start,end in ((0,100000),(13212,101000),(24000,120000)):
        windows=window_positions(16000*130,start,end)
        positions=np.concatenate([p for _,p,_ in windows])/16
        np.testing.assert_array_equal(np.diff(positions),np.full(len(positions)-1,16))
        assert positions[0]>=start and positions[0]<start+16
        assert positions[-1]<=end and positions[-1]>end-16
        for base,p,mask in windows:
            np.testing.assert_array_equal(p,base+np.flatnonzero(mask)*256)


def test_physical_time_shift_and_independent_endpoint():
    times=np.arange(0,24000,16,dtype=float)
    values=np.tile((times<1520)[:,None],(1,32)).astype(np.float32)
    evidence=EmbeddingEvidence(times,values)
    context=MapperReleaseContext([Note(1000,1000,0),Note(2000,2000,0)],6,
                                ['0,500,4,2,1,60,1,0'],.35,evidence)
    row=list(context.options(0)).index(1500);features=context.features(0)
    assert features[row,context.feature_names.index('mapper_tail_before_0')]==1
    assert features[row,context.feature_names.index('mapper_tail_after_0')]==0
    assert features[row,context.feature_names.index('release_at_attack')]==0
    assert np.isfinite(features).all()
    assert features.shape[1]==len(context.feature_names)


def test_missing_embedding_context_is_rejected_instead_of_clamped():
    times=np.arange(1000,4000,16,dtype=float)
    evidence=EmbeddingEvidence(times,np.zeros((len(times),32)))
    with pytest.raises(ValueError,match='outside'): evidence.mean(900,1100)
    with pytest.raises(ValueError,match='outside'): evidence.features(1200,[5000])
    with pytest.raises(ValueError,match='Invalid embedding'): EmbeddingEvidence(times[::2],np.zeros((len(times[::2]),32)))


def test_interval_pooling_is_independent_of_samples_outside_interval():
    times=np.arange(0,4000,16,dtype=float)
    values=np.random.default_rng(17).normal(size=(len(times),32)).astype(np.float32)
    original=EmbeddingEvidence(times,values).mean([1000,2000],[1500,2500])
    values[times>3000]*=20
    changed=EmbeddingEvidence(times,values).mean([1000,2000],[1500,2500])
    np.testing.assert_array_equal(original,changed)


def test_planning_view_retains_global_rhythm_context():
    from manialab.release_candidates import ReleaseContext
    from manialab.gesture_flow import plan_flow
    from scripts.evaluate_mapper_releases import ContextView
    full=ReleaseContext([Note(i*500,i*500,0) for i in range(40)],6,['0,500,4,2,1,60,1,0'])
    view=ContextView(full,[10,11,12])
    np.testing.assert_array_equal(view.features(0),full.features(10))
    predicted=[]
    for i in range(3):
        options=view.options(i);probability=np.zeros(len(options));probability[0]=1
        predicted.append(dict(options=options,probabilities=probability))
    events,report=plan_flow(view,predicted,hierarchical=True)
    assert [n.start for n in events]==[5000,5500,6000]
    assert report['endpoint_source']==full.endpoint_source
