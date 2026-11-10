"""Source-preserving controls and one-to-one acoustic-anchor matching tests."""
import copy
import hashlib
import itertools
import json

import numpy as np
import pytest

from scripts.piano_source_controls import compose_case, score_attacks


def component(pitch=60, anchor=.25, gain=1.):
    return dict(pitch=pitch, anchor_seconds=anchor, gain=gain)


def ref(anchor, pitch=60):
    return dict(pitch=pitch, anchor_seconds=anchor)


def pred(onset, pitch=60, duration=.1):
    return dict(pitch=pitch, onset=onset, offset=onset+duration)


def test_mixture_is_exact_sum_with_complete_source_identity_and_no_input_mutation():
    sources = {48: np.array([.02,.2,-.1,.04], dtype=np.float64),
               60: np.array([.01,.1,-.03], dtype=np.float32)}
    before = {key:value.copy() for key,value in sources.items()}
    components = [component(48,.25,1.5),component(60,.3,.25)]
    old_components = copy.deepcopy(components)
    anchors = {48:1,60:1}
    mixed, metadata = compose_case(sources,components,20,1.,source_anchor_samples=anchors)
    expected = np.zeros(20,dtype=np.float64)
    expected[4:8] += sources[48]*1.5
    expected[5:8] += sources[60].astype(np.float64)*.25
    np.testing.assert_array_equal(mixed,expected.astype(np.float32))
    assert mixed.dtype == np.float32 and metadata[0]['kind']=='mixture'
    assert metadata[0]['mix_pcm_sha256']==hashlib.sha256(mixed.astype('<f4').tobytes()).hexdigest()
    assert metadata[0]['component_count']==2
    assert metadata[0]['normalization_applied'] is False and metadata[0]['clipping_applied'] is False
    assert metadata[1]['source_pcm_sha256']==hashlib.sha256(sources[48].astype('<f8').tobytes()).hexdigest()
    assert metadata[2]['source_pcm_sha256']==hashlib.sha256(sources[60].astype('<f4').tobytes()).hexdigest()
    assert metadata[1]['placement_start_sample']==4 and metadata[1]['placement_stop_sample']==8
    assert metadata[1]['anchor_sample']==5 and metadata[1]['source_anchor_sample']==1
    assert metadata[1]['source_dtype']=='<f8' and metadata[2]['source_dtype']=='<f4'
    assert components==old_components and anchors=={48:1,60:1}
    for pitch in sources:
        np.testing.assert_array_equal(sources[pitch],before[pitch])
        assert not np.shares_memory(mixed,sources[pitch])
    json.dumps(metadata,allow_nan=False)


def test_adding_high_source_keeps_low_component_identical_and_reproducible():
    sources={48:np.linspace(.01,.08,8),72:np.linspace(.01,.04,4)}
    anchors={48:2,72:1}
    low,_=compose_case(sources,[component(48,.25)],20,1.,source_anchor_samples=anchors)
    high,_=compose_case(sources,[component(72,.4)],20,1.,source_anchor_samples=anchors)
    paired,metadata=compose_case(sources,[component(48,.25),component(72,.4)],20,1.,source_anchor_samples=anchors)
    reconstructed=np.zeros(20,dtype=np.float64)
    for row in metadata[1:]:
        reconstructed[row['placement_start_sample']:row['placement_stop_sample']]+=sources[row['pitch']]*row['gain']
    np.testing.assert_array_equal(paired,reconstructed.astype(np.float32))
    np.testing.assert_array_equal(paired[:7],low[:7])
    np.testing.assert_allclose(paired,low.astype(float)+high.astype(float),atol=1e-8,rtol=0)
    assert metadata[1]['source_pcm_sha256']==hashlib.sha256(sources[48].astype('<f8').tobytes()).hexdigest()


def test_retrigger_adds_without_overwriting_previous_tail():
    source=np.array([.1,.2,.3,.2,.1,.05])
    mixed,metadata=compose_case({60:source},[component(anchor=.2),component(anchor=.4)],10,1.,source_anchor_samples={60:0})
    expected=np.array([0,0,.1,.2,.4,.4,.4,.25,.1,.05],dtype=np.float32)
    np.testing.assert_allclose(mixed,expected,rtol=0,atol=3e-8)
    assert metadata[1]['placement_stop_sample']>metadata[2]['placement_start_sample']


def test_components_are_accumulated_in_float64_before_one_float32_cast():
    sources={48:np.array([.5],dtype=np.float32),60:np.array([2**-26],dtype=np.float32)}
    components=[component(48,0),component(60,0),component(60,0),component(60,0)]
    mixed,_=compose_case(sources,components,1,1.,source_anchor_samples={48:0,60:0})
    assert mixed[0]==np.float32(.5+3*2**-26)
    assert mixed[0]>np.float32(.5)


def test_whole_sample_translation_preserves_complete_mixture():
    sources={60:np.array([.02,.3,.1,-.05]),72:np.array([.1,.2])}
    anchors={60:1,72:0}
    a,_=compose_case(sources,[component(60,.2),component(72,.3)],20,1.,source_anchor_samples=anchors)
    b,m=compose_case(sources,[component(60,.4),component(72,.5)],20,1.2,source_anchor_samples=anchors)
    np.testing.assert_array_equal(b[:4],np.zeros(4,np.float32))
    np.testing.assert_array_equal(b[4:],a)
    assert m[1]['anchor_sample']==8


def test_source_anchor_zero_and_exact_last_sample_fit():
    mixed,m=compose_case({60:np.array([.1,.2])},[component(anchor=.8)],10,1.,source_anchor_samples={60:0})
    assert mixed[-1]==np.float32(.2) and m[1]['placement_stop_sample']==10


def test_silence_retains_mixture_identity_and_zero_gain_is_allowed():
    empty,m=compose_case({},[],source_anchor_samples={})
    assert empty.shape==(160000,) and not empty.any() and len(m)==1
    assert m[0]['mix_pcm_sha256']==hashlib.sha256(empty.tobytes()).hexdigest()
    zero,zm=compose_case({60:np.ones(100)},[component(gain=0)],source_anchor_samples={60:0})
    np.testing.assert_array_equal(zero,empty)
    assert zm[1]['gain']==0


def test_low_amplitude_and_noncontiguous_big_endian_sources_are_not_normalized():
    samples=np.array([1e-8,0,-2e-8,0,3e-8],dtype='>f8')[::2]
    mixed,m=compose_case({60:samples},[component(anchor=0)],10,.3,source_anchor_samples={60:0})
    np.testing.assert_array_equal(mixed,samples.astype(np.float32))
    assert m[1]['source_pcm_sha256']==hashlib.sha256(np.asarray(samples,dtype='<f8').tobytes()).hexdigest()
    assert m[0]['peak_float32']<1e-7


def test_rounding_metadata_reports_realized_sample_time():
    _,m=compose_case({60:np.array([.1])},[component(anchor=.125)],20,.525,source_anchor_samples={60:0})
    assert m[0]['sample_count']==10 and m[0]['duration_seconds_requested']==.525
    assert m[0]['duration_seconds_realized']==.5
    assert m[1]['anchor_sample']==2 and m[1]['anchor_seconds_realized']==.1


@pytest.mark.parametrize('source', [np.array([1.]),np.array([-1.]),np.array([.99999999])])
def test_peak_at_one_in_float64_or_after_float32_rounding_is_rejected(source):
    with pytest.raises(ValueError,match='peak'):
        compose_case({60:source},[component()],source_anchor_samples={60:0})


def test_overlapping_sum_is_rejected_instead_of_clipped():
    with pytest.raises(ValueError,match='peak'):
        compose_case({60:np.array([.6])},[component(),component()],source_anchor_samples={60:0})


def test_nonfinite_accumulation_from_finite_source_and_gain_is_rejected():
    with pytest.raises(ValueError,match='Nonfinite mixed PCM'):
        compose_case({60:np.array([1e308])},[component(gain=1e308)],source_anchor_samples={60:0})


@pytest.mark.parametrize('source',[[],[1,2],[[.1]],[float('nan')],[float('inf')],np.array([1j]),np.array([.1],dtype=np.float16)])
def test_invalid_source_pcm_is_rejected(source):
    with pytest.raises(ValueError,match='Source PCM'):
        compose_case({60:source},[component()],source_anchor_samples={60:0})


@pytest.mark.parametrize('anchor',[-1,3,True,1.2,float('nan')])
def test_source_anchor_must_be_valid_integer_sample(anchor):
    with pytest.raises(ValueError,match='Source anchor'):
        compose_case({60:np.ones(3)*.1},[component()],source_anchor_samples={60:anchor})


@pytest.mark.parametrize('anchor',[-.1,.0,.99,float('nan'),float('inf'),True,'1'])
def test_invalid_target_anchor_or_head_tail_truncation_is_rejected(anchor):
    with pytest.raises(ValueError):
        compose_case({60:np.ones(10)*.1},[component(anchor=anchor)],20,1.,source_anchor_samples={60:2})


@pytest.mark.parametrize('gain',[-1,float('nan'),float('inf'),True,'1'])
def test_invalid_gain_is_rejected(gain):
    with pytest.raises(ValueError,match='Gain'):
        compose_case({60:np.array([.1])},[component(gain=gain)],source_anchor_samples={60:0})


@pytest.mark.parametrize('sample_rate',[0,-1,True,16000.,'16000'])
def test_invalid_sample_rate_is_rejected(sample_rate):
    with pytest.raises(ValueError,match='Sample rate'):
        compose_case({},[],sample_rate,source_anchor_samples={})


@pytest.mark.parametrize('duration',[0,-1,float('nan'),float('inf'),True,'10',1e-8])
def test_invalid_duration_is_rejected(duration):
    with pytest.raises(ValueError,match='Duration'):
        compose_case({},[],duration_seconds=duration,source_anchor_samples={})


@pytest.mark.parametrize('sources,anchors,components',[
    ({60:np.array([.1])},{},[component()]),({}, {60:0}, []),
    ({60:np.array([.1])},{60:0},[component(61)]),
    ({True:np.array([.1])},{True:0},[]),({128:np.array([.1])},{128:0},[]),
    ({1:np.array([.1])},{True:0},[]),
    ({60:np.array([.1])},{60:0},[{}]),({60:np.array([.1])},{60:0},[None]),
    ([],{},[]),({},[],[]),({}, {}, None),
])
def test_invalid_source_and_component_mappings_are_rejected(sources,anchors,components):
    with pytest.raises(ValueError):
        compose_case(sources,components,source_anchor_samples=anchors)


def test_score_maximum_cardinality_beats_greedy_nearest_first():
    # Ref 1 is closest to pred 1.04, but only it can use .91; ref 1.09 needs 1.04.
    scored=score_attacks([ref(1.),ref(1.09)],[pred(1.04),pred(.91)],.1)
    assert scored['matched']==2 and scored['missing']==scored['extra']==0
    assert [(r['ref_index'],r['pred_index']) for r in scored['matches']]==[(0,1),(1,0)]
    assert [r['signed_error_seconds'] for r in scored['matches']]==pytest.approx([-.09,-.05])


def test_cardinality_precedes_time_cost_even_with_one_very_distant_allowed_pair():
    scored=score_attacks([ref(1.),ref(100.)],[pred(1.),pred(2.)],99.)
    assert scored['matched']==2
    assert sum(r['absolute_error_seconds'] for r in scored['matches'])==pytest.approx(98.)


def test_equal_cardinality_minimizes_total_absolute_error():
    scored=score_attacks([ref(1.),ref(1.2)],[pred(1.01),pred(1.19),pred(1.1)],.2)
    assert [(r['ref_index'],r['pred_index']) for r in scored['matches']]==[(0,0),(1,1)]
    assert scored['extra_pred_indices']==[2]


def test_duplicate_events_are_one_to_one_and_different_pitch_never_matches():
    scored=score_attacks([ref(1.)],[pred(1.),pred(1.),pred(1.,61)],.1)
    assert scored['matched']==1 and scored['extra']==2
    assert len(set(r['pred_index'] for r in scored['matches']))==1
    assert 2 in scored['extra_pred_indices']
    scored=score_attacks([ref(1.),ref(1.)],[pred(1.)],.1)
    assert scored['matched']==1 and scored['missing']==1


@pytest.mark.parametrize('anchor',[1.,1.2,1000.])
@pytest.mark.parametrize('shift',[-.1,.1])
def test_exact_tolerance_boundary_is_inclusive(anchor,shift):
    assert score_attacks([ref(anchor)],[pred(anchor+shift)],.1)['matched']==1
    outside=shift+np.sign(shift)*.000001
    assert score_attacks([ref(anchor)],[pred(anchor+outside)],.1)['matched']==0


def test_zero_tolerance_is_exact_not_ulp_expanded():
    assert score_attacks([ref(1.)],[pred(1.)],0)['matched']==1
    assert score_attacks([ref(1.)],[pred(np.nextafter(1.,2.))],0)['matched']==0


def test_extra_predictions_outside_case_remain_in_denominator_and_offsets_are_not_scored():
    expected=[ref(1.)]
    predicted=[pred(1.,duration=30.),pred(-5.),pred(50.),pred(1.,61)]
    before=copy.deepcopy(predicted)
    scored=score_attacks(expected,predicted)
    assert scored['predicted']==4 and scored['matched']==1 and scored['extra']==3
    assert scored['precision']==.25 and scored['recall']==1 and scored['f1']==.4
    assert scored['extra_pred_indices']==[1,2,3] and scored['offsets_scored'] is False
    assert predicted==before
    json.dumps(scored,allow_nan=False)


def test_optional_offset_and_unrelated_prediction_fields_do_not_change_onset_score():
    assert score_attacks([ref(1.)],[dict(pitch=60,onset=1.,confidence=.01)])['matched']==1


def test_silent_reference_false_positives_and_empty_denominators_are_explicit():
    a=score_attacks([],[])
    assert a['matched']==a['missing']==a['extra']==0 and a['f1']==0
    b=score_attacks([],[pred(0),pred(20)])
    assert b['extra']==2 and b['precision']==0 and b['recall']==0
    c=score_attacks([ref(1.)],[])
    assert c['missing_ref_indices']==[0] and c['missing']==1


def test_score_clock_translation_keeps_pair_identity_and_errors():
    expected=[ref(1.),ref(1.4),ref(2.,72)]
    predicted=[pred(1.03),pred(1.32),pred(2.03,72),pred(1.,61)]
    a=score_attacks(expected,predicted)
    b=score_attacks([ref(r['anchor_seconds']+100,r['pitch']) for r in expected],
                    [pred(r['onset']+100,r['pitch']) for r in predicted])
    assert [(r['ref_index'],r['pred_index']) for r in a['matches']]==[(r['ref_index'],r['pred_index']) for r in b['matches']]
    assert a['extra_pred_indices']==b['extra_pred_indices']
    assert [r['signed_error_seconds'] for r in a['matches']]==pytest.approx([r['signed_error_seconds'] for r in b['matches']])


def test_matching_agrees_with_exhaustive_small_graph_assignment():
    # Independent enumeration covers ambiguous, duplicate and unreachable events.
    rng=np.random.default_rng(5)
    for _ in range(30):
        expected=[ref(float(t)) for t in rng.integers(0,9,3)/10]
        predicted=[pred(float(t)) for t in rng.integers(0,9,4)/10]
        choices=[]
        for assignment in itertools.product(range(-1,len(predicted)),repeat=len(expected)):
            chosen=[j for j in assignment if j>=0]
            if len(set(chosen))!=len(chosen):
                continue
            errors=[abs(predicted[j]['onset']-expected[i]['anchor_seconds'])
                    for i,j in enumerate(assignment) if j>=0]
            if any(error>.25 for error in errors):
                continue
            choices.append((-len(chosen),sum(errors)))
        best=min(choices)
        score=score_attacks(expected,predicted,.25)
        assert score['matched']==-best[0]
        assert sum(row['absolute_error_seconds'] for row in score['matches'])==pytest.approx(best[1])


@pytest.mark.parametrize('tolerance',[-1,float('nan'),float('inf'),True,'0.1'])
def test_invalid_score_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError,match='Tolerance'):
        score_attacks([],[],tolerance)


@pytest.mark.parametrize('expected,predicted',[
    ([ref(-1.)],[]),([ref(float('nan'))],[]),([ref(True)],[]),
    ([ref(1.,60.)],[]),([ref(1.,128)],[]),([{}],[]),([None],[]),
    ([],[pred(float('inf'))]),([],[pred(1.,True)]),([],[pred(1.,-1)]),
    ([],[dict(pitch=60,onset=1.,offset=.9)]),([],[dict(pitch=60,onset=1.,offset=float('nan'))]),
    ([],[{}]),([],[None]),(None,[]),([],{}),
])
def test_invalid_scoring_inputs_are_rejected(expected,predicted):
    with pytest.raises(ValueError):
        score_attacks(expected,predicted)
