import pytest

from manialab.arrange import parameters
from manialab.expression import propose_expression,expression_evidence
from scripts.instrument_candidate_adapter import accompaniment_candidates


def event(at=1.,end=2.,program=40,drum=False):
    return dict(onset=at,offset=end,pitch=60,program=program,is_drum=drum,velocity=100)


def test_ambiguous_release_preserves_raw_attack_and_end_but_cannot_become_ln():
    raw=[event(),event(at=3.,end=4.)]
    audit=[dict(note_index=0,endpoint_evidence=['explicit_model_note_off','forced_segment_boundary']),
           dict(note_index=1,endpoint_evidence=['explicit_model_note_off'])]
    notes,trace=accompaniment_candidates(raw,audit,10000)
    analysis=dict(duration=10,candidates=[n.data() for n in notes],
                  structure=dict(segments=[dict(start_ms=0,end_ms=10000)]))
    settings=parameters({'ln_ratio':1.})
    plan=propose_expression(analysis,settings)
    result,_,_=expression_evidence(analysis,plan,settings)
    assert [(n.start,n.end) for n in result]==[(1000,1000),(3000,4000)]
    assert notes[0].end==2000 and raw[0]['offset']==2.
    assert trace[0]['weight_semantics']=='unit_weight_not_confidence'


def test_all_incompatible_predictions_remain_in_trace_and_unknown_audit_fails():
    events=[event(drum=True,program=128),event(program=100),event(at=12.,end=13.),event()]
    audit=[dict(note_index=i,endpoint_evidence=['drum_fixed_10ms'] if i==0 else ['explicit_model_note_off'])
           for i in range(len(events))]
    notes,trace=accompaniment_candidates(events,audit,10000)
    assert len(notes)==1 and len(trace)==4
    assert [r['decision'] for r in trace]==['incompatible_drum_prediction','incompatible_vocal_prediction',
        'outside_audio','candidate']
    with pytest.raises(ValueError,match='exactly once'):accompaniment_candidates(events,audit[:-1],10000)
