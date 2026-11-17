from copy import deepcopy

import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from scripts.replay_frozen_generation import capacity_report, compare_variants, replay_case, run


def test_capacity_checks_held_limit_without_new_attack():
    notes = [Note(1000,4000,id='a'), Note(1500,4500,id='b')]
    result = capacity_report(notes, parameters(dict(max_held=2)),
        [dict(start_ms=2000,end_ms=5000,preferences=dict(max_held=1))])
    assert not result['valid']
    assert dict(time_ms=2000,constraint='max_held',actual=2,limit=1) in result['violations']


def test_pure_boundary_density_is_observation_not_capacity_failure():
    notes = [Note(1000,1000,id='a'), Note(1500,1500,id='b')]
    result = capacity_report(notes, parameters(dict(density=3)),
        [dict(start_ms=1600,end_ms=5000,preferences=dict(density=1))])
    assert result['valid']
    assert result['boundary_density_observations'][0]['time_ms'] == 1600


def test_diagnostic_generation_preserves_rejected_automatic_timing():
    analysis = dict(revision=3,duration=5,candidates=[Note(1000,2000,source='pitch',id='a').data()],
        phrase_candidates=[],timing_attacks=[],tempo=dict(bpm=120,offset=0,division=4,stable=False))
    case = dict(settings=dict(keys=6),tempo=dict(analysis['tempo'],audio_shift_ms=0))
    original = deepcopy(analysis)
    variant, summary, _ = replay_case(analysis,case)
    assert analysis == original
    assert summary['passed'] and not summary['automatic_timing_gate']['accepted']
    assert variant['timing_confirmed'] is True
    assert summary['timing_diagnostics'] == dict(status='insufficient',window_clock='audio',
        assessed_windows=0,insufficient_windows=1,fallback_windows=0)
    assert summary['source_snapshot_notes'] == 1


def test_release_summary_distinguishes_grid_movement_from_detection_provenance():
    analysis = dict(revision=13,duration=5,candidates=[Note(1000,1940,source='pitch',id='a',
        release_reason='model_predicted_offset').data()],phrase_candidates=[],timing_attacks=[])
    _,summary,_ = replay_case(analysis,dict(settings=dict(keys=6,ln_ratio=1),
        tempo=dict(bpm=120,offset=0,division=4,stable=True,audio_shift_ms=0)))
    voice = summary['voices']['mix']
    assert voice['large_release_shifts'] == 1 and voice['max_release_shift_ms'] == 60
    assert voice['later_release_shifts'] == 1 and voice['earlier_release_shifts'] == 0
    assert voice['release_sources'] == {'model_predicted_offset':1}


def test_compare_preserves_ids_and_distinguishes_lane_and_tail_changes():
    before = dict(notes=[Note(0,100,lane=0,id='a').data(),Note(200,200,id='b').data()])
    after = dict(notes=[Note(0,80,lane=1,id='a').data(),Note(300,300,id='c').data()])
    diff = compare_variants(before,after)
    assert diff['removed'] == ['b'] and diff['added'] == ['c']
    assert diff['changed'][0]['before']['end'] == 100
    assert diff['changed'][0]['after']['lane'] == 1
    assert diff['timing_changed'] == ['a'] and not diff['lane_only_changed']


def test_compare_separates_lane_only_and_evidence_changes():
    before = dict(notes=[Note(0,100,lane=0,id='a').data()])
    after = dict(notes=[Note(0,100,lane=1,id='a',release_reason='input_boundary').data()])
    diff = compare_variants(before,after)
    assert diff['lane_only_changed'] == ['a'] and not diff['timing_changed']
    assert diff['evidence_changed'] == [dict(id='a',fields=['release_reason'])]


def test_replay_rejects_changed_inputs_before_creating_output(tmp_path):
    from scripts.replay_frozen_generation import write
    analysis = tmp_path/'analysis.json'
    analysis.write_text('{}',encoding='utf-8')
    plan = tmp_path/'plan.json'
    write(plan,dict(schema='manialab-frozen-generation-v1',cases=[dict(id='case',
        analysis=str(analysis),analysis_sha256='wrong')]))
    output = tmp_path/'output'
    with pytest.raises(ValueError,match='Frozen analysis changed'):
        run(plan,output)
    assert not output.exists()


def frozen_plan(tmp_path):
    from scripts.replay_frozen_generation import sha, write
    analysis = tmp_path/'analysis.json'
    write(analysis,dict(revision=3,duration=5,candidates=[Note(1000,2000,source='pitch',id='a').data()],
        phrase_candidates=[],timing_attacks=[],tempo=dict(bpm=120,offset=0,division=4,stable=True)))
    plan = tmp_path/'plan.json'
    write(plan,dict(schema='manialab-frozen-generation-v1',cases=[dict(id='case',
        analysis=str(analysis),analysis_sha256=sha(analysis),settings=dict(keys=6),
        tempo=dict(bpm=120,offset=0,division=4,stable=True,audio_shift_ms=0))]))
    return plan,analysis


def test_replay_rejects_tampered_baseline_variant(tmp_path):
    plan,_ = frozen_plan(tmp_path)
    baseline = tmp_path/'baseline'
    assert run(plan,baseline)['passed']
    (baseline/'case/variant.json').write_text('{}',encoding='utf-8')
    report = run(plan,tmp_path/'comparison',baseline)
    assert not report['passed']
    assert 'Baseline variant changed' in report['cases'][0]['error']


@pytest.mark.parametrize('target', ['source','input'])
def test_replay_rejects_midrun_identity_change(tmp_path,monkeypatch,target):
    import scripts.replay_frozen_generation as replay
    plan,analysis = frozen_plan(tmp_path)
    original = replay.replay_case
    snapshot = {'fixture.py':'before'}
    monkeypatch.setattr(replay,'sources',lambda: dict(snapshot))
    def changing_case(*args):
        result = original(*args)
        if target == 'source':
            snapshot['fixture.py'] = 'after'
        else:
            analysis.write_text('{}',encoding='utf-8')
        return result
    monkeypatch.setattr(replay,'replay_case',changing_case)
    output = tmp_path/'output'
    with pytest.raises(ValueError,match='changed during replay'):
        replay.run(plan,output)
    assert not (output/'report.json').exists()
