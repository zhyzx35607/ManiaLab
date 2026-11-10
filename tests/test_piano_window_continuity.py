import numpy as np
import pytest

from manialab.piano import OUTPUTS, decode, infer
from manialab.piano_continuity import continue_overlap


def windows(anchor=400, fade=900):
    old = {k: np.zeros((1001, 88), np.float32) for k in OUTPUTS}
    new = {k: np.zeros((1001, 88), np.float32) for k in OUTPUTS}
    old['reg_onset_output'][anchor-1:anchor+2, 39] = [.2, .8, .2]
    old['frame_output'][anchor:fade, 39] = .9
    old['velocity_output'][:] = new['velocity_output'][:] = .5
    old['reg_offset_output'][fade-9:fade, 39] = [.1, .2, .4, .6, .9, .6, .4, .2, .1]
    return old, new


def assembled(old, new):
    return {k: np.concatenate([old[k][:750], new[k][250:1000]]) for k in OUTPUTS}


def test_observed_outgoing_release_survives_context_switch_without_new_attack():
    old, new = windows()
    originals = [{k:v.copy() for k,v in w.items()} for w in (old, new)]
    repaired, trace = continue_overlap(old, new)
    before, after = decode(assembled(old, new)), decode(assembled(old, repaired))
    assert len(before) == len(after) == len(trace) == 1
    assert before[0]['offset'] == 7.5
    assert after[0]['offset'] == pytest.approx(8.95, abs=1e-6)
    assert after[0]['release_reason'] == 'release_regression'
    for key in ('onset', 'pitch', 'velocity', 'onset_score'):
        assert before[0][key] == after[0][key]
    for k in ('reg_onset_output', 'velocity_output'):
        np.testing.assert_array_equal(repaired[k], new[k])
    for w, original in zip((old, new), originals):
        for k in OUTPUTS:
            np.testing.assert_array_equal(w[k], original[k])
    for k in OUTPUTS:
        np.testing.assert_array_equal(repaired[k][:, :39], new[k][:, :39])
        np.testing.assert_array_equal(repaired[k][:, 40:], new[k][:, 40:])


@pytest.mark.parametrize('condition', ['no_fade', 'edge_fade', 'unknown_attack', 'outside_owner',
    'attack_in_new_input', 'broken_old_sustain', 'new_attack_before_seam', 'new_reattack',
    'old_reattack', 'already_active', 'actual_seam_release'])
def test_ambiguous_or_unobserved_tail_cannot_be_extended(condition):
    old, new = windows()
    if condition == 'no_fade': old['frame_output'][900:, 39] = .9
    if condition == 'edge_fade': old['frame_output'][900:998, 39] = .9
    if condition == 'unknown_attack': old['reg_onset_output'][:] = 0
    if condition == 'outside_owner': old, new = windows(anchor=200)
    if condition == 'attack_in_new_input': old, new = windows(anchor=600)
    if condition == 'broken_old_sustain': old['frame_output'][600, 39] = 0
    if condition == 'new_attack_before_seam': new['reg_onset_output'][199:202, 39] = [.2, .8, .2]
    if condition == 'new_reattack': new['reg_onset_output'][349:352, 39] = [.2, .8, .2]
    if condition == 'old_reattack': old['reg_onset_output'][849:852, 39] = [.2, .8, .2]
    if condition == 'already_active': new['frame_output'][250, 39] = .9
    if condition == 'actual_seam_release': old['frame_output'][750, 39] = 0
    repaired, trace = continue_overlap(old, new)
    assert not trace
    for k in OUTPUTS:
        np.testing.assert_array_equal(repaired[k], new[k])


def test_first_window_owns_earlier_attacks_and_later_new_note_stays_separate():
    old, new = windows(anchor=200)
    new['reg_onset_output'][449:452, 39] = [.2, .8, .2]
    new['frame_output'][450:470, 39] = .9
    repaired, trace = continue_overlap(old, new, owned_left=0)
    notes = decode(assembled(old, repaired))
    assert len(trace) == 1 and len(notes) == 2
    assert notes[0]['offset'] < notes[1]['onset'] == 9.5
    assert notes[1] == decode(assembled(old, new))[1]


def test_regression_variant_requires_an_explicit_outgoing_release_peak():
    old, new = windows()
    repaired, trace = continue_overlap(old, new, require_release_regression=True)
    assert len(trace) == 1
    assert decode(assembled(old, repaired))[0]['release_reason'] == 'release_regression'
    old['reg_offset_output'][:] = 0
    repaired, trace = continue_overlap(old, new, require_release_regression=True)
    assert not trace
    for key in OUTPUTS:
        np.testing.assert_array_equal(repaired[key], new[key])


def test_regression_variant_does_not_replace_an_earlier_observed_release():
    old, new = windows()
    old['reg_offset_output'][586:595, 39] = [.1, .2, .4, .6, .9, .6, .4, .2, .1]
    assert decode(assembled(old,new))[0]['release_reason'] == 'release_regression'
    _, trace = continue_overlap(old, new, require_release_regression=True)
    assert not trace


def test_application_runtime_applies_the_rule_and_preserves_a_single_window():
    old, new = windows()
    class Replay:
        def __init__(self, values): self.values=iter(values)
        def run(self, names, inputs):
            value=next(self.values)
            return [value[k][None] for k in names]
    trace=[]
    # 20 seconds pads to three ten-second windows with five-second overlap.
    empty={k:np.zeros_like(v) for k,v in new.items()}
    result=infer(np.zeros(16000*20,np.float32),16000,Replay([old,new,empty]),continuation_trace=trace)
    assert len(trace)==1 and trace[0]['window_start_seconds']==5
    assert decode(result)[0]['offset']==pytest.approx(8.95,abs=1e-6)
    single=infer(np.zeros(16000*10,np.float32),16000,Replay([old]))
    for k in OUTPUTS:np.testing.assert_array_equal(single[k],old[k][:1000])


def test_cache_revision_and_context_trace_are_persisted_and_recovered(tmp_path,monkeypatch):
    import hashlib
    import json
    import onnxruntime as ort
    from manialab import piano
    detector=piano.PianoDetector(tmp_path)
    detector.model.parent.mkdir(parents=True)
    detector.model.write_bytes(b'model')
    model_sha=hashlib.sha256(b'model').hexdigest()
    monkeypatch.setattr(piano,'MODEL_SHA256',model_sha)
    monkeypatch.setattr(ort,'InferenceSession',lambda *a,**k:object())
    samples=np.zeros(16000,np.float32)
    cache=tmp_path/'cached.npz'
    old_identity=dict(revision=1,model_sha256=model_sha,stem_identity='stem',
        pcm_sha256=hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest(),sample_rate=16000,samples=16000)
    values={k:np.zeros((100,88),np.float32) for k in OUTPUTS}
    np.savez(cache,identity=json.dumps(old_identity,sort_keys=True),**values)
    calls=[]
    def prediction(*args):
        calls.append(1);args[-1].append(dict(pitch=60,window_start_seconds=5))
        return values
    monkeypatch.setattr(piano,'infer',prediction)
    _,fresh=detector.analyze(samples,16000,cache,'stem')
    _,cached=detector.analyze(samples,16000,cache,'stem')
    assert not fresh['cached'] and cached['cached'] and len(calls)==1
    assert fresh['identity']['revision']==2
    assert fresh['window_context_continuations']==cached['window_context_continuations']==[dict(pitch=60,window_start_seconds=5)]
    with np.load(cache,allow_pickle=False) as z:broken={k:z[k] for k in z.files}
    broken['continuations']='['
    np.savez(cache,**broken)
    _,recovered=detector.analyze(samples,16000,cache,'stem')
    assert not recovered['cached'] and len(calls)==2
