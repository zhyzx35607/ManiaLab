"""Research I/O checks; no optional Torch or model download in the test suite."""
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.transkun_runtime import event_record, input_pcm, transcribe


def note(start=1., end=2., pitch=60, velocity=64, has_onset=True, has_offset=False):
    return SimpleNamespace(start=start, end=end, pitch=pitch, velocity=velocity,
                           hasOnset=has_onset, hasOffset=has_offset)


def test_record_copies_values_before_upstream_merger_mutates_the_note():
    original = note()
    saved = event_record(original)
    original.start = 9.
    original.end = 12.
    original.hasOffset = True
    assert saved == dict(onset=1., offset=2., pitch=60, velocity=64,
                         has_onset=True, has_offset=False)


@pytest.mark.parametrize('pitch', [-67, -64, 21, 108])
def test_known_pedal_and_keyboard_symbols_are_retained(pitch):
    assert event_record(note(pitch=pitch))['pitch'] == pitch


@pytest.mark.parametrize('changes', [dict(start=np.nan), dict(end=np.inf),
                                   dict(start=3., end=2.), dict(pitch=20),
                                   dict(pitch=109), dict(pitch=-1)])
def test_invalid_events_are_not_silently_discarded(changes):
    with pytest.raises(ValueError):
        event_record(note(**changes))


def test_sample_rate_change_preserves_origin_duration_and_relative_level():
    original = np.zeros(16000, dtype=np.float32)
    original[8000] = .03125
    converted = input_pcm(original, 16000)
    assert converted.shape == (44100,) and converted.dtype == np.float32
    assert np.argmax(converted) == 22050
    assert .03 < converted.max() < .033
    np.testing.assert_array_equal(input_pcm(original*4, 16000), converted*4)
    assert np.count_nonzero(input_pcm(np.zeros(16000,dtype=np.float32),16000)) == 0


def test_native_rate_does_not_normalize_or_round_float32_values():
    original = np.array([0., .00001, -.125, .375], dtype=np.float32)
    np.testing.assert_array_equal(input_pcm(original,44100), original)


@pytest.mark.parametrize('samples,rate', [
    (np.array([1],dtype=np.int16),16000),
    (np.array([1.],dtype=np.float64),16000),
    (np.array([np.nan],dtype=np.float32),16000),
    (np.zeros((4,2),dtype=np.float32),16000),
    (np.zeros(0,dtype=np.float32),16000),
    (np.zeros(4,dtype=np.float32),0),
    (np.zeros(4,dtype=np.float32),True),
])
def test_invalid_audio_clock_or_pcm_is_rejected(samples,rate):
    with pytest.raises(ValueError):
        input_pcm(samples,rate)


class MutatingModel:
    fs = 44100
    hopSize = 1024
    segmentHopSizeInSecond = 8
    segmentSizeInSecond = 16

    def transcribeFrames(self,frames,**kwargs):
        return [[note()]], [100]*90

    def transcribe(self,x,**kwargs):
        assert x.shape == (441000,1)
        assert kwargs == dict(discardSecondHalf=False,mergeIncompleteEvent=True)
        self.raw_returned=[]
        for i in range(4):
            batch,last=self.transcribeFrames(np.zeros((1,1,691,4096),dtype=np.uint8),
                forcedStartPos=[344 if i==0 else 0]*90,lastFrameIdx=689)
            self.raw_returned.append(batch[0][0])
            # Mimics the important aliasing hazard in upstream shifting/merge.
            batch[0][0].start += 20
            batch[0][0].end += 20
            batch[0][0].hasOffset=True
        return [note(1.,3.,-64),note(11.,12.,60),note(2.,2.,72)]


def test_trace_preserves_premerge_values_pedals_and_outside_input_events(monkeypatch):
    fake_torch=SimpleNamespace(inference_mode=nullcontext,from_numpy=lambda x:x)
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    model=MutatingModel()
    previous=model.transcribeFrames
    result=transcribe(model,np.zeros(441000,dtype=np.float32))
    assert model.transcribeFrames==previous
    assert len(result['windows'])==4
    assert [r['start_sample'] for r in result['windows']]==[-352800,480,353760,707040]
    for row in result['windows']:
        assert row['origin_seconds']==pytest.approx(row['start_sample']/44100)
        assert row['events'][0][0]['onset']==1.
        assert not row['events'][0][0]['has_offset']
        assert row['last_frame_index']==689
    assert result['piano_onsets_outside_input']==1
    assert len(result['piano_events'])==2
    assert result['piano_events'][0]['onset']==11.
    assert result['piano_events'][1]['offset']==2.
    assert len(result['pedal_events'])==1 and result['pedal_events'][0]['pitch']==-64
    assert result['final_offset_flags_are_postprocessed']
    assert not result['release_truth_available']


def test_trace_restores_original_method_when_model_fails(monkeypatch):
    fake_torch=SimpleNamespace(inference_mode=nullcontext,from_numpy=lambda x:x)
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    model=MutatingModel()
    previous=model.transcribeFrames
    def fail(*args,**kwargs):
        raise RuntimeError('model failure')
    model.transcribe=fail
    with pytest.raises(RuntimeError,match='model failure'):
        transcribe(model,np.zeros(441000,dtype=np.float32))
    assert model.transcribeFrames==previous
