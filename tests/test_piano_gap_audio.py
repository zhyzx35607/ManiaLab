import json

import numpy as np
import pytest

from scripts.piano_gap_audio import measure_gap, sample_range


def test_tone_decay_is_measured_without_claiming_release_or_audibility():
    rate = 22050;t = np.arange(rate*2)/rate
    samples = np.sin(2*np.pi*440*t)*np.where((t>=1)&(t<1.2), .01, .1)
    original = samples.copy()
    meta, arrays = measure_gap(samples, rate, 1000, 1200, [69])
    assert meta['waveform_gap_pre_db'] == pytest.approx(-20, abs=.01)
    band = next(b for b in meta['windows']['2048']['bands'] if b['partial'] == 1)
    assert band['gap_pre_db'] == pytest.approx(-20, abs=.02)
    assert not meta['release_established'] and not meta['audibility_established']
    assert meta['windows']['8192']['region_frames']['gap'] == 0
    assert all(b['regions']['gap']['power'] is None for b in meta['windows']['8192']['bands'])
    np.testing.assert_array_equal(samples, original)
    json.dumps(meta, allow_nan=False)
    assert all(np.isfinite(a).all() for a in arrays.values())


def test_next_attack_cannot_leak_into_full_support_gap_statistics():
    samples = np.zeros(22050*2);samples[22050:22060] = 1
    meta, _ = measure_gap(samples, 22050, 800, 1000, [69])
    assert meta['waveform']['gap']['power'] == 0
    assert meta['waveform_gap_pre_db'] is None
    for window in meta['windows'].values():
        for b in window['bands']:
            assert b['regions']['gap']['power'] in (0, None)


def test_scaling_changes_power_but_not_relative_decay():
    rate = 8000;signal = np.sin(2*np.pi*440*np.arange(rate*2)/rate)*.1
    signal[8000:9600]*=.1
    a, _ = measure_gap(signal, rate, 1000, 1200, [69], sizes=(512,))
    b, _ = measure_gap(signal*3, rate, 1000, 1200, [69], sizes=(512,))
    assert b['waveform']['gap']['power'] == pytest.approx(a['waveform']['gap']['power']*9)
    assert b['waveform_gap_pre_db'] == pytest.approx(a['waveform_gap_pre_db'])


def test_frame_spectra_and_sample_clock_match_direct_periodic_hann():
    rate=8000;signal=np.random.default_rng(1).normal(0,.1,rate*2)
    _, arrays=measure_gap(signal,rate,500,700,[],sizes=(512,),hop=128)
    starts=arrays['starts_512'];window=.5-.5*np.cos(2*np.pi*np.arange(512)/512)
    expected=np.stack([np.abs(np.fft.rfft(signal[x:x+512]*window))/window.sum() for x in starts],axis=1)
    np.testing.assert_allclose(arrays['magnitude_512'],expected[1:],rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(arrays['times_512'],(starts+256)/rate)


def test_short_actual_input_and_boundary_regions_remain_unmeasurable():
    m, a=measure_gap(np.zeros(10),1000,0,10,[],sizes=(512,),hop=128)
    assert m['waveform']['pre']['samples']==0 and m['waveform']['pre']['power'] is None
    assert m['windows']['512']['region_frames']==dict(pre=0,gap=0,post=0)
    assert a['magnitude_512'].shape[1]==0


def test_sample_coordinates_do_not_include_one_extra_boundary_sample():
    assert sample_range(1000,1100,22050,50000)==(22050,24255)
    assert sample_range(1000+1000/22050,1100,22050,50000)==(22051,24255)


@pytest.mark.parametrize('begin,end', [(500,500),(-1,500),(0,3000),(True,500),(0,float('nan'))])
def test_invalid_intervals_are_rejected(begin,end):
    with pytest.raises(ValueError):measure_gap(np.zeros(22050),22050,begin,end,[])
