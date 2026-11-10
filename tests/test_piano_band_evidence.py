"""Physical-clock and missing-evidence tests for descriptive FFT bands."""
import json

import numpy as np
import pytest

from scripts.piano_band_evidence import analyze_window


RATE = 22050


def pcm(seconds=2.):
    return np.zeros(round(seconds * RATE), dtype=np.float64)


def band(metadata, size, pitch, partial=1):
    return next(row for row in metadata['windows'][str(size)]['bands']
                if row['pitch'] == pitch and row['partial'] == partial)


def test_metadata_and_arrays_are_strictly_serializable_finite_and_do_not_modify_inputs():
    samples = pcm()
    samples[22050] = .5
    before = samples.copy()
    lowers = [57, 45, 57]
    metadata, arrays = analyze_window(samples, RATE, 1., 69, lowers)
    json.dumps(metadata, allow_nan=False)
    np.testing.assert_array_equal(samples, before)
    assert lowers == [57, 45, 57]
    assert metadata['lower_pitches'] == [45, 57]
    assert metadata['descriptive_only'] and not metadata['independent_attack_established']
    assert metadata['boundary'] is None and metadata['padded'] is False
    assert metadata['window'] == 'periodic Hann'
    for value in arrays.values():
        assert isinstance(value, np.ndarray) and np.isfinite(value).all()
        assert not np.shares_memory(samples, value)


def test_spectra_match_independent_periodic_hann_fft_and_sample_clock():
    samples = pcm()
    rng = np.random.default_rng(12)
    samples[:] = rng.normal(0, .1, len(samples))
    metadata, arrays = analyze_window(samples, RATE, 1., 69, [57], sizes=(512,), hop=44)
    times = arrays['times_512']
    starts = np.rint(times * RATE - 256).astype(int)
    window = .5 - .5 * np.cos(2*np.pi*np.arange(512)/512)
    expected = np.stack([np.abs(np.fft.rfft(samples[start:start+512] * window))/window.sum()
                         for start in starts], axis=1)
    np.testing.assert_allclose(arrays['magnitude_512'], expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.diff(times), 44/RATE, rtol=0, atol=1e-15)
    assert (times >= .75).all() and (times <= 1.45).all()
    assert (starts >= metadata['extraction_samples'][0]).all()
    assert (starts+512 <= metadata['extraction_samples'][1]).all()
    assert metadata['windows']['512']['window_seconds'] == 512/RATE
    assert metadata['windows']['512']['bin_hz'] == RATE/512


def test_band_amplitude_is_root_sum_square_and_segment_mean_is_power():
    samples = pcm()
    samples[:] = .1*np.sin(2*np.pi*440*np.arange(len(samples))/RATE)
    metadata, arrays = analyze_window(samples, RATE, 1., 69, [], sizes=(2048,))
    row = band(metadata, 2048, 69)
    expected = np.sqrt(np.square(arrays['magnitude_2048'][row['bins']]).sum(axis=0))
    np.testing.assert_array_equal(arrays[row['array_key']], expected)
    times = arrays['times_2048']
    support_start, support_end = times - 1024/RATE, times + 1024/RATE
    mask = (support_start >= 1.) & (support_end <= 1.15)
    assert row['segments']['attack']['count'] == int(mask.sum())
    assert row['segments']['attack']['mean_power'] == pytest.approx(float(np.mean(expected[mask]**2)))


@pytest.mark.parametrize('impulse_time', [1., .975])
def test_cross_window_attack_leakage_never_enters_pre_statistics(impulse_time):
    samples = pcm()
    samples[round(impulse_time*RATE)] = 1.
    metadata, arrays = analyze_window(samples, RATE, 1., 69, [], sizes=(2048,))
    row = band(metadata, 2048, 69)
    assert row['segments']['pre']['count'] > 0
    assert row['segments']['pre']['mean_power'] == 0.
    assert row['attack_pre_db'] is None
    centers = arrays['times_2048']
    center_only_pre = (centers >= .82) & (centers <= .97)
    # Centers in pre can still see a later pulse; whole-window support excludes them.
    if impulse_time == .975:
        assert (arrays[row['array_key']][center_only_pre] > 0).any()


def test_8192_frames_do_not_claim_short_segment_statistics():
    samples = np.ones(2*RATE, dtype=float)
    metadata, arrays = analyze_window(samples, RATE, 1., 69, [57], sizes=(8192,))
    window = metadata['windows']['8192']
    assert window['frame_count'] > 0
    assert window['segment_frame_counts'] == dict(pre=0, attack=0, late=0)
    for row in window['bands']:
        assert row['segments'] == {name: dict(count=0, mean_power=None) for name in ('pre', 'attack', 'late')}
        assert row['attack_pre_db'] is None
    assert window['correlations'][0] == dict(target_pitch=69, lower_pitch=57,
        segment='attack', count=0, correlation=None)
    assert arrays['magnitude_8192'].shape[1] == window['frame_count']


def test_silence_has_no_correlation_or_infinite_ratio_but_zero_measured_power():
    metadata, _ = analyze_window(pcm(), RATE, 1., 69, [57])
    json.dumps(metadata, allow_nan=False)
    for size in (512, 2048):
        for row in metadata['windows'][str(size)]['bands']:
            assert row['attack_pre_db'] is None
            assert all(value['count'] > 0 and value['mean_power'] == 0 for value in row['segments'].values())
        assert metadata['windows'][str(size)]['correlations'][0]['correlation'] is None


def test_same_onset_octaves_share_all_target_partials_including_lower_above_eight():
    samples = pcm()
    t = np.arange(len(samples))/RATE
    samples[:] = np.where(t >= 1., .2*np.sin(2*np.pi*440*t)+.2*np.sin(2*np.pi*220*t), 0.)
    metadata, _ = analyze_window(samples, RATE, 1., 69, [57], sizes=(2048,))
    for partial in range(1, 9):
        row = band(metadata, 2048, 69, partial)
        overlap = row['lower_overlap'][0]
        assert overlap['shared_bin_fraction'] == 1. and overlap['shares_bin']
        assert any(item['partial'] == 2*partial for item in overlap['overlapping_partials'])
    assert not metadata['independent_attack_established']


def test_target_eighth_partial_compares_to_lower_sixty_fourth_partial():
    metadata, _ = analyze_window(pcm(), RATE, 1., 69, [33], sizes=(2048,))
    overlap = band(metadata, 2048, 69, 8)['lower_overlap'][0]
    assert overlap['shared_bin_fraction'] == 1.
    assert overlap['checked_partials'] > 64
    assert any(item['partial'] == 64 for item in overlap['overlapping_partials'])


def test_low_frequency_neighbor_semitone_bin_alias_is_reported():
    metadata, _ = analyze_window(pcm(), RATE, 1., 40, [39], sizes=(512,))
    row = band(metadata, 512, 40)
    assert row['bins'] and 0 not in row['bins']
    neighbor = next(n for n in row['neighbor_semitone_overlap'] if n['semitones'] == -1)
    assert neighbor['shares_bin'] and neighbor['shared_bins']
    overlap = row['lower_overlap'][0]
    assert overlap['shares_bin'] and any(item['partial'] == 1 for item in overlap['overlapping_partials'])


def test_dc_is_excluded_and_empty_frequency_support_is_not_silence_evidence():
    metadata, arrays = analyze_window(pcm(), RATE, 1., 0, [], sizes=(512,))
    row = band(metadata, 512, 0)
    assert row['bins'] == [] and row['frequency_supported'] is False
    assert row['segments']['attack']['count'] > 0
    assert row['segments']['attack']['mean_power'] is None
    assert row['attack_pre_db'] is None
    assert (arrays[row['array_key']] == 0).all()


def test_only_partial_centers_strictly_below_nyquist_are_saved():
    metadata, arrays = analyze_window(np.zeros(2*1760, float), 1760, 1., 69, [57], sizes=(128,), hop=16)
    bands = metadata['windows']['128']['bands']
    assert [row['partial'] for row in bands if row['pitch'] == 69] == [1]
    assert [row['partial'] for row in bands if row['pitch'] == 57] == [1, 2, 3]
    assert 'band_128_69_2' not in arrays
    assert all(row['center_hz'] < 880 for row in bands)


@pytest.mark.parametrize('shift', [4000, 4000000])
def test_time_translation_preserves_spectrum_support_and_segment_statistics(shift):
    rate = 20000
    original = np.random.default_rng(1).normal(0, .1, 40000)
    moved = np.concatenate((np.zeros(shift), original))
    a, x = analyze_window(original, rate, 1., 69, [57], sizes=(512, 2048), hop=40)
    b, y = analyze_window(moved, rate, 1.+shift/rate, 69, [57], sizes=(512, 2048), hop=40)
    assert b['extraction_samples'] == [value+shift for value in a['extraction_samples']]
    for size in (512, 2048):
        np.testing.assert_allclose(y[f'times_{size}'], x[f'times_{size}']+shift/rate, atol=1e-14)
        np.testing.assert_allclose(y[f'magnitude_{size}'], x[f'magnitude_{size}'], atol=1e-14)
        for left, right in zip(a['windows'][str(size)]['bands'], b['windows'][str(size)]['bands']):
            for segment in ('pre', 'attack', 'late'):
                assert left['segments'][segment]['count'] == right['segments'][segment]['count']
                assert left['segments'][segment]['mean_power'] == pytest.approx(right['segments'][segment]['mean_power'])


@pytest.mark.parametrize('onset_sample', [2400, 2401, 2399])
def test_support_includes_exact_boundaries_but_rejects_crossing_by_one_sample(onset_sample):
    rate, size = 2000, 100
    samples = np.random.default_rng(10).normal(0, .1, 5000)
    metadata, arrays = analyze_window(samples, rate, onset_sample/rate, 69, [], sizes=(size,), hop=1)
    starts = np.rint(arrays['times_100']*rate - size/2).astype(int)
    row = band(metadata, size, 69)
    envelope = arrays[row['array_key']]
    for name, offsets in {'pre': (-360, -60), 'attack': (0, 300), 'late': (300, 700)}.items():
        first, last = onset_sample+offsets[0], onset_sample+offsets[1]-size
        mask = (starts >= first) & (starts <= last)
        assert all(sample in starts for sample in (first-1, first, last, last+1))
        assert row['segments'][name]['count'] == last-first+1
        assert row['segments'][name]['mean_power'] == pytest.approx(
            float(np.mean(envelope[mask]**2)), rel=1e-13, abs=0)


@pytest.mark.parametrize('onset', [0., .01, .099])
def test_short_or_edge_audio_is_never_zero_padded(onset):
    metadata, arrays = analyze_window(pcm(.1), RATE, onset, 69, [], sizes=(512, 8192))
    assert metadata['extraction_samples'] == [0, 2205]
    assert arrays['times_8192'].shape == (0,)
    assert arrays['magnitude_8192'].shape == (4097, 0)
    times = arrays['times_512']
    assert (times-256/RATE >= 0).all()
    assert (times+256/RATE <= .1).all()
    assert metadata['windows']['8192']['first_frame_support_samples'] is None


def test_at_least_three_whole_attack_frames_are_required_for_correlation():
    samples = np.random.default_rng(3).normal(0, .1, 4000)
    metadata, _ = analyze_window(samples, 2000, 1., 69, [57], sizes=(128,), hop=128)
    row = metadata['windows']['128']['correlations'][0]
    assert row['count'] < 3 and row['correlation'] is None


def test_reported_correlation_matches_full_support_amplitude_envelopes():
    samples = np.random.default_rng(4).normal(0, .1, 2*RATE)
    metadata, arrays = analyze_window(samples, RATE, 1., 69, [57], sizes=(512,))
    starts = np.rint(arrays['times_512']*RATE-256).astype(int)
    mask = (starts/RATE >= 1.) & ((starts+512)/RATE <= 1.15)
    expected = np.corrcoef(arrays['band_512_69_1'][mask], arrays['band_512_57_1'][mask])[0, 1]
    row = metadata['windows']['512']['correlations'][0]
    assert row['count'] == int(mask.sum()) and row['count'] >= 3
    assert row['correlation'] == pytest.approx(expected)


@pytest.mark.parametrize('samples', [[], [1, 2], [[0.]], [np.nan], [np.inf], [-np.inf],
                                    [1+2j], np.asarray([0.], dtype=object)])
def test_invalid_pcm_is_rejected(samples):
    with pytest.raises(ValueError, match='Samples'):
        analyze_window(samples, RATE, 0., 69, [])


@pytest.mark.parametrize('sample_rate', [0, -1, True, 22050., np.nan, '22050'])
def test_invalid_sample_rate_is_rejected(sample_rate):
    with pytest.raises(ValueError, match='Sample rate'):
        analyze_window(pcm(), sample_rate, 1., 69, [])


@pytest.mark.parametrize('onset', [-.1, 2., np.nan, np.inf, True, '1'])
def test_invalid_onset_is_rejected(onset):
    with pytest.raises(ValueError, match='Onset'):
        analyze_window(pcm(), RATE, onset, 69, [])


@pytest.mark.parametrize('pitch', [-1, 128, 60., True, np.nan, '60'])
def test_invalid_target_pitch_is_rejected(pitch):
    with pytest.raises(ValueError, match='Pitch'):
        analyze_window(pcm(), RATE, 1., pitch, [])


@pytest.mark.parametrize('lowers', [[69], [70], [-1], [128], [57.], [True], [np.nan], ['57'], None])
def test_invalid_lower_pitches_are_rejected(lowers):
    with pytest.raises(ValueError):
        analyze_window(pcm(), RATE, 1., 69, lowers)


@pytest.mark.parametrize('sizes', [(), (512, 512), (1,), (0,), (-1,), (512.,), (True,), None])
def test_invalid_window_sizes_are_rejected(sizes):
    with pytest.raises(ValueError):
        analyze_window(pcm(), RATE, 1., 69, [], sizes=sizes)


@pytest.mark.parametrize('hop', [0, -1, 513, True, 44., np.nan])
def test_invalid_hop_is_rejected(hop):
    with pytest.raises(ValueError):
        analyze_window(pcm(), RATE, 1., 69, [], sizes=(512,), hop=hop)


def test_target_at_or_above_nyquist_is_rejected():
    with pytest.raises(ValueError, match='Nyquist'):
        analyze_window(np.zeros(1760, float), 880, 1., 69, [])
