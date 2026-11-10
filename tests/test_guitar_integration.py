"""Source routing, weak-residual filtering and LN policy without neural work."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from manialab import audio, guitar
from manialab.arrange import parameters
from manialab.domain import Note, UNOBSERVED_RELEASES


@pytest.mark.parametrize('selected,release', [(False, False), (True, False), (True, True)])
def test_only_explicit_guitar_uses_fl_and_keeps_source_energy_checks(tmp_path, monkeypatch, selected, release):
    mix = np.full(audio.SR, .2, dtype=np.float32)
    roles = ('guitar', 'other', 'piano')
    for index, voice in enumerate(roles):
        signal = np.full(audio.SR, .1 * (index + 1), dtype=np.float32)
        if voice == 'guitar':
            signal[int(.45 * audio.SR):int(.65 * audio.SR)] = 0
        np.save(tmp_path / (voice + '.npy'), signal)
    separation = dict(source='provided', voices=list(roles), audio_sha256='mix', set_sha256='stems', cached=True)
    class Sources:
        def separate(self, *args):
            return tmp_path, separation
    calls, generic, refinements, fusions, bleed = [], [], [], [], []
    class Specialist:
        def analyze(self, signal, rate, cache, identity, **kwargs):
            calls.append(kwargs)
            assert cache == tmp_path / 'guitar-fl' and rate == audio.SR
            np.testing.assert_array_equal(signal, np.load(tmp_path / 'guitar.npy'))
            events = [dict(onset=start, offset=.7, pitch=60, velocity=90, onset_score=.31,
                           release_reason='frame_decay', release_observed=True) for start in (.1, .5)]
            notes, trace = guitar.candidates(events, allow_predicted_releases=kwargs['allow_predicted_releases'])
            return notes, dict(cached=True, candidate_trace=trace, source_semantics='unit_weight_not_confidence')
    def infer(signal, *args):
        generic.append(float(signal[0]))
        return np.zeros((10, 88)), np.zeros((10, 88)), np.arange(10) / 10
    def refine(notes, *args):
        refinements.append(notes)
        return notes, {}
    def fuse(notes, *args):
        fusions.append(notes)
        return [replace(n, start=n.start + 1) for n in notes], [], 0
    def reject(notes, *args):
        bleed.append({n.voice for n in notes})
        return notes, []
    monkeypatch.setattr(audio, 'decode', lambda *args: mix)
    monkeypatch.setattr(audio, 'infer', infer)
    monkeypatch.setattr(audio, 'pitched_notes', lambda *args, **kwargs: [Note(100, 700, source='pitch', id='general')])
    monkeypatch.setattr(audio, 'refine_releases', refine)
    monkeypatch.setattr(audio, 'fuse_attacks', fuse)
    monkeypatch.setattr(audio, 'reject_source_bleed', reject)
    monkeypatch.setattr('manialab.structure.detect_structure', lambda *args: dict(segments=[dict(start_ms=0, end_ms=1000)]))
    result = audio.analyze_deep(Path('mix'), tmp_path, Path('unused'), parameters(), Sources(),
        guitar_detector=Specialist() if selected else None, allow_guitar_predicted_releases=release)
    assert len(generic) == len(refinements) == len(fusions) == (2 if selected else 3)
    assert result['guitar_engine'] == ('fl' if selected else 'general')
    assert result['instrument_engine'] == 'general' and result['piano_engine'] == 'general'
    assert result['allow_guitar_predicted_releases'] is release
    guitar_notes = [n for n in result['candidates'] if n['voice'] == 'guitar']
    if selected:
        assert len(calls) == 1 and bleed == [{'other', 'piano'}]
        assert [(n['start'], n['end']) for n in guitar_notes] == [(100, 700)]
        assert guitar_notes[0]['strength'] == 1.
        assert (guitar_notes[0]['release_reason'] == 'model_predicted_offset') is release
        summary = next(v for v in result['voices'] if v['voice'] == 'guitar')
        assert summary['detector'] == 'guitar-fl' and summary['rejected_low_energy_pitch'] == 1
        assert not summary['cross_source_bleed_checked']
        assert summary['release_refinement']['source_energy_filter']['rejected_ids'] == ['guitar:fl:1']
        assert result['source_bleed_policy']['excluded_unit_weight_voices'] == ['guitar']
    else:
        assert not calls and bleed == [set(roles)]
        assert guitar_notes[0]['start'] == 101
        assert result['source_bleed_policy']['excluded_unit_weight_voices'] == []


def test_invalid_guitar_tail_request_fails_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, 'decode', lambda *args: pytest.fail('Invalid options reached decoding'))
    for value in (True, 'true', 1, None):
        with pytest.raises(ValueError, match='吉他模型音尾'):
            audio.analyze_deep(Path('mix'), tmp_path, Path('unused'), parameters(), None,
                allow_guitar_predicted_releases=value)


def test_guitar_cannot_run_on_other_as_a_substitute(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, 'decode', lambda *args: np.zeros(audio.SR))
    class Sources:
        def separate(self, *args):
            return tmp_path, dict(source='provided', voices=['other'], audio_sha256='mix', set_sha256='stems')
    with pytest.raises(ValueError, match='吉他声轨'):
        audio.analyze_deep(Path('mix'), tmp_path, Path('unused'), parameters(), Sources(), guitar_detector=object())


@pytest.mark.parametrize('release', [False, True])
def test_ln_expression_cannot_use_unknown_guitar_release_even_at_full_ln_ratio(release):
    from manialab.expression import propose_expression, expression_evidence
    raw = [dict(onset=1., offset=5., pitch=60, velocity=100, onset_score=.31,
                release_reason='reattack', release_observed=False)]
    notes, trace = guitar.candidates(raw, allow_predicted_releases=release)
    assert notes[0].release_reason in UNOBSERVED_RELEASES
    analysis = dict(duration=10, candidates=[n.data() for n in notes],
                    structure=dict(segments=[dict(start_ms=0, end_ms=10000)]))
    prefs = parameters({'ln_ratio': 1.})
    plan = propose_expression(analysis, prefs)
    plan['sections'][0]['voices']['guitar']['mode'] = 'sustain'
    selected, _, decisions = expression_evidence(analysis, plan, prefs)
    assert len(selected) == 1 and selected[0].start == selected[0].end == 1000
    assert decisions[notes[0].id]['reason'] == 'unobserved_release'
    assert trace[0]['raw_event']['offset'] == 5.
