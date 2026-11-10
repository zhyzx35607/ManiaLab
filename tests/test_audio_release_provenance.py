"""Release provenance must survive the generic detector and both LN selectors."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from manialab import audio
from manialab.arrange import choose_holds, parameters
from manialab.domain import Note, note_from
from manialab.expression import expression_evidence, propose_expression


def activations(end=96, peaks=(5,)):
    times = np.arange(96) * audio.FRAME / audio.SR
    frames = np.zeros((96, 88), dtype=np.float32)
    onsets = np.zeros_like(frames)
    frames[5:end, 39] = .8
    onsets[list(peaks), 39] = 1.
    return frames, onsets, times, 96 * audio.FRAME / audio.SR


@pytest.mark.parametrize('end,peaks,expected', [
    (96, (5,), [(58, 1115, 'input_boundary')]),
    (93, (5,), [(58, 1080, 'input_boundary')]),
    (91, (5,), [(58, 1057, 'frame_decay')]),
    (40, (5,), [(58, 464, 'frame_decay')]),
    (80, (5, 40), [(58, 464, 'same_pitch_reattack'), (464, 929, 'frame_decay')]),
])
def test_decoder_records_why_scan_ended_without_changing_geometry(end, peaks, expected):
    notes = audio.pitched_notes(*activations(end, peaks))
    assert [(n.start, n.end, n.release_reason) for n in notes] == expected
    assert all(n.pitch == 60 and n.source == 'pitch' for n in notes)


def test_decay_before_a_later_reattack_is_not_reported_as_reattack():
    frames, onsets, times, duration = activations(80, (5, 60))
    frames[30:60, 39] = 0.
    notes = audio.pitched_notes(frames, onsets, times, duration)
    assert [n.release_reason for n in notes] == ['frame_decay', 'frame_decay']
    assert notes[0].end == round(times[30] * 1000) < notes[1].start


@pytest.mark.parametrize('second_attack', [142, 143, 284])
def test_window_seam_tail_cannot_pass_next_same_pitch_attack(second_attack):
    # Match infer()'s real retained-window clock, including its short seam.
    times = np.concatenate([(window * (audio.WINDOW - audio.OVERLAP)
                             + np.arange(142) * audio.FRAME) / audio.SR
                            for window in range(3)])
    frames = np.zeros((len(times), 88), dtype=np.float32)
    onsets = np.zeros_like(frames)
    first_attack = second_attack - 12
    frames[first_attack:second_attack + 20, 39] = .8
    onsets[[first_attack, second_attack], 39] = 1.
    # A different pitch is allowed to continue across the same boundary.
    frames[first_attack:second_attack + 20, 40] = .8
    onsets[first_attack, 40] = 1.
    notes = audio.pitched_notes(frames, onsets, times, times[-1] + audio.FRAME / audio.SR)
    same = [n for n in notes if n.pitch == 60]
    assert len(same) == 2
    assert same[0].end == same[1].start == round(times[second_attack] * 1000)
    assert same[0].release_reason == 'same_pitch_reattack'
    assert next(n for n in notes if n.pitch == 61).end > same[1].start


def test_suppressed_seam_attack_does_not_shorten_continuing_note():
    times = np.concatenate([(window * (audio.WINDOW - audio.OVERLAP)
                             + np.arange(142) * audio.FRAME) / audio.SR
                            for window in range(2)])
    frames = np.zeros((len(times), 88), dtype=np.float32)
    onsets = np.zeros_like(frames)
    frames[130:180, 39] = .8
    onsets[[130, 142], 39] = 1.
    notes = audio.pitched_notes(frames, onsets, times, times[-1] + audio.FRAME / audio.SR,
        suppressed=[dict(pitch=60, time_ms=round(times[142] * 1000))])
    assert len(notes) == 1 and notes[0].end == round(times[180] * 1000)
    assert notes[0].release_reason == 'frame_decay'


@pytest.mark.parametrize('end', [80, 96])
def test_shorter_audio_cannot_borrow_future_decay_from_a_longer_cache(end):
    frames, onsets, times, _ = activations(end)
    with pytest.raises(ValueError, match='时间戳超出音频时长'):
        audio.pitched_notes(frames, onsets, times, .65)


@pytest.mark.parametrize('trim_ms', [.1814, 5.])
def test_final_frame_cell_may_extend_past_eof_without_invalidating_input(trim_ms):
    frames, onsets, times, duration = activations()
    duration -= trim_ms / 1000
    notes = audio.pitched_notes(frames, onsets, times, duration)
    assert notes[0].end == round(duration * 1000)
    assert notes[0].release_reason == 'input_boundary'


@pytest.mark.parametrize('silence_at', [None, .7])
def test_waveform_trimming_and_attack_fusion_keep_unresolved_provenance(silence_at):
    frames, onsets, times, duration = activations()
    original = replace(audio.pitched_notes(frames, onsets, times, duration)[0],
                       voice='other', instrument_id='test:class:48')
    wave = np.full(round(duration * audio.SR), .2, dtype=np.float32)
    if silence_at is not None:
        wave[round(silence_at * audio.SR):] = 0.
    notes, _ = audio.source_evidence([original], wave, wave)
    notes, report = audio.refine_releases(notes, wave)
    notes, _, fused = audio.fuse_attacks(notes, [Note(original.start + 10, original.start + 10)])
    assert len(notes) == fused == 1
    assert notes[0].start == original.start + 10
    assert (report['shortened'] == 1) == (silence_at is not None)
    assert notes[0].end <= original.end
    assert notes[0].release_reason == 'input_boundary'
    assert notes[0].instrument_id == original.instrument_id
    assert note_from(notes[0].data()) == notes[0]


@pytest.mark.parametrize('voice', ['mix', 'bass', 'guitar', 'vocals', 'other', 'piano'])
@pytest.mark.parametrize('end,should_hold', [(96, False), (93, False), (40, True)])
def test_generic_release_eligibility_in_global_and_section_expression(voice, end, should_hold):
    note = replace(audio.pitched_notes(*activations(end))[0], voice=voice)
    p = parameters({'ln_ratio': 1.})
    chosen, _ = choose_holds([note], [p], p)
    assert chosen[0].hold == should_hold
    analysis = dict(duration=2, candidates=[note.data()],
                    structure=dict(segments=[dict(start_ms=0, end_ms=2000)]))
    plan = propose_expression(analysis, p)
    plan['sections'][0]['voices'][voice]['mode'] = 'sustain'
    chosen, _, trace = expression_evidence(analysis, plan, p)
    assert len(chosen) == 1 and chosen[0].hold == should_hold
    assert chosen[0].start == note.start
    assert trace[note.id]['reason'] == ('supported_sustain' if should_hold else 'unobserved_release')
    assert analysis['candidates'][0] == note.data()


def test_cached_analysis_reconstructs_generic_release_metadata_without_inference(tmp_path, monkeypatch):
    from manialab.stems import separation_identity
    frames, onsets, times, duration = activations()
    samples = np.full(round(duration * audio.SR), .2, dtype=np.float32)
    voices = ['bass', 'guitar', 'vocals', 'other', 'piano']
    separation = dict(source='provided', voices=voices, audio_sha256='audio',
                      set_sha256='stems', model_sha256='stems', cached=True)
    for voice in voices:
        np.save(tmp_path / f'{voice}.npy', samples)
        np.savez_compressed(tmp_path / f'{voice}-activations.npz',
                            frames=frames, onsets=onsets, times=times, version=3,
                            audio_sha256='audio', separation_sha256=separation_identity(separation))

    class Sources:
        def separate(self, *args):
            return tmp_path, separation

    monkeypatch.setattr(audio, 'decode', lambda *args: samples)
    def unused(*args):
        raise AssertionError('Existing activations must be reused')
    monkeypatch.setattr(audio, 'infer', unused)
    result = audio.analyze_deep(Path('mix.wav'), tmp_path, Path('unused.onnx'), parameters(), Sources())
    assert result['cached']
    assert {n['voice'] for n in result['candidates']} == set(voices)
    assert all(n['release_reason'] == 'input_boundary' for n in result['candidates'])
    assert all(n['start'] == 58 and n['end'] == 1115 for n in result['candidates'])
