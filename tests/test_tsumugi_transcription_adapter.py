from dataclasses import asdict, dataclass

import pytest

from scripts.research_tsumugi_transcription import convert_notes


@dataclass
class Note:
    instrument_id: int = 0
    pitch: int = 60
    start_sample: int = 100
    end_sample: int = 300
    slot_index: int = 0
    has_onset: bool = True
    has_offset: bool = True
    instrument_candidates: tuple = (0,)


def convert(notes):
    return convert_notes([asdict(n) for n in notes], sample_rate=1000, total_samples=1000,
                         classes=['strings', 'drums'])


def test_forced_offset_is_not_accepted_as_natural_ln_release():
    events = convert([Note(has_offset=True), Note(start_sample=600, end_sample=1000, has_offset=False)])
    assert [e['release_reason'] for e in events] == ['unverified_stitched_boundary']*2
    assert events[0]['offset'] == .3
    assert events[0]['strength_semantics'] == 'unit_weight_not_confidence'


def test_repeated_overlapping_and_multiple_slot_events_are_not_midi_pruned():
    events = convert([Note(), Note(start_sample=125, end_sample=320), Note(slot_index=1)])
    assert len(events) == 3
    assert [e['onset'] for e in events] == [.1, .125, .1]
    assert [e['offset'] for e in events] == [.3, .32, .3]
    assert [e['slot_index'] for e in events] == [0, 0, 1]


def test_wrong_drum_is_preserved_and_broad_class_does_not_claim_exact_program():
    events = convert([Note(instrument_id=1)])
    assert events[0]['is_drum']
    assert events[0]['program'] == -1
    assert events[0]['instrument_class'] == 'drums'


@pytest.mark.parametrize('note', [Note(start_sample=-1), Note(end_sample=1001), Note(end_sample=50),
    Note(pitch=128), Note(instrument_id=2), Note(start_sample=100.5)])
def test_invalid_model_clock_or_labels_fail_instead_of_silent_filtering(note):
    with pytest.raises(ValueError):
        convert([note])
