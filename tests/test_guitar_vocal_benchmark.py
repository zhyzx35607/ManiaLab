import math

import pytest

from scripts.prepare_guitar_vocal_benchmark import guitar_labels, select_guitar_names, vocal_labels, STYLES


def test_fixed_guitar_selection_covers_players_styles_and_modes_independent_of_zip_order():
    names = [f'{p:02}_{style}{i}-100-C_{mode}.jams' for p in range(6)
             for style in STYLES for mode in ('comp', 'solo') for i in (2, 1)]
    selected = select_guitar_names(names)
    assert len(selected) == len(set(selected)) == 60
    assert all('1-100' in n for n in selected)
    assert selected == select_guitar_names(list(reversed(names)))
    with pytest.raises(ValueError, match='Missing'):
        select_guitar_names([n for n in names if not n.startswith('05_Rock')])


def test_guitar_note_strings_keep_unison_layers_and_original_pitch_time():
    annotations = [dict(namespace='note_midi', annotation_metadata={'data_source':str(s)},
                        data=[dict(time=.125, duration=.875, value=60.2)]) for s in range(6)]
    rows = guitar_labels({'annotations':annotations})
    assert len(rows) == 6 and {r['part'] for r in rows} == set(range(6))
    assert all(r['onset'] == .125 and r['offset'] == 1 and r['pitch'] == 60
               and r['midi_continuous'] == 60.2 for r in rows)
    with pytest.raises(ValueError, match='each guitar string'):
        guitar_labels({'annotations':annotations[:-1]})


def test_singing_csv_preserves_frequency_and_does_not_retime_or_quantize_intervals():
    rows = vocal_labels('0.125,440,0.875\n\n1.123,445,0.456\n')
    assert rows[0]['onset'] == .125 and rows[0]['offset'] == 1 and rows[0]['pitch'] == 69
    assert rows[1]['onset'] == 1.123 and math.isclose(rows[1]['offset'], 1.579)
    assert rows[1]['frequency_hz'] == 445 and rows[1]['midi_continuous'] > 69
    with pytest.raises(ValueError):
        vocal_labels('0.1,0,1')
    with pytest.raises(ValueError):
        vocal_labels('0.1,440,-1')
