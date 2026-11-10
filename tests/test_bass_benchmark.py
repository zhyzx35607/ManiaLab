import pytest

from scripts.prepare_bass_benchmark import parse_labels


def xml(*, pitch='33', onset='0', offset='.2', expression='DN'):
    return f'''<instrumentRecording><globalParameter><audioFileName>001.wav</audioFileName>
    <instrument>EBSS</instrument></globalParameter><transcription><event><pitch>{pitch}</pitch>
    <onsetSec>{onset}</onsetSec><offsetSec>{offset}</offsetSec><fretNumber>5</fretNumber>
    <stringNumber>1</stringNumber><excitationStyle>SP</excitationStyle>
    <expressionStyle>{expression}</expressionStyle></event></transcription></instrumentRecording>'''


@pytest.mark.parametrize('expression', ['DN', 'HA', 'SL', 'BE', 'VI', 'NO'])
def test_bass_techniques_preserve_nominal_pitch_and_zero_time_onset(expression):
    row, = parse_labels(xml(expression=expression), '001.wav')
    assert row['onset'] == 0 and row['offset'] == .2 and row['pitch'] == 33
    assert row['program'] == -1 and row['expression_style'] == expression
    assert row['plucking_style'] == 'SP'


@pytest.mark.parametrize('values', [dict(pitch='nan'), dict(pitch='33.5'), dict(onset='-.1'),
    dict(offset='0'), dict(expression='unknown')])
def test_invalid_bass_annotation_is_not_silently_dropped(values):
    with pytest.raises(ValueError):
        parse_labels(xml(**values), '001.wav')


def test_bass_annotation_must_name_the_actual_recording():
    with pytest.raises(ValueError, match='mismatch'):
        parse_labels(xml(), '002.wav')
