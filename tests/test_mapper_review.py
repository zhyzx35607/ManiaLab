import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from manialab.domain import Note, export_osu
from scripts.build_mapper_song_review import read_probe
from scripts.probe_mapper_adapter import sha, write_review


def test_model_only_review_keeps_source_clocks_and_labels(tmp_path):
    # Resampled sources must produce clicks at the same absolute song time.
    mix, stem = tmp_path/'mix.wav', tmp_path/'piano.wav'
    wavfile.write(mix, 22050, np.zeros(44100, dtype=np.int16))
    wavfile.write(stem, 44100, np.zeros(88200, dtype=np.int16))
    output = tmp_path/'review'
    results = {'adapter': {'notes': [Note(625, 1075, 5).data()]}}
    original = json.dumps(results)
    write_review(output, mix, 500, 1500, results, labels={'adapter': '实验预测 </script>'},
        sources={'piano': dict(label='分离钢琴', path=stem)}, notice='没有人工答案', keys=6)
    for prefix in ('review', 'review-source-piano'):
        metadata = json.loads((output/f'{prefix}-adapter.json').read_text())
        assert metadata['samples'] == metadata['sample_rate']
        for event in metadata['events']:
            clock = (metadata['first_sample'] + event['sample']) / metadata['sample_rate'] * 1000
            assert abs(clock-event['time_ms']) < .03
    html = (output/'review.html').read_text(encoding='utf-8')
    assert 'review-reference.wav' not in html
    assert '实验预测 </script>' not in html
    data = json.loads(html.split('const data=', 1)[1].split('; const audio=', 1)[0])
    assert data['notice'] == '没有人工答案'
    assert [chart['label'] for chart in data['charts']] == ['实验预测 </script>']
    assert json.dumps(results) == original


def test_review_rejects_short_source_instead_of_misaligning_audio(tmp_path):
    source = tmp_path/'short.wav'
    wavfile.write(source, 8000, np.zeros(8000, dtype=np.int16))
    with pytest.raises(ValueError, match='same complete time interval'):
        write_review(tmp_path/'review', source, 500, 1500, {'adapter': dict(notes=[])})


def test_song_review_checks_raw_chart_and_audio_provenance(tmp_path):
    root = Path(__file__).resolve().parents[1]
    raw = tmp_path/'model.osu'
    raw.write_text(export_osu([Note(900, 1300, 0), Note(1600, 1600, 1)], 10,
        root/'vendor/mapperatorinator/template.osu'), encoding='utf-8')
    manifest = tmp_path/'result.json'
    manifest.write_text(json.dumps(dict(context=dict(audio_sha256='song-hash', start_ms=1000, seconds=1),
        raw_file=raw.name, raw_sha256=sha(raw))), encoding='utf-8')
    result = read_probe(manifest, 'song-hash')
    assert len(result['notes']) == 2  # Keep the LN crossing the listening boundary.
    with pytest.raises(ValueError, match='supplied original song'):
        read_probe(manifest, 'other-song')
    raw.write_text(raw.read_text(encoding='utf-8')+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='output changed'):
        read_probe(manifest, 'song-hash')
