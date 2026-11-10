import numpy as np
import json
import pytest

from manialab.domain import Note
from manialab.stem_release_evidence import observe, StemReleaseContext, ROLES


def test_early_observations_do_not_use_future_peak_normalisation():
    rate = 22050; t = np.arange(rate*3)/rate
    signal = (.01*np.sin(2*np.pi*440*t)).astype(np.float32)
    a = observe(signal, rate)
    signal[rate*2:] *= 50
    b = observe(signal, rate)
    for key in ('level', 'flux', 'low', 'high'):
        np.testing.assert_array_equal(a[key][:60], b[key][:60])


def test_release_evidence_uses_physical_audio_clock_and_off_attack_tail():
    rate = 22050; t = np.arange(rate*3)/rate
    signal = (.2*np.sin(2*np.pi*440*t)*(t < .5225)).astype(np.float32)
    observation = observe(signal, rate)
    evidence = {role:observation for role in ROLES}
    context = StemReleaseContext([Note(1000, 1000, 0), Note(2000, 2000, 0)], 6,
        ['0,500,4,2,1,60,1,0'], .35, evidence, audio_start_ms=1000)
    index = list(context.options(0)).index(1500)
    features = context.features(0)
    assert features[index, context.feature_names.index('other_tail_drop')] > .1
    assert features[index, context.feature_names.index('release_at_attack')] == 0
    assert features.shape[1] == len(context.feature_names) and np.isfinite(features).all()


def test_crop_loader_keeps_absolute_clock_and_checks_separation_identity(tmp_path):
    from scripts.paired_stem_inputs import prepare_observations, load_observations
    from scripts.probe_mapper_adapter import sha
    audio=tmp_path/'audio.mp3'; chart=tmp_path/'reference.osu'
    audio.write_bytes(b'original audio identity fixture'); chart.write_text('chart identity fixture')
    record=dict(audio=str(audio), chart=str(chart), audio_sha256=sha(audio), chart_sha256=sha(chart))
    stems=tmp_path/'deep/stems'; stems.mkdir(parents=True)
    meta=dict(audio_sha256='cropped-input-identity',model_sha256='model-fixture',sample_rate=22050,frames=22050*8)
    analysis=dict(duration=8.,separation=meta,research_clip=dict(start_ms=10000,end_ms=18000,
        excluded_edge_ms=3000,original_audio_sha256=sha(audio),chart_sha256=sha(chart)))
    (stems.parent/'analysis.json').write_text(json.dumps(analysis))
    (stems/'separation.json').write_text(json.dumps(meta))
    for role in ROLES: np.save(stems/(role+'.npy'), np.zeros(meta['frames'],np.float32))
    observed, identity=prepare_observations(record,tmp_path/'observations')
    assert identity['audio_start_ms']==10000 and identity['supported_bounds']==[13000,15000]
    frozen=load_observations(tmp_path/'observations',identity)
    np.testing.assert_array_equal(frozen['drums']['level'],observed['drums']['level'])
    (stems/'separation.json').write_text(json.dumps(meta|dict(frames=1)))
    with pytest.raises(ValueError,match='metadata disagrees'):
        prepare_observations(record,tmp_path/'invalid')
