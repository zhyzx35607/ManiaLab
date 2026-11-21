import hashlib
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile

from manialab import audio
from manialab.audio import SR, analyze, rhythmic_evidence
from manialab.arrange import parameters


def test_basic_pitch_inference_respects_cpu_thread_limits(tmp_path, monkeypatch):
    from manialab import audio

    captured = {}

    class Options:
        intra_op_num_threads = None
        inter_op_num_threads = None

    def session(model, sess_options, providers):
        captured.update(intra=sess_options.intra_op_num_threads,
                        inter=sess_options.inter_op_num_threads,
                        providers=providers)
        raise RuntimeError('captured configuration')

    model = tmp_path / 'model.onnx'
    model.write_bytes(b'synthetic model')
    monkeypatch.setattr(audio, '_session', None)
    monkeypatch.setattr(audio.ort, 'SessionOptions', Options)
    monkeypatch.setattr(audio.ort, 'InferenceSession', session)
    import pytest
    with pytest.raises(RuntimeError, match='captured configuration'):
        audio.infer(np.zeros(audio.SR, dtype=np.float32), model)

    assert captured == dict(intra=2, inter=1, providers=['CPUExecutionProvider'])


def test_basic_pitch_session_reloads_when_model_identity_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, '_session', None)
    monkeypatch.setattr(audio, '_session_model_identity', None)
    model_a = tmp_path / 'model-a.onnx'
    model_b = tmp_path / 'model-b.onnx'
    model_a.write_bytes(b'synthetic model A')
    model_b.write_bytes(b'synthetic model B')
    loaded = []

    class Options:
        intra_op_num_threads = None
        inter_op_num_threads = None

    class Session:
        def run(self, outputs, inputs):
            result = np.zeros((1, 172, 88), dtype=np.float32)
            return result, result

    def session(model, sess_options, providers):
        loaded.append(Path(model))
        return Session()

    monkeypatch.setattr(audio.ort, 'SessionOptions', Options)
    monkeypatch.setattr(audio.ort, 'InferenceSession', session)

    samples = np.zeros(audio.SR, dtype=np.float32)
    audio.infer(samples, model_a)
    audio.infer(samples, model_b)

    assert loaded == [model_a, model_b]


def test_stem_eof_rounding_never_changes_the_attack_clock():
    import pytest
    from manialab.audio import align_stem_samples
    samples=np.zeros(1000,dtype=np.float32);samples[91]=.8
    longer,delta=align_stem_samples(samples,1001)
    assert delta==1 and len(longer)==1001 and np.argmax(longer)==91
    assert np.array_equal(longer[:-1],samples) and longer[-1]==0
    shorter,delta=align_stem_samples(samples,999)
    assert delta==-1 and np.array_equal(shorter,samples[:-1])
    with pytest.raises(ValueError):align_stem_samples(samples,997)
    invalid=samples.copy();invalid[30]=np.nan
    with pytest.raises(ValueError):align_stem_samples(invalid,1000)


def test_audio_analysis_streams_identity_hash_and_reuses_cache(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic audio identity")
    model_a = tmp_path / "model-a.onnx"
    model_b = tmp_path / "model-b.onnx"
    model_a.write_bytes(b"synthetic model A")
    model_b.write_bytes(b"synthetic model B")
    inferred = []

    def fake_infer(*args, **kwargs):
        inferred.append(True)
        return (
            np.zeros((1, 88), dtype=np.float32),
            np.zeros((1, 88), dtype=np.float32),
            np.zeros(1, dtype=np.float64),
        )

    def reject_read_bytes(path):
        raise AssertionError(f"whole-file read attempted for {path}")

    monkeypatch.setattr(audio, "decode", lambda *args, **kwargs: np.zeros(SR, dtype=np.float32))
    monkeypatch.setattr(audio, "infer", fake_infer)
    monkeypatch.setattr(audio, "rhythmic_evidence", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(audio, "pitched_notes", lambda *args, **kwargs: [])
    monkeypatch.setattr(audio, "refine_releases", lambda notes, samples: (notes, {}))
    monkeypatch.setattr(audio, "fuse_attacks", lambda notes, attacks: (notes, [], 0))
    monkeypatch.setattr(
        "manialab.structure.detect_structure",
        lambda analysis, samples, sample_rate, **kwargs: {"segments": []},
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        reject_read_bytes,
    )

    first = analyze(source, tmp_path, model_a, parameters())
    second = analyze(source, tmp_path, model_a, parameters())
    third = analyze(source, tmp_path, model_b, parameters())
    fourth = analyze(source, tmp_path, model_b, parameters())

    assert not first["cached"] and second["cached"]
    assert not third["cached"] and fourth["cached"]
    assert len(inferred) == 2
    monkeypatch.setattr(audio, "BASIC_PITCH_INFERENCE_REVISION", 2)
    fifth = analyze(source, tmp_path, model_b, parameters())
    sixth = analyze(source, tmp_path, model_b, parameters())
    assert not fifth["cached"] and sixth["cached"]
    assert len(inferred) == 3


def test_audio_decode_and_cache_identity_use_the_same_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source_a, source_b = b"synthetic source A", b"synthetic source B"
    source.write_bytes(source_a)
    decoded = []
    inferred = []
    replaced = False

    def decode_snapshot(path, work, cancelled=None):
        nonlocal replaced
        path = Path(path)
        assert path != source and path.suffix == source.suffix
        payload = path.read_bytes()
        decoded.append(payload)
        if not replaced:
            source.write_bytes(source_b)
            replaced = True
        return np.full(SR, 0.1 if payload == source_a else 0.2, dtype=np.float32)

    def fake_infer(samples, *args, **kwargs):
        inferred.append(float(samples[0]))
        values = np.full((1, 88), samples[0], dtype=np.float32)
        return values, values.copy(), np.zeros(1, dtype=np.float64)

    monkeypatch.setattr(audio, "decode", decode_snapshot)
    monkeypatch.setattr(audio, "infer", fake_infer)
    monkeypatch.setattr(audio, "rhythmic_evidence", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(audio, "pitched_notes", lambda *args, **kwargs: [])
    monkeypatch.setattr(audio, "refine_releases", lambda notes, samples: (notes, {}))
    monkeypatch.setattr(audio, "fuse_attacks", lambda notes, attacks: (notes, [], 0))
    monkeypatch.setattr(
        "manialab.structure.detect_structure",
        lambda analysis, samples, sample_rate, **kwargs: {"segments": []},
    )

    model = tmp_path / "missing-model.onnx"
    first = analyze(source, tmp_path, model, parameters())
    assert first["audio_sha256"] == hashlib.sha256(source_a).hexdigest()
    assert not list(tmp_path.glob(".analysis-input-*"))

    second = analyze(source, tmp_path, model, parameters())
    assert second["audio_sha256"] == hashlib.sha256(source_b).hexdigest()
    assert decoded == [source_a, source_b]
    assert inferred == pytest.approx([0.1, 0.2])
    assert not first["cached"] and not second["cached"]
    assert not list(tmp_path.glob(".analysis-input-*"))


def test_audio_snapshot_is_removed_after_copy_cancellation_and_decode_error(tmp_path, monkeypatch):
    from manialab.audio import _audio_snapshot

    source = tmp_path / "large-source.wav"
    source.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    checks = 0

    def cancel_during_copy():
        nonlocal checks
        checks += 1
        return checks == 2

    with pytest.raises(InterruptedError, match="快照"):
        with _audio_snapshot(source, tmp_path, cancel_during_copy):
            pytest.fail("cancelled snapshot must not be yielded")
    assert not list(tmp_path.glob(".analysis-input-*"))

    monkeypatch.setattr(audio, "decode", lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic decode failure")))
    with pytest.raises(RuntimeError, match="synthetic decode failure"):
        analyze(source, tmp_path, tmp_path / "missing-model.onnx", parameters())
    assert not list(tmp_path.glob(".analysis-input-*"))


def test_audio_snapshot_preserves_body_error_when_cleanup_fails(tmp_path, monkeypatch):
    from manialab.audio import _audio_snapshot

    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic source")
    missing = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        with _audio_snapshot(missing, tmp_path, lambda: False):
            pytest.fail("missing source must not yield a snapshot")
    assert not list(tmp_path.glob(".analysis-input-*"))

    real_unlink = Path.unlink
    cleanup_targets = []

    def fail_snapshot_cleanup(path, *args, **kwargs):
        if path.name.startswith(".analysis-input-"):
            cleanup_targets.append(path)
            raise OSError("synthetic cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_snapshot_cleanup)
    body_error = RuntimeError("synthetic body failure")
    try:
        with pytest.raises(RuntimeError, match="synthetic body failure") as raised:
            with _audio_snapshot(source, tmp_path, lambda: False):
                raise body_error

        assert raised.value is body_error
        assert len(cleanup_targets) == 1
        assert any("synthetic cleanup failure" in note
                   for note in raised.value.__notes__)
        assert any(str(cleanup_targets[0]) in note
                   for note in raised.value.__notes__)
        assert cleanup_targets[0].parent == tmp_path
    finally:
        for path in cleanup_targets:
            real_unlink(path, missing_ok=True)


def test_audio_snapshot_reports_normal_exit_cleanup_failure(tmp_path, monkeypatch):
    from manialab.audio import _audio_snapshot

    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic source")
    real_unlink = Path.unlink
    cleanup_targets = []

    def fail_snapshot_cleanup(path, *args, **kwargs):
        if path.name.startswith(".analysis-input-"):
            cleanup_targets.append(path)
            raise OSError("synthetic cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_snapshot_cleanup)
    try:
        with pytest.raises(OSError, match="synthetic cleanup failure") as raised:
            with _audio_snapshot(source, tmp_path, lambda: False) as (snapshot, _):
                assert snapshot.exists()

        assert len(cleanup_targets) == 1
        assert raised.value.__notes__ == [
            f"分析音频快照无法删除（路径：{cleanup_targets[0]}）"]
        assert cleanup_targets[0].exists()
    finally:
        for path in cleanup_targets:
            real_unlink(path, missing_ok=True)


def test_deep_analysis_decodes_and_separates_the_same_audio_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "mix.wav"
    source_a, source_b = b"synthetic mix A", b"synthetic mix B"
    source.write_bytes(source_a)
    expected_hash = hashlib.sha256(source_a).hexdigest()
    calls = []
    samples = np.full(SR, 0.2, dtype=np.float32)
    separation = dict(source='provided', voices=['drums'], audio_sha256=expected_hash,
        set_sha256='a' * 64, model_sha256='a' * 64, cached=True)
    np.save(tmp_path / 'drums.npy', samples, allow_pickle=False)

    def decode_snapshot(path, work, cancelled=None):
        path = Path(path)
        calls.append(('decode', path, path.read_bytes()))
        assert path != source
        source.write_bytes(source_b)
        return samples

    class Separator:
        def separate(self, path, work, progress, cancelled):
            path = Path(path)
            calls.append(('separate', path, path.read_bytes()))
            return tmp_path, separation

    monkeypatch.setattr(audio, 'decode', decode_snapshot)
    monkeypatch.setattr(audio, 'rhythmic_evidence', lambda *args, **kwargs: ([], {'bpm': 120}))
    monkeypatch.setattr(audio, 'source_evidence', lambda notes, *args: (notes, 0))
    monkeypatch.setattr(audio, 'reject_source_bleed', lambda notes, *args: (notes, []))
    monkeypatch.setattr(audio, 'timing_evidence_quality', lambda tempo: {'coverage': 0, 'score': 0})
    monkeypatch.setattr(
        'manialab.structure.detect_structure',
        lambda analysis, samples, sample_rate, **kwargs: {'segments': []},
    )

    result = audio.analyze_deep(source, tmp_path, tmp_path / 'missing-model.onnx',
        parameters(), Separator())

    assert calls[0][0] == 'decode' and calls[1][0] == 'separate'
    assert calls[0][1] == calls[1][1]
    assert calls[0][2] == calls[1][2] == source_a
    assert source.read_bytes() == source_b
    assert result['audio_sha256'] == result['separation']['audio_sha256'] == expected_hash
    assert not calls[0][1].exists()
    assert not list(tmp_path.glob('.analysis-input-*'))


def test_deep_analysis_rejects_separation_for_a_different_audio_identity(tmp_path, monkeypatch):
    source = tmp_path / 'mix.wav'
    source.write_bytes(b'synthetic mix')
    monkeypatch.setattr(audio, 'decode', lambda *args, **kwargs: np.zeros(SR, dtype=np.float32))

    class MismatchedSeparator:
        def separate(self, path, work, progress, cancelled):
            return tmp_path, dict(source='provided', voices=['drums'], audio_sha256='0' * 64,
                                  set_sha256='a' * 64, model_sha256='a' * 64, cached=True)

    with pytest.raises(ValueError, match='声部来源摘要与音频快照不匹配'):
        audio.analyze_deep(source, tmp_path, tmp_path / 'missing-model.onnx',
                           parameters(), MismatchedSeparator())

    assert not list(tmp_path.glob('.analysis-input-*'))


def test_silence():
    notes,_=rhythmic_evidence(np.zeros(SR*3))
    assert notes==[]


def test_residual_stem_cannot_be_amplified_into_musical_evidence():
    from manialab.audio import source_evidence
    from manialab.domain import Note
    mix=np.full(SR*3,.2,dtype=np.float32);mix[SR*2:]=0
    residual=np.full_like(mix,.00006)
    notes=[Note(500,500,source='transient'),Note(2500,2500,source='transient')]
    kept,rejected=source_evidence(notes,residual,mix)
    assert kept==[] and rejected==2
    # A quiet but real source remains eligible; this is a conservative noise
    # floor, not a rule that only the dominant instrument may be mapped.
    quiet=residual.copy();quiet[:SR]=.008
    kept,rejected=source_evidence(notes,quiet,mix)
    assert kept==[notes[0]] and rejected==1


def test_release_refinement_recovers_silence_without_extending_notes():
    from manialab.audio import refine_releases
    from manialab.domain import Note
    wave=np.zeros(SR*4,dtype=np.float32);wave[SR:SR*3]=.1
    notes=[Note(1000,3050,source='pitch'),Note(1000,2500,source='pitch')]
    output,report=refine_releases(notes,wave)
    assert abs(output[0].end-3000)<=5 and output[1].end==2500
    assert report['shortened']==1
    assert all(new.start==old.start and new.end<=old.end for new,old in zip(output,notes))


def test_real_model_timing_and_sustains(tmp_path):
    y=np.zeros(SR*30,dtype=np.float32)
    expected=[(1,2,60),(5,4,64),(12,1,67),(20,3,60),(26,2,72)]
    for start,duration,midi in expected:
        t=np.arange(duration*SR)/SR
        freq=440*2**((midi-69)/12)
        tone=.3*(np.sin(2*np.pi*freq*t)+.3*np.sin(4*np.pi*freq*t)+.1*np.sin(6*np.pi*freq*t))
        tone*=np.minimum(1,t/.015)*np.minimum(1,(duration-t)/.04)
        y[start*SR:(start+duration)*SR]+=tone.astype(np.float32)
    wavfile.write(tmp_path/'synth.wav',SR,y)
    result=analyze(tmp_path/'synth.wav',tmp_path,Path(__file__).parents[1]/'assets/basic-pitch.onnx',parameters())
    pitched=[n for n in result['candidates'] if n['source']=='pitch']
    assert len(pitched) == len(expected), pitched
    assert all(n['release_reason'] == 'frame_decay' for n in pitched)
    for start,duration,midi in expected:
        matches=[n for n in pitched if n['pitch']==midi and abs(n['start']-start*1000)<120]
        assert matches, (start,midi,pitched)
        assert abs(matches[0]['end']-(start+duration)*1000)<180
    assert result['duration']==30
    attacks=[n['start'] for n in result['candidates'] if n['source']=='transient']
    assert not any(abs(attack-(start+length)*1000)<100 for attack in attacks for start,length,_ in expected)
def test_short_cross_source_leakage_is_not_an_extra_instrument():
    import numpy as np
    from manialab.audio import reject_source_bleed
    from manialab.domain import Note
    rate=22050;t=np.arange(rate*4)/rate
    other=np.sin(2*np.pi*261.63*t).astype(np.float32)
    stems={'other':other,'bass':other*.05,'vocals':other*.5}
    notes=[Note(1000,3000,pitch=60,strength=.8,source='pitch',voice='other',id='real'),
           Note(1020,1120,pitch=61,strength=.4,source='pitch',voice='bass',id='duplicate'),
           Note(2900,3000,pitch=60,strength=.4,source='pitch',voice='bass',id='tail'),
           Note(1500,1600,pitch=67,strength=.4,source='pitch',voice='bass',id='other-pitch'),
           Note(1020,1120,pitch=60,strength=.4,source='pitch',voice='vocals',id='unison')]
    result,rejected=reject_source_bleed(notes,stems,[])
    assert {n.id for n in result}=={'real','other-pitch','unison'}
    assert {r['reason'] for r in rejected}=={'weak_duplicate_attack','release_edge_bleed'}
    result,_=reject_source_bleed(notes,stems,[Note(2900,2900)])
    assert 'tail' in {n.id for n in result}


def test_steady_bass_retrigger_can_be_removed_without_smoothing_a_real_rest():
    from manialab.articulation import steady_bass_onsets
    from manialab.audio import pitched_notes
    times=np.arange(300)*256/SR
    frames=np.zeros((300,88));onsets=np.zeros_like(frames)
    frames[40:180,12]=.7;onsets[40,12]=.8;onsets[100,12]=.5
    t=np.arange(SR*4)/SR;frequency=440*2**((33-69)/12)
    wave=.2*np.sin(2*np.pi*frequency*t)
    original=onsets.copy()
    suppressed=steady_bass_onsets(frames,onsets,times,wave)
    assert len(suppressed)==1 and suppressed[0]['time_ms']==round(times[100]*1000)
    notes=pitched_notes(frames,onsets,times,4,suppressed=suppressed)
    assert len(notes)==1 and notes[0].end>2000
    assert np.array_equal(onsets,original)
    # Even a short true rest around the same suspected reattack prevents
    # suppression; the test leaves model activation constant intentionally.
    at=times[100];wave[(t>=at-.015)&(t<at)]=0
    assert steady_bass_onsets(frames,onsets,times,wave)==[]
