from scripts.research_adt_str import token_audit


def test_complete_drum_tokens_and_silence():
    assert token_audit([2,4,335,500,259,344,490,3,3])==[]
    assert token_audit([2,0,3])==[]


def test_incomplete_events_are_not_silently_accepted_by_zip_decoder():
    assert 'incomplete_triple' in token_audit([2,5,342,490,10,3])
    assert 'time_outside_chunk' in token_audit([2,299,342,490,3])
    assert 'invalid_custom_percussion' in token_audit([2,5,381,490,3])
    assert 'missing_eos' in token_audit([2,5,342,490])


def test_old_success_flag_cannot_hide_invalid_saved_tokens(tmp_path):
    import json
    import numpy as np
    from scripts.audit_adt_tokens import audit_run
    from scripts.probe_mapper_adapter import sha
    np.savez_compressed(tmp_path/'tokens.npz', **{'0':np.array([2,4,336,500,4,336,510,3])})
    (tmp_path/'events.json').write_text('[]')
    report=dict(identity=dict(chunk_samples=61440,sample_rate=24000),duration_ms=2560,
        tokens_sha256=sha(tmp_path/'tokens.npz'),events_sha256=sha(tmp_path/'events.json'),token_issues={})
    (tmp_path/'report.json').write_text(json.dumps(report))
    audited=audit_run(tmp_path)
    assert not audited['suitable_for_event_scoring']
    assert audited['issues']=={'duplicate_percussion_event':1}


def test_independent_decode_rejects_saved_event_reassociation(tmp_path):
    import json
    import numpy as np
    from scripts.audit_adt_tokens import audit_run
    from scripts.probe_mapper_adapter import sha
    np.savez_compressed(tmp_path/'tokens.npz', **{'0':np.array([2,10,336,500,3])})
    (tmp_path/'events.json').write_text(json.dumps([dict(onset=.06,custom_pitch=42,velocity=100)]))
    report=dict(identity=dict(chunk_samples=61440,sample_rate=24000),duration_ms=2560,
        tokens_sha256=sha(tmp_path/'tokens.npz'),events_sha256=sha(tmp_path/'events.json'),token_issues={})
    (tmp_path/'report.json').write_text(json.dumps(report))
    assert audit_run(tmp_path)['issues']=={'exact_decoder_disagreement':1}
