"""Pure bridge/provenance controls: no Torch, weights, inference or labels."""
import copy

import pytest

from manialab.arrange import parameters
from manialab.domain import UNOBSERVED_RELEASES, note_from
from manialab.expression import expression_evidence, propose_expression
from scripts import tsumugi_candidate_adapter as adapter
from scripts.audit_tsumugi_releases import classify_endpoints, loss_diagnostics, replay_lineage
from scripts.recover_tsumugi_attacks import data_sha, recover_attacks
from scripts.research_tsumugi_transcription import convert_notes


CLASSES = ['guitar', 'drums', 'melody', 'wind_chimes']


def note(start=100, end=1000, *, pitch=60, slot=0, instrument=0, onset=True, offset=True):
    return dict(instrument_id=instrument, pitch=pitch, start_sample=start, end_sample=end,
        velocity=100, slot_index=slot, has_onset=onset, has_offset=offset, instrument_candidates=[instrument])


def construction(value, *, total=5000, boundary=True):
    return dict(arguments=dict(pair_id=value['pitch']-21, begin_frame=value['start_sample']//10,
        end_frame=value['end_sample']//10, boundary_flag=[value['has_onset'], value['has_offset'],
            (value['start_sample'] % 10)/10, (value['end_sample'] % 10)/10] if boundary else None,
        window_start_frame=0, valid_audio_frames=total, valid_model_frames=(total+9)//10),
        constructed_note=copy.deepcopy(value), v1_slot_index=value['slot_index'],
        hop_length=10, total_audio_samples=total)


def fixture(values=None, *, constructions=None, model='guitar_v1_5', total=5000):
    values = [note()] if values is None else values
    rows = constructions if constructions is not None else [construction(n, total=total) for n in values]
    identity = dict(source_commit=adapter.SOURCE_COMMIT, model_revision=adapter.MODEL_REVISION,
        model=model, model_sha256=adapter.MODELS[model], evidence_revision=2,
        original_sample_rate=1000, original_samples=total, resampled_sample_rate=1000, resampled_samples=total,
        audio_sha256='1'*64, model_config=dict(semi_crf_version='v1', sample_rate=1000,
            num_pitch_slots=3, num_instrument_classes=len(CLASSES)))
    tracks = {}
    for i, value in enumerate(values):
        n = copy.deepcopy(value)
        n['instrument_id'] = 0
        tracks.setdefault((n['pitch'], n['slot_index']), []).append(dict(source=i, note=n))
    kept, dropped, forced = replay_lineage(tracks, merge_gap_samples=10, merge_onset_samples=50)
    classify_endpoints(kept, rows, identity, forced)
    audited = dict(identity=identity, notes=kept, filtered_groups=dropped,
        official_finalize_equal=True, saved_raw_notes_equal=True, window_snapshots_equal=True,
        **loss_diagnostics(kept, dropped, rows))
    baseline = convert_notes([r['note'] for r in kept], sample_rate=1000, total_samples=total, classes=CLASSES)
    return audited, baseline, CLASSES, rows


def bridge(args, **kwargs):
    return adapter.tsumugi_candidates(*args[:3], **kwargs)


def recovered(args):
    audited, baseline, classes, rows = args
    return dict(identity=copy.deepcopy(audited['identity']), verified_replay_sha256=data_sha(audited),
        protocol_sha256=adapter.PROTOCOL_SHA256, **recover_attacks(audited, rows, baseline, classes))


def express(notes):
    analysis = dict(duration=5, candidates=[n.data() for n in notes],
        structure=dict(segments=[dict(start_ms=0, end_ms=5000)]))
    settings = parameters({'ln_ratio': 1.})
    plan = propose_expression(analysis, settings)
    for section in plan['sections']:
        for voice in section['voices'].values():
            voice['mode'] = 'sustain'
    return expression_evidence(analysis, plan, settings)


def test_default_raw_end_survives_but_actual_expression_becomes_tap():
    args = fixture()
    before = copy.deepcopy(args)
    notes, trace = bridge(args)
    assert args == before
    assert notes[0].start == 100 and notes[0].end == 1000
    assert notes[0].release_reason == 'unresolved_or_postprocessed'
    assert notes[0].release_reason in UNOBSERVED_RELEASES
    assert trace[0]['raw_event'] == args[1][0]
    assert trace[0]['evidence'] == args[0]['notes'][0]
    assert trace[0]['weight_semantics'] == 'unit_weight_not_confidence'
    assert not trace[0]['authored_ln_verified']
    assert not trace[0]['model_predicted_release_hypothesis']
    actual, _, decisions = express(notes)
    assert actual[0].start == actual[0].end == 100
    assert decisions[notes[0].id]['reason'] == 'unobserved_release'
    assert notes[0].end == 1000


def test_explicit_predicted_release_is_a_duration_hypothesis_not_ln_truth():
    notes, trace = bridge(fixture(), allow_predicted_releases=True)
    assert notes[0].release_reason == 'model_predicted_offset'
    assert trace[0]['model_predicted_release_hypothesis']
    assert not trace[0]['authored_ln_verified']
    assert trace[0]['final_endpoint_model_supported']
    assert express(notes)[0][0].end == 1000
    assert note_from(notes[0].data()) == notes[0]


@pytest.mark.parametrize('kind', ['forced', 'mixed', 'extended_unknown', 'fallback', 'clipped', 'window', 'input'])
def test_finalizer_flags_never_unlock_unsupported_endpoints(kind):
    values = [note()]
    if kind == 'forced':
        values = [note(offset=False)]
    elif kind == 'mixed':
        values = [note(), note(120, 1000, offset=False), note(2000, 2500)]
    elif kind == 'extended_unknown':
        values = [note(), note(120, 1200, offset=False), note(2000, 2500)]
    elif kind == 'input':
        values = [note(end=5000)]
    rows = [construction(n) for n in values]
    if kind == 'fallback':
        rows[0]['arguments']['boundary_flag'] = None
    elif kind == 'clipped':
        rows[0]['arguments']['end_frame'] = 110
    elif kind == 'window':
        rows[0]['arguments']['valid_audio_frames'] = 1000
    args = fixture(values, constructions=rows)
    assert args[0]['notes'][0]['note']['has_offset'] is True
    notes, trace = bridge(args, allow_predicted_releases=True)
    assert notes[0].release_reason == adapter.UNKNOWN_RELEASE
    assert not trace[0]['final_endpoint_model_supported']
    assert not trace[0]['model_predicted_release_hypothesis']
    assert express(notes)[0][0].end == notes[0].start


def test_extended_endpoint_is_allowed_only_from_its_supported_final_source():
    args = fixture([note(offset=False), note(990, 1200, onset=False), note(2000, 2500)])
    item = args[0]['notes'][0]
    assert item['classification'] == 'merge_extended_predicted_offset'
    assert item['endpoint_sources'] == [1] and not item['unmodified_boundary_head_endpoint']
    notes, trace = bridge(args, allow_predicted_releases=True)
    assert notes[0].end == 1200 and notes[0].release_reason == adapter.PREDICTED_RELEASE
    assert trace[0]['evidence']['merge_extended_endpoint']


def test_forced_shorter_merged_source_cannot_invalidate_separate_supported_final_end():
    args = fixture([note(), note(120, 500, offset=False)])
    item = args[0]['notes'][0]
    assert item['forced_offset_sources'] == [1] and item['endpoint_sources'] == [0]
    assert not item['unmodified_boundary_head_endpoint']
    notes, trace = bridge(args, allow_predicted_releases=True)
    assert notes[0].end == 1000 and notes[0].release_reason == adapter.PREDICTED_RELEASE
    assert trace[0]['evidence']['forced_offset_sources'] == [1]


def test_all_equal_endpoint_sources_must_be_supported():
    args = fixture([note(), note(120, 1000), note(2000, 2500)])
    assert args[0]['notes'][0]['endpoint_sources'] == [0, 1]
    assert bridge(args, allow_predicted_releases=True)[0][0].release_reason == adapter.PREDICTED_RELEASE


def test_all_wrong_non_drum_classes_remain_and_drums_keep_rejection_trace():
    args = fixture([note(pitch=60, instrument=0), note(pitch=61, instrument=1),
                    note(pitch=62, instrument=2), note(pitch=63, instrument=3)])
    before = copy.deepcopy(args)
    notes, trace = bridge(args, voice='guitar', allow_predicted_releases=True)
    assert len(notes) == 3 and len(trace) == 4 and args == before
    assert [row['instrument_class'] for row in trace] == CLASSES
    assert [row['decision'] for row in trace] == ['candidate', 'incompatible_drum_prediction', 'candidate', 'candidate']
    assert {n.voice for n in notes} == {'guitar'}
    assert not trace[1]['model_predicted_release_hypothesis']
    assert trace[1]['release_reason'] == adapter.UNKNOWN_RELEASE
    assert len(args[1]) == 4


def test_slot_is_trace_not_physical_instrument_and_ids_remain_unique():
    notes, trace = bridge(fixture([note(slot=0), note(slot=1)]))
    assert len(notes) == 2 and len({n.id for n in notes}) == 2
    assert notes[0].instrument_id == notes[1].instrument_id
    assert [row['slot_index'] for row in trace] == [0, 1]
    assert 'slot' not in notes[0].instrument_id
    assert notes[0].instrument_id.startswith('tsumugi:guitar_v1_5:')
    assert notes[0].instrument_id.endswith(':class:0:guitar')
    assert note_from(notes[0].data()).instrument_id == notes[0].instrument_id
    other = bridge(fixture(model='bass_v2'))[0][0]
    assert other.instrument_id != notes[0].instrument_id


@pytest.mark.parametrize('field', ['official_finalize_equal', 'saved_raw_notes_equal', 'window_snapshots_equal'])
def test_unverified_replay_is_rejected(field):
    args = fixture()
    args[0][field] = False
    with pytest.raises(ValueError, match='verified'):
        bridge(args)


@pytest.mark.parametrize('field,value', [('model', 'unknown'), ('model_revision', 'wrong'),
    ('model_sha256', 'wrong'), ('source_commit', 'wrong'), ('evidence_revision', 1),
    ('original_samples', 4998), ('resampled_sample_rate', 2000), ('resampled_samples', True)])
def test_identity_or_clock_mismatch_rejected(field, value):
    args = fixture()
    args[0]['identity'][field] = value
    with pytest.raises(ValueError):
        bridge(args)


@pytest.mark.parametrize('changed', ['baseline_offset', 'baseline_drum', 'baseline_class', 'class_order', 'slot', 'flag', 'index', 'boolean_index'])
def test_baseline_and_raw_correspondence_must_be_exact(changed):
    args = fixture()
    if changed == 'baseline_offset': args[1][0]['offset'] += .1
    elif changed == 'baseline_drum': args[1][0]['is_drum'] = True
    elif changed == 'baseline_class': args[1][0]['instrument_class'] = 'melody'
    elif changed == 'class_order': args = (args[0], args[1], list(reversed(args[2])), args[3])
    elif changed == 'slot': args[0]['notes'][0]['note']['slot_index'] = 5
    elif changed == 'flag': args[0]['notes'][0]['note']['has_offset'] = 1
    elif changed == 'index': args[0]['notes'][0]['note_index'] = 1
    elif changed == 'boolean_index': args[0]['notes'][0]['note_index'] = False
    with pytest.raises(ValueError):
        bridge(args)


@pytest.mark.parametrize('changed', ['missing_source', 'added_source', 'different_end', 'false_support', 'reason', 'summary', 'boundary'])
def test_endpoint_evidence_cannot_be_replaced_by_convenient_summary(changed):
    args = fixture()
    item = args[0]['notes'][0]
    value = item['endpoint_evidence']['0']
    if changed == 'missing_source': item['endpoint_evidence'] = {}
    elif changed == 'added_source': item['endpoint_sources'].append(1)
    elif changed == 'different_end': value['constructed_end_sample'] = 900
    elif changed == 'false_support': value['model_predicted_offset_support'] = False
    elif changed == 'reason': value['unsupported_reasons'] = ['at_window_observation_boundary']
    elif changed == 'summary': item['model_supported_endpoint_sources'] = []
    elif changed == 'boundary': value['at_original_input_boundary'] = True
    with pytest.raises(ValueError):
        bridge(args, allow_predicted_releases=True)


@pytest.mark.parametrize('flag', [1, 0, 'true', None])
def test_release_optin_cannot_be_a_truthy_nonboolean(flag):
    with pytest.raises(ValueError, match='boolean'):
        bridge(fixture(), allow_predicted_releases=flag)


def test_recovered_attack_stays_zero_with_full_raw_end_and_identity_trace():
    args = fixture([note(100, 500, offset=False), note(1000, 1500)])
    recovery = recovered(args)
    before = copy.deepcopy((args, recovery))
    notes, trace = bridge(args, allow_predicted_releases=True, recovery=recovery)
    assert len(notes) == 2 and (args, recovery) == before
    n, row = notes[1], trace[1]
    assert n.start == n.end == 100 and n.release_reason == adapter.UNKNOWN_RELEASE
    assert row['raw_end_ms'] == 500 and row['raw_note']['end_sample'] == 500
    assert row['raw_event']['offset'] == row['raw_event']['onset'] == .1
    assert row['evidence']['original_model_interval']['end_sample'] == 500
    assert row['origin'] == 'recovered' and row['recovery_source_index'] == 0
    assert row['verified_replay_sha256'] == data_sha(args[0])
    assert not row['final_endpoint_model_supported'] and not row['model_predicted_release_hypothesis']
    assert express(notes)[0][1].end == 100


def test_recovered_wrong_class_survives_and_recovered_drum_is_audited_rejection():
    args = fixture([note(100, 500, offset=False, instrument=1), note(1000, 1500),
                    note(100, 500, offset=False, pitch=61, instrument=3), note(1000, 1500, pitch=61)])
    notes, trace = bridge(args, recovery=recovered(args), voice='vocals')
    assert len(notes) == 3 and len(trace) == 4
    assert trace[2]['decision'] == 'incompatible_drum_prediction'
    assert trace[3]['decision'] == 'candidate' and trace[3]['instrument_class'] == 'wind_chimes'
    assert all(n.voice == 'vocals' for n in notes)


@pytest.mark.parametrize('changed', ['identity', 'replay', 'protocol', 'baseline', 'modified', 'source', 'duration', 'class', 'decision', 'slot', 'clock', 'release_claim'])
def test_recovery_must_match_replay_and_original_onset_only_evidence(changed):
    args = fixture([note(100, 500, offset=False), note(1000, 1500)])
    result = recovered(args)
    if changed == 'identity': result['identity']['audio_sha256'] = '2'*64
    elif changed == 'replay': result['verified_replay_sha256'] = 'wrong'
    elif changed == 'protocol': result['protocol_sha256'] = 'wrong'
    elif changed == 'baseline': result['baseline_notes'][0]['pitch'] += 1
    elif changed == 'modified': result['baseline_modified'] = True
    elif changed == 'source': result['recovered_attacks'][0]['recovery_source_index'] = 1
    elif changed == 'duration': result['recovered_attacks'][0]['offset'] = .5
    elif changed == 'class': result['recovered_attacks'][0]['instrument_class'] = 'drums'
    elif changed == 'decision': result['decisions'][0]['source_index'] = 1
    elif changed == 'slot': result['decisions'][0]['v1_slot_index'] = 5
    elif changed == 'clock': result['decisions'][0]['sample_rate'] = 22050
    elif changed == 'release_claim': result['decisions'][0]['original_interval_is_not_confirmed_release'] = False
    with pytest.raises(ValueError):
        bridge(args, recovery=result, allow_predicted_releases=True)


def test_submillisecond_interval_rounding_cannot_invent_a_hold():
    args = fixture([note(100, 101)])
    identity = args[0]['identity']
    identity.update(resampled_sample_rate=10000, original_sample_rate=10000)
    identity['model_config']['sample_rate'] = 10000
    baseline = convert_notes([args[0]['notes'][0]['note']], sample_rate=10000, total_samples=5000, classes=CLASSES)
    notes, trace = bridge((args[0], baseline, CLASSES, args[3]), allow_predicted_releases=True)
    assert notes[0].start == notes[0].end == 10
    assert notes[0].release_reason == adapter.UNKNOWN_RELEASE
    assert not trace[0]['model_predicted_release_hypothesis']


def test_integer_sample_rounding_and_original_input_boundary_are_preserved():
    args = fixture([note(1, 5)], total=5)
    identity = args[0]['identity']
    identity.update(original_sample_rate=2, original_samples=3, resampled_sample_rate=3)
    identity['model_config']['sample_rate'] = 3
    classify_endpoints(args[0]['notes'], args[3], identity, [])
    baseline = convert_notes([args[0]['notes'][0]['note']], sample_rate=3, total_samples=5, classes=CLASSES)
    notes, trace = bridge((args[0], baseline, CLASSES, args[3]), allow_predicted_releases=True)
    assert (notes[0].start, notes[0].end) == (333, 1667)
    assert notes[0].release_reason == adapter.UNKNOWN_RELEASE
    assert trace[0]['raw_note']['end_sample'] == 5


def test_adapt_case_calls_verified_audit_and_preserves_baseline_artifact(tmp_path, monkeypatch):
    import json
    args = fixture()
    audited, baseline, classes, _ = args
    path = tmp_path/'notes.json'
    path.write_text(json.dumps(baseline), encoding='utf-8')
    audited.update(evidence_sha256={'notes.json': adapter.sha(path)}, report_sha256='report')
    calls = []
    def audit(folder, source):
        calls.append((folder, source))
        return copy.deepcopy(audited)
    monkeypatch.setattr(adapter.release_audit, 'audit', audit)
    monkeypatch.setattr(adapter, 'load_classes', lambda source, identity: classes)
    result = adapter.adapt_case(tmp_path, tmp_path/'upstream', voice='guitar')
    assert calls == [(tmp_path, tmp_path/'upstream')]
    assert result['baseline_notes'] == baseline and json.loads(path.read_text()) == baseline
    assert not result['enabled_in_application'] and not result['scoring'] and not result['neural_inference']
    assert result['counts'] == dict(baseline=1, recovered=0, candidates=1, rejected_drums=0, predicted_release_hypotheses=0)


def test_cli_overwrite_guard_precedes_reading_or_auditing(tmp_path, monkeypatch):
    path = tmp_path/'prior.json'
    path.write_text('original', encoding='utf-8')
    monkeypatch.setattr(adapter.sys, 'argv', ['adapter', '--input', str(tmp_path/'absent'),
        '--voice', 'other', '--output', str(path)])
    monkeypatch.setattr(adapter, 'adapt_case', lambda *a, **k: pytest.fail('read attempted before overwrite guard'))
    with pytest.raises(ValueError, match='Preserve'):
        adapter.main()
    assert path.read_text() == 'original'
