"""Research-only bridge from verified Tsumugi replay to ManiaLab candidates.

No model loading, inference, scoring or application defaults. Raw predictions and
all non-drum classes survive. Predicted releases require an explicit opt-in and
source-level support at the final endpoint; they are not authored LN truth.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import Note, note_from
from scripts import audit_tsumugi_releases as release_audit
from scripts.recover_tsumugi_attacks import (
    DEFAULT_SOURCE, PROTOCOL_SHA256, data_sha, load_classes, recover_attacks, write_new,
)
from scripts.research_tsumugi_transcription import (
    MODELS, MODEL_REVISION, SOURCE_COMMIT, convert_notes,
)

require, read_json, sha = release_audit.require, release_audit.read_json, release_audit.sha
UNKNOWN_RELEASE = 'unresolved_or_postprocessed'
PREDICTED_RELEASE = 'model_predicted_offset'
VOICES = ('mix', 'piano', 'bass', 'guitar', 'other', 'vocals')


def _identity(audited, classes):
    require(all(audited.get(k) is True for k in
        ('official_finalize_equal', 'saved_raw_notes_equal', 'window_snapshots_equal')),
        'Require verified official/raw/window replay')
    identity = audited['identity']
    config = identity['model_config']
    require(identity.get('evidence_revision') == 2 and config.get('semi_crf_version') == 'v1',
            'Require traced V1 evidence revision 2')
    require(identity.get('source_commit') == SOURCE_COMMIT and identity.get('model_revision') == MODEL_REVISION
            and identity.get('model') in MODELS and identity.get('model_sha256') == MODELS[identity['model']],
            'Model/source identity differs from reviewed Tsumugi revision')
    clock = [identity.get(k) for k in
             ('resampled_sample_rate', 'resampled_samples', 'original_sample_rate', 'original_samples')]
    require(all(type(value) is int and value > 0 for value in clock), 'Invalid recorded sample clock')
    sr, total, original_rate, original_total = clock
    require(type(config.get('sample_rate')) is int and config['sample_rate'] == sr
            and total == (original_total * sr + original_rate - 1) // original_rate,
            'Original/model sample clocks differ')
    require(type(config.get('num_pitch_slots')) is int and config['num_pitch_slots'] > 0, 'Invalid slot configuration')
    require(isinstance(classes, (list, tuple)) and len(classes) == config.get('num_instrument_classes')
            and all(isinstance(c, str) and c and not any(ord(ch) < 32 for ch in c) for c in classes)
            and len(set(classes)) == len(classes), 'Invalid instrument class mapping')
    scope = {key: identity[key] for key in ('source_commit', 'model_revision', 'model', 'model_sha256')}
    return identity, sr, total, f"tsumugi:{identity['model']}:{data_sha(scope)[:16]}"


def _raw_note(note, identity):
    require(type(note.get('slot_index')) is int
            and 0 <= note['slot_index'] < identity['model_config']['num_pitch_slots'], 'Invalid raw slot index')
    require(all(type(note.get(key)) is bool for key in ('has_onset', 'has_offset')), 'Invalid raw boundary flags')
    candidates = note.get('instrument_candidates')
    require(isinstance(candidates, list) and all(type(c) is int
            and 0 <= c < identity['model_config']['num_instrument_classes'] for c in candidates),
            'Invalid raw instrument candidates')


def _endpoint_support(item, identity):
    """Check every final-end source; a merged/finalized has_offset is insufficient."""
    sources, endpoints = item['sources'], item['endpoint_sources']
    require(isinstance(sources, list) and sources and all(type(i) is int and i >= 0 for i in sources)
            and len(sources) == len(set(sources)), 'Invalid lineage source indices')
    require(isinstance(endpoints, list) and endpoints and len(endpoints) == len(set(endpoints))
            and all(type(i) is int and i in sources for i in endpoints), 'Invalid endpoint source indices')
    evidence = item['endpoint_evidence']
    require(set(evidence) == {str(i) for i in endpoints}, 'Endpoint audit must cover every final source exactly once')
    supported = []
    for source in endpoints:
        value = evidence[str(source)]
        raw_end, end, window_end = [value.get(key) for key in
            ('raw_predicted_end_sample', 'constructed_end_sample', 'window_valid_end_sample')]
        require(all(type(v) is int and v >= 0 for v in (raw_end, end, window_end))
                and end == item['note']['end_sample'] and window_end <= identity['resampled_samples'],
                'Endpoint coordinates differ from audited final note')
        flag = value.get('boundary_head_offset')
        require(flag is None or type(flag) is bool, 'Invalid endpoint boundary-head flag')
        at_input = end * identity['original_sample_rate'] >= identity['original_samples'] * identity['resampled_sample_rate']
        reasons = []
        if flag is None:
            reasons.append('no_boundary_head_fallback')
        elif not flag:
            reasons.append('boundary_head_has_no_offset')
        if raw_end != end:
            reasons.append('constructor_adjusted_endpoint')
        if end >= window_end:
            reasons.append('at_window_observation_boundary')
        if at_input:
            reasons.append('at_original_input_boundary')
        require(value.get('endpoint_adjusted') is (raw_end != end)
                and value.get('at_original_input_boundary') is at_input
                and value.get('unsupported_reasons') == reasons
                and value.get('model_predicted_offset_support') is (not reasons),
                'Endpoint support is inconsistent with source evidence')
        if not reasons:
            supported.append(source)
    require(item.get('model_supported_endpoint_sources') == supported, 'Endpoint support summary differs')
    return len(supported) == len(endpoints)


def _recovered_events(recovery, audited, baseline, classes):
    """Validate the fixed recovery result; do not implement a second deduplicator."""
    if recovery is None:
        return []
    identity = audited['identity']
    require(recovery.get('identity') == identity
            and recovery.get('verified_replay_sha256') == data_sha(audited)
            and recovery.get('protocol_sha256') == PROTOCOL_SHA256, 'Recovery identity/replay/protocol differs')
    require(recovery.get('baseline_modified') is False and recovery.get('baseline_notes') == baseline,
            'Recovery changed the baseline')
    events = recovery.get('recovered_attacks')
    require(isinstance(events, list) and recovery.get('notes') == baseline + events, 'Recovery event list differs')
    decisions = recovery['decisions']
    decision_sources = [d['source_index'] for d in decisions]
    expected_sources = audited['filtered_early_unclosed_onset_source_indices']
    require(len(decision_sources) == len(set(decision_sources)) and set(decision_sources) == set(expected_sources),
            'Recovery decisions omit or add audited onset sources')
    by_source = {d['source_index']: d for d in decisions if d['decision'] == 'recovered_onset_only'}
    ids = [event.get('recovery_source_index') for event in events]
    require(all(type(i) is int for i in ids) and len(ids) == len(set(ids)) and set(ids) == set(by_source),
            'Recovery attacks differ from accepted source decisions')
    result = []
    for event in events:
        decision = by_source[event['recovery_source_index']]
        original = decision['original_model_interval']
        note = copy.deepcopy(original)
        note['slot_index'] = decision['v1_slot_index']
        _raw_note(note, identity)
        require(note['has_onset'] is True and note['has_offset'] is False
                and decision.get('sample_rate') == identity['resampled_sample_rate']
                and decision.get('original_interval_is_not_confirmed_release') is True,
                'Recovery must remain an unclosed onset on the recorded clock')
        expected = convert_notes([note], sample_rate=identity['resampled_sample_rate'],
                                 total_samples=identity['resampled_samples'], classes=classes)[0]
        expected.update(offset=expected['onset'], release_reason=UNKNOWN_RELEASE, event_kind='onset_only',
            application_ln_eligible=False, recovery_source_index=event['recovery_source_index'],
            model_has_offset_after_stitching=False)
        require(event == expected, 'Recovered event differs from original onset-only evidence')
        result.append((event, note, decision))
    return result


def tsumugi_candidates(audited, baseline, classes, *, voice='other', allow_predicted_releases=False, recovery=None):
    """Return (Notes, trace) from trusted replay, preserving the input predictions.

    Obtain ``audited`` with audit_tsumugi_releases.audit(). The CLI does this
    itself. ``voice`` is a known input-stem role, never inferred from a benchmark
    label or the model class. A recovery, when supplied, is the verified result
    of recover_case(). All non-drum classes remain, including wrong classes.
    """
    require(type(allow_predicted_releases) is bool, 'Release opt-in must be an explicit boolean')
    require(voice in VOICES, 'Invalid known source voice')
    identity, sr, total, model_scope = _identity(audited, classes)
    rows = audited['notes']
    require(all(type(row.get('note_index')) is int for row in rows)
            and [row['note_index'] for row in rows] == list(range(len(rows))),
            'Audit must cover every baseline event exactly once in order')
    raw = [row['note'] for row in rows]
    for note in raw:
        _raw_note(note, identity)
    require(baseline == convert_notes(raw, sample_rate=sr, total_samples=total, classes=classes),
            'Baseline events differ from audited raw conversion')
    recovered = _recovered_events(recovery, audited, baseline, classes)
    verified = data_sha(audited)
    namespace = data_sha(identity)[:20]
    notes, trace = [], []

    def append(event, raw_note, origin, index, evidence, supported):
        start = round(Fraction(raw_note['start_sample'] * 1000, sr))
        raw_end = round(Fraction(raw_note['end_sample'] * 1000, sr))
        end = start if origin == 'recovered' else raw_end
        predicted = allow_predicted_releases and supported and end > start and origin == 'baseline' and not event['is_drum']
        reason = PREDICTED_RELEASE if predicted else UNKNOWN_RELEASE
        class_id = event['instrument_class_id']
        instrument = f"{model_scope}:class:{class_id}:{event['instrument_class']}"
        identifier = f'{voice}:tsumugi:{namespace}:{origin}:{index}'
        note = Note(start, end, pitch=event['pitch'], strength=1., source='pitch', id=identifier,
                    voice=voice, release_reason=reason, instrument_id=instrument)
        note_from(note.data())
        row = dict(id=identifier, origin=origin, note_index=index if origin == 'baseline' else None,
            recovery_source_index=index if origin == 'recovered' else None, slot_index=raw_note['slot_index'],
            instrument_id=instrument, instrument_class=event['instrument_class'], model_scope=model_scope,
            instrument_semantics='model_scoped_class_not_verified_physical_instrument',
            weight_semantics='unit_weight_not_confidence', source_voice=voice,
            verified_replay_sha256=verified, raw_event=copy.deepcopy(event), raw_note=copy.deepcopy(raw_note),
            evidence=copy.deepcopy(evidence), raw_end_ms=raw_end, candidate_start_ms=start, candidate_end_ms=end,
            sample_to_ms_rounding='nearest_integer_ties_to_even', final_endpoint_model_supported=supported,
            allow_predicted_releases=allow_predicted_releases, model_predicted_release_hypothesis=predicted,
            authored_ln_verified=False, release_reason=reason)
        if event['is_drum']:
            row['decision'] = 'incompatible_drum_prediction'
        else:
            row['decision'] = 'candidate'
            notes.append(note)
        trace.append(row)

    for index, (event, item) in enumerate(zip(baseline, rows)):
        append(event, item['note'], 'baseline', index, item, _endpoint_support(item, identity))
    for event, raw_note, decision in recovered:
        append(event, raw_note, 'recovered', event['recovery_source_index'], decision, False)
    return notes, trace


def adapt_case(input_dir, source, *, voice, allow_predicted_releases=False, recover_filtered_attacks=False):
    """Verify saved evidence before bridging, without neural or label access."""
    folder = Path(input_dir)
    audited = release_audit.audit(folder, source)
    baseline = read_json(folder / 'notes.json')
    classes = load_classes(source, audited['identity'])
    recovery = None
    if recover_filtered_attacks:
        path = folder / 'interval-constructions.json'
        require(sha(path) == audited['evidence_sha256']['interval-constructions.json'], 'Recovery constructions changed')
        recovery = dict(identity=copy.deepcopy(audited['identity']), verified_replay_sha256=data_sha(audited),
            protocol_sha256=PROTOCOL_SHA256,
            **recover_attacks(audited, read_json(path), baseline, classes))
    notes, trace = tsumugi_candidates(audited, baseline, classes, voice=voice,
        allow_predicted_releases=allow_predicted_releases, recovery=recovery)
    return dict(revision=1, runner_sha256=sha(__file__), auditor_sha256=sha(release_audit.__file__),
        identity=copy.deepcopy(audited['identity']), verified_replay_sha256=data_sha(audited),
        evidence_sha256=copy.deepcopy(audited['evidence_sha256']), report_sha256=audited['report_sha256'],
        source_voice=voice, allow_predicted_releases=allow_predicted_releases,
        recover_filtered_attacks=recover_filtered_attacks, baseline_notes=copy.deepcopy(baseline),
        recovery=recovery, candidates=[n.data() for n in notes], trace=trace,
        baseline_modified=False, enabled_in_application=False, neural_inference=False, scoring=False,
        counts=dict(baseline=len(baseline), recovered=len(trace)-len(baseline), candidates=len(notes),
            rejected_drums=sum(row['decision'] == 'incompatible_drum_prediction' for row in trace),
            predicted_release_hypotheses=sum(row['model_predicted_release_hypothesis'] for row in trace)),
        limits=['Candidate filtering is for arrangement only; do not replace raw evaluation denominators.',
                'All non-drum classes are retained, including misclassified instruments; slots are trace metadata.',
                'Default raw ends remain unknown and expression_evidence must reduce them to taps.',
                'Opt-in releases are model duration hypotheses, not acoustic or authored LN truth.'])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, required=True)
    cli.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    cli.add_argument('--voice', choices=VOICES, required=True, help='Known input source, never a reference-label filter')
    cli.add_argument('--allow-predicted-releases', action='store_true', help='Research model-duration hypothesis only')
    cli.add_argument('--recover-filtered-attacks', action='store_true')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve earlier output; choose a new output file')
    result = adapt_case(args.input, args.source, voice=args.voice,
        allow_predicted_releases=args.allow_predicted_releases, recover_filtered_attacks=args.recover_filtered_attacks)
    write_new(args.output, result)
    print(result['counts'], flush=True)


if __name__ == '__main__':
    main()
