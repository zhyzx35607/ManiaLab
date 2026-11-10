import copy

import pytest

from manialab.arrange import parameters
from manialab.domain import Note, UNOBSERVED_RELEASES
from manialab.expression import expression_evidence, propose_expression
from manialab.expression_streams import candidate_signature, stream_articulation_policy
from tests.test_structure_expression import analysis


def piano(at, end, pitch, identity, **extra):
    return Note(at, end, pitch=pitch, id=identity, voice='piano', source='pitch', **extra)


def example():
    return [piano(1000, 5000, 48, 'held'), piano(1000, 2500, 72, 'r1'),
            piano(1500, 3000, 74, 'r2'), piano(2000, 3500, 76, 'r3')]


def sidecar(notes, groups, scope='source', until=6000):
    return dict(revision=1, voice='piano', observed_until_ms=until,
        candidates_sha256=candidate_signature(notes),
        hypotheses=[dict(method='fixture', identity_scope=scope, prediction_sha256='a'*64, groups=groups)])


def test_explicit_streams_preserve_low_sustain_and_high_rearticulation_without_changing_raw_notes():
    notes = example()
    a = analysis(notes, length=6000)
    p = parameters()
    plan = propose_expression(a, p)
    original = copy.deepcopy(a)
    evidence = sidecar(notes, {'held':'left', 'r1':'right', 'r2':'right', 'r3':'right'})
    baseline, _, _ = expression_evidence(a, plan, p)
    result, _, trace = expression_evidence(a, plan, p, stream_evidence=evidence)
    assert [n.end for n in baseline] == [1000, 1000, 1500, 3500]
    assert [n.end for n in result] == [5000, 1000, 1500, 3500]
    assert trace['held']['articulation']['stream']['changed']
    assert a == original


def test_fragment_id_disappearing_is_not_an_observed_musical_rest():
    notes = [piano(t, t+1500, 72+i, str(i)) for i,t in enumerate(range(1000, 5000, 500))]
    local = sidecar(notes, {n.id:'a' if n.start<3000 else 'b' for n in notes}, 'fragment', 7000)
    choices = stream_articulation_policy(notes, local, observed_until_ms=7000)
    assert choices['3']['tap'] and choices['3']['stream']['status'] == 'unknown_source_context'
    assert choices['3']['stream']['observations'][0]['right_identity_context'] == 'unknown_identity'
    global_hint = copy.deepcopy(local['hypotheses'][0])
    global_hint.update(method='whole_source', identity_scope='source', groups={n.id:'right' for n in notes})
    local['hypotheses'].append(global_hint)
    choices = stream_articulation_policy(notes, local, observed_until_ms=7000)
    assert choices['3']['tap']
    assert not choices['7']['tap']  # Keep the baseline last event, not a newly inferred fragment tail.
    assert choices['7']['stream']['status'] == 'grouping_disagreement'


def test_singleton_fragments_cannot_override_a_continuing_global_run():
    notes = example()
    global_groups = {n.id:'right' if n.id != 'held' else 'left' for n in notes}
    evidence = sidecar(notes, global_groups)
    evidence['hypotheses'].append(dict(method='fragments', identity_scope='fragment',
        prediction_sha256='b'*64, groups={n.id:n.id for n in notes}))
    choices = stream_articulation_policy(notes, evidence, observed_until_ms=6000)
    assert not choices['held']['tap']
    assert choices['r1']['tap'] and choices['r2']['tap']
    assert choices['r1']['stream']['status'] == 'grouping_disagreement'


def test_unknown_identity_is_not_merged_and_does_not_claim_complete_group_context():
    notes = example()
    evidence = sidecar(notes, {'held':'left', 'r1':'right', 'r2':None, 'r3':'right'})
    choices = stream_articulation_policy(notes, evidence, observed_until_ms=6000)
    assert choices['held']['tap']
    assert choices['held']['stream']['status'] == 'unknown_source_context'
    assert choices['r2']['stream']['status'] == 'unknown_assignment'


@pytest.mark.parametrize('reason', sorted(UNOBSERVED_RELEASES))
def test_group_labels_cannot_promote_an_unknown_release(reason):
    notes = example()
    notes[0].release_reason = reason
    a = analysis(notes, length=6000)
    p = parameters()
    evidence = sidecar(notes, {'held':'left', 'r1':'right', 'r2':'right', 'r3':'right'})
    result, _, trace = expression_evidence(a, propose_expression(a, p), p, stream_evidence=evidence)
    assert not result[0].hold
    assert trace['held']['reason'] == 'unobserved_release'


@pytest.mark.parametrize('mode', ['sustain', 'onsets', 'omit'])
def test_explicit_section_modes_override_stream_hints(mode):
    notes = example()
    a = analysis(notes, length=6000)
    p = parameters()
    plan = propose_expression(a, p)
    plan['sections'][0]['voices']['piano']['mode'] = mode
    evidence = sidecar(notes, {n.id:'all' for n in notes})
    result, _, _ = expression_evidence(a, plan, p, stream_evidence=evidence)
    baseline, _, _ = expression_evidence(a, plan, p)
    assert result == baseline


def test_section_cuts_do_not_change_stream_context_and_other_instruments_are_unchanged():
    notes = example()
    source = sidecar(notes, {'held':'left', 'r1':'right', 'r2':'right', 'r3':'right'})
    notes.append(Note(1250, 4500, id='strings', voice='other', source='pitch'))
    a = analysis(notes, length=6000)
    p = parameters()
    full, _, _ = expression_evidence(a, propose_expression(a, p), p, stream_evidence=source)
    a['structure']['segments'] = [dict(start_ms=0, end_ms=1700), dict(start_ms=1700, end_ms=6000)]
    split, _, _ = expression_evidence(a, propose_expression(a, p), p, stream_evidence=source)
    assert sorted((n.id, n.start, n.end) for n in split) == sorted((n.id, n.start, n.end) for n in full)
    assert next(n for n in split if n.id == 'strings') == notes[-1]


@pytest.mark.parametrize('change', ['clock', 'tail', 'unknown_id', 'missing_id', 'integer_group', 'bad_scope', 'duplicate_method'])
def test_changed_inputs_and_incomplete_group_maps_are_rejected(change):
    notes = example()
    evidence = sidecar(notes, {n.id:'all' for n in notes})
    if change == 'clock':evidence['observed_until_ms'] += 1
    elif change == 'tail':notes[0].end += 1
    elif change == 'unknown_id':evidence['hypotheses'][0]['groups']['absent'] = 'all'
    elif change == 'missing_id':del evidence['hypotheses'][0]['groups']['r1']
    elif change == 'integer_group':evidence['hypotheses'][0]['groups']['r1'] = 1
    elif change == 'bad_scope':evidence['hypotheses'][0]['identity_scope'] = 'guess'
    else:evidence['hypotheses'].append(copy.deepcopy(evidence['hypotheses'][0]))
    with pytest.raises(ValueError):stream_articulation_policy(notes, evidence, observed_until_ms=6000)
