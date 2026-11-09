"""Controls for sustained-event identity and masked local notation targets."""
from dataclasses import asdict
import copy
from fractions import Fraction

import numpy as np
import pytest

from scripts.piano_event_training import (_geometry, build_piece_items,
                                         load_event_items, read_meter_map)
from scripts.piano_musicxml_events import parse_musicxml


def note(step='C', duration=1, *, ties=(), voice='1', staff=1, extra=''):
    sound = ''.join(f'<tie type="{tag}"/>' for tag in ties)
    return (f'<note>{extra}<pitch><step>{step}</step><octave>4</octave></pitch>'
            f'<duration>{duration}</duration>{sound}<voice>{voice}</voice><staff>{staff}</staff></note>')


def xml(*measures, meter='<beats>4</beats><beat-type>4</beat-type>'):
    signature = f'<time>{meter}</time>' if meter is not None else ''
    body = ''.join(f'<measure number="{i+1}">' +
                   (f'<attributes><divisions>1</divisions>{signature}</attributes>' if i == 0 else '') +
                   content + '</measure>' for i, content in enumerate(measures))
    return f'<score-partwise><part id="P1">{body}</part></score-partwise>'.encode()


def prepare(data, **kwargs):
    source = asdict(parse_musicxml(data))
    return build_piece_items(source, read_meter_map(data, source), piece_id='control', **kwargs)


def pair_labels(item):
    p = item['payload']
    return {tuple(edge): (bool(y), bool(mask)) for edge, y, mask in
            zip(p['potential_edges'].T, p['voice'], p['voice_mask'])}


def test_tie_is_one_node_and_terminal_local_voice_controls_its_successor():
    item = prepare(xml(note(duration=4, ties=('start',)),
        note(ties=('stop',), voice='7') + note('E', voice='7') + note('G', voice='1')))[0]
    p = item['payload']
    assert len(p['notes']) == 3
    assert list(p['notes']['onset_div']) == [0, 5, 6]
    assert p['notes']['duration_div'][0] == 5
    assert len(item['metadata']['written_ids'][0]) == 2
    assert pair_labels(item)[(0, 1)] == (True, True)
    assert pair_labels(item)[(0, 2)] == (False, True)
    assert not p['chord_mask'].any()


def test_equal_pitch_without_tie_is_two_attacks():
    item = prepare(xml(note() + note()))[0]
    assert len(item['payload']['notes']) == 2
    assert pair_labels(item)[(0, 1)] == (True, True)


def test_reused_voice_number_in_next_bar_is_unlabelled_not_negative():
    item = prepare(xml(note(duration=4), note('D', duration=4)))[0]
    assert pair_labels(item)[(0, 1)] == (False, False)


def test_unknown_note_masks_local_neighbors_not_only_its_own_node():
    item = prepare(xml(note('D') + note('C', ties=('start',)) + note('E')))[0]
    assert not item['payload']['voice_mask'].any()
    assert item['metadata']['bad_part_measures'] == [['P1', 1]]


def test_excluded_grace_masks_the_bar_instead_of_skipping_to_a_fake_successor():
    item = prepare(xml(note() + note('D', extra='<grace/>') + note('E')))[0]
    assert len(item['payload']['notes']) == 2
    assert not item['payload']['voice_mask'].any()


def test_staff_loss_requires_stable_two_staff_notation():
    item = prepare(xml(note(ties=('start',)) + note(ties=('stop',), staff=2) + note('E', staff=2)))[0]
    assert list(item['payload']['staff_mask']) == [False, True]
    three = prepare(xml(note(staff=1) + note('D', staff=2) + note('E', staff=3)))[0]
    assert not three['payload']['staff_mask'].any()


def test_long_carry_preserves_true_attack_and_duration_across_windows():
    measures = [note(duration=4, ties=('start',))]
    measures += [note(duration=4, ties=('stop', 'start')) for _ in range(4)]
    measures += [note(duration=2, ties=('stop',), voice='7') + note('E', duration=2, voice='7')]
    measures += [note('F', duration=4), note('G', duration=4)]
    items = prepare(xml(*measures))
    assert len(items) == 2
    first, second = (item['payload'] for item in items)
    assert first['event_ids'][0] == second['event_ids'][0]
    assert (first['notes']['onset_div'][0], second['notes']['onset_div'][0]) == (0, 0)
    assert (first['notes']['duration_div'][0], second['notes']['duration_div'][0]) == (22, 22)
    assert first['carry_out'][0] and second['carry_in'][0]
    assert first['core_mask'][0] and not second['core_mask'][0]
    assert pair_labels(items[1])[(0, 1)] == (True, True)
    positive_owners = [tuple(item['payload']['event_ids'][edge]) for item in items
        for edge, yes, mask in zip(item['payload']['potential_edges'].T,
                                 item['payload']['voice'], item['payload']['voice_mask']) if yes and mask]
    assert len(positive_owners) == len(set(positive_owners))


def test_geometry_is_full_graph_induced_subgraph_without_rest_teleportation():
    events = [{'start_tick': 0, 'end_tick': 1}, {'start_tick': 2, 'end_tick': 3},
              {'start_tick': 4, 'end_tick': 5}]
    whole = _geometry(events, [0, 1, 2])
    subset = _geometry(events, [0, 2])
    assert subset.shape == (3, 0)
    assert set(map(tuple, whole.T)) == {(0, 1, 3), (1, 2, 3)}
    # No future onset exists for the last event; infinity minima must not
    # manufacture an edge back to all previous notes.
    assert not (whole[0] == 2).any()


def test_geometry_overlap_and_boundary_relations_are_not_retimed():
    events = [{'start_tick': 0, 'end_tick': 4}, {'start_tick': 0, 'end_tick': 1},
              {'start_tick': 1, 'end_tick': 2}, {'start_tick': 4, 'end_tick': 5}]
    edges = set(map(tuple, _geometry(events, list(range(4))).T))
    assert {(0, 1, 0), (1, 0, 0), (0, 2, 2), (1, 2, 1), (0, 3, 1)} <= edges


def test_pickup_keeps_nominal_meter_and_meter_change_uses_quarters():
    change = '<attributes><time><beats>6</beats><beat-type>8</beat-type></time></attributes>'
    item = prepare(xml(note(), change + note('D', duration=3)))[0]
    p = item['payload']['notes']
    assert list(p['ts_beats']) == [4., 3.]
    assert list(p['duration_beat']) == [1., 3.]
    assert list(p['onset_beat']) == [0., 1.]


def test_additive_time_signature_is_exact_quarter_length():
    data = xml(note(), meter='<beats>3+2</beats><beat-type>8</beat-type>')
    source = asdict(parse_musicxml(data))
    assert read_meter_map(data, source) == {('P1', 1): Fraction(5, 2)}


@pytest.mark.parametrize('meter', [None, '<senza-misura/>',
    '<beats>4</beats><beat-type>3</beat-type>',
    '<beats>0</beats><beat-type>4</beat-type>'])
def test_unsupported_meter_rejects_instead_of_inventing_four_four(meter):
    data = xml(note(), meter=meter)
    source = asdict(parse_musicxml(data))
    with pytest.raises(ValueError):
        read_meter_map(data, source)


def test_mid_measure_signature_rejects():
    data = xml(note() + '<attributes><time><beats>3</beats><beat-type>4</beat-type></time></attributes>' + note('D'))
    source = asdict(parse_musicxml(data))
    with pytest.raises(ValueError, match='Mid-measure'):
        read_meter_map(data, source)


def test_test_partition_cannot_be_opened(tmp_path):
    with pytest.raises(ValueError, match='Only train/validation'):
        load_event_items(tmp_path, 'test')


def test_all_unknown_pair_targets_have_false_loss_masks():
    item = prepare(xml(note(ties=('start',)) + note(ties=('stop', 'start')) + note('E')))[0]
    p = item['payload']
    assert not p['voice_mask'].any()
    assert not p['voice'].any()
    assert not p['staff_mask'].any()


def test_label_changes_do_not_change_input_nodes_features_or_geometry():
    data = xml(note() + note('D') + note('E', staff=2))
    source = asdict(parse_musicxml(data))
    changed = copy.deepcopy(source)
    for row in changed['written']:
        row['note']['voice'] = str(row['note']['start_tick'] + 7)
        row['note']['staff'] = 3
    changed['representation']['events'][1]['supervision_eligible'] = False
    original = build_piece_items(source, read_meter_map(data, source), piece_id='a')[0]['payload']
    altered = build_piece_items(changed, read_meter_map(data, changed), piece_id='a')[0]['payload']
    for field in ('notes', 'event_ids', 'observed_edges', 'potential_edges', 'chord_edges'):
        assert np.array_equal(original[field], altered[field])
    assert original['voice_mask'].any() and not altered['voice_mask'].any()


def test_long_empty_rest_context_does_not_create_an_empty_training_graph():
    rest = '<note><rest/><duration>4</duration></note>'
    items = prepare(xml(note(duration=4), *([rest]*16), note('E', duration=4)))
    assert items and all(len(item['payload']['notes']) for item in items)
    owned = [identifier for item in items for identifier, own in
             zip(item['payload']['event_ids'], item['payload']['core_mask']) if own]
    assert len(owned) == 2 and len(set(owned)) == 2
