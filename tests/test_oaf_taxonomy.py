import pytest

from scripts.oaf_taxonomy import (CLASSES, PITCH_CLASSES, SUBCLASS_CLASSES,
    load_mdb_subclasses, prediction_events, score)


def fixture(tmp_path, classes, subclasses):
    coarse, fine = tmp_path/'class.txt', tmp_path/'subclass.txt'
    coarse.write_text(classes); fine.write_text(subclasses)
    return coarse, fine


def test_fixed_mapping_does_not_confuse_different_taxonomies():
    assert SUBCLASS_CLASSES['SST'] == 'snare'
    assert SUBCLASS_CLASSES['TMB'] == 'hi-hat'
    assert SUBCLASS_CLASSES['RDB'] == 'bell'
    assert SUBCLASS_CLASSES['RDC'] == 'ride'
    assert SUBCLASS_CLASSES['CHC'] == SUBCLASS_CLASSES['SPC'] == 'unknown'
    assert PITCH_CLASSES[53] == 'bell' and 52 not in PITCH_CLASSES
    assert set(PITCH_CLASSES.values()) == set(CLASSES)


def test_decimal_rounding_preserves_original_class_clock_and_duplicate_rows(tmp_path):
    paths=fixture(tmp_path,'0.005805 HH\n1.000000 SD\n1.000000 SD\n',
                  '0.005804988 CHH\n1.000000 SD\n1.000000 SD\n')
    result=load_mdb_subclasses(*paths)
    assert len(result['events']) == 3
    assert result['events'][0]['time_ms'] == float('0.005805') * 1000
    assert result['events'][0]['subclass_time_ms'] != result['events'][0]['time_ms']
    assert result['maximum_timestamp_difference_seconds'] == 0.000000012


@pytest.mark.parametrize('classes,subclasses',[
    ('1 KD\n','1.001 KD\n'), ('1 KD\n','1 SD\n'),
    ('1 KD\n2 SD\n','1 KD\n'), ('1 CY\n','1 UNDOCUMENTED\n'),
    ('nan KD\n','nan KD\n'), ('-1 KD\n','-1 KD\n'),
    ('1 KD\n2 SD\n','2 SD\n1 KD\n'),
])
def test_invalid_or_reordered_labels_cannot_be_repaired_by_nearest_match(tmp_path, classes, subclasses):
    with pytest.raises(ValueError):load_mdb_subclasses(*fixture(tmp_path,classes,subclasses))


def test_unknowns_are_retained_but_cannot_score_as_correct():
    refs=[dict(time_ms=10.,kind='unknown'),dict(time_ms=20.,kind='bell')]
    preds=prediction_events([dict(onset=.01,pitch=52),dict(onset=.02,pitch=53)])
    result=score(refs,preds,40)
    assert result['unknown']['reference'] == result['unknown']['predicted'] == 1
    assert result['unknown']['matched'] == 0 and result['unknown']['f1'] == 0
    assert result['unknown']['unmatched_reference'] == result['unknown']['unmatched_predicted'] == [0]
    assert result['bell']['matched'] == 1


def test_duplicate_predictions_stay_in_precision_denominator():
    refs=[dict(time_ms=100.,kind='snare')]
    preds=prediction_events([dict(onset=.1,pitch=38),dict(onset=.1,pitch=38)])
    result=score(refs,preds,40)['snare']
    assert result['matched'] == 1 and result['extra'] == 1
    assert result['precision'] == .5


def test_no_silent_loss_of_unsupported_reference_classes(tmp_path):
    result=load_mdb_subclasses(*fixture(tmp_path,'1 CY\n2 CY\n3 CY\n','1 CHC\n2 SPC\n3 CRC\n'))
    assert result['unknown_reference_count'] == 2
    scored=score(result['events'],[],40)
    assert sum(v['reference'] for v in scored.values()) == 3
    assert scored['crash']['reference'] == 1
