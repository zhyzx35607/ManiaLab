import pytest


def test_cold_prefix_loss_covers_one_complete_event_without_context_or_next_note():
    from scripts.mapper_training_context import cold_prefix_positions
    labels=[-100,-100,5,26,1648,2877,3897,3980,4071,67,1649,2878,4071,6]
    assert cold_prefix_positions(labels,(9,1647),{4071,4087})==list(range(3,9))
    assert cold_prefix_positions([-100,5,6],(9,1647),{4071})==[]
    with pytest.raises(ValueError,match='Incomplete'):
        cold_prefix_positions([5,26,1648],(9,1647),{4071})
from scripts.prepare_mapper_training import grouped_split


def paired_rows():
    rows=[]
    for key in range(6,11):
        for band in ('tap','hybrid','ln'):
            for variation in range(3):
                i=len(rows)+1
                rows.append(dict(beatmap_id=i,beatmapset_id=i,keys=key,band=band,
                    audio_sha256=f'audio-{i}',song_keys=[f'song-{i}']))
    return rows


def test_mapper_split_keeps_related_audio_songs_and_sets_together():
    rows=paired_rows()
    rows[1]['audio_sha256']=rows[0]['audio_sha256']
    rows[2]['song_keys']=rows[1]['song_keys']
    rows[3]['beatmapset_id']=rows[2]['beatmapset_id']
    result=grouped_split(rows)
    lookup={r['beatmap_id']:r for r in result}
    assert len({lookup[i]['group'] for i in (1,2,3,4)})==1
    assert len({lookup[i]['split'] for i in (1,2,3,4)})==1
    for split in ('train','validation'):
        local=[r for r in result if r['split']==split]
        assert {r['keys'] for r in local}=={6,7,8,9,10}
        assert any(r['keys']==10 and r['band']=='ln' for r in local)
    assert grouped_split(rows)==result


def test_mapper_split_refuses_to_validate_ln_on_the_only_training_song():
    rows=paired_rows()
    for r in rows:
        if r['keys']==10 and r['band']=='ln':r['song_keys']=['only-ln-song']
    with pytest.raises(ValueError,match='10K ln'):grouped_split(rows)


def test_audio_comparison_cannot_count_one_prediction_twice():
    from scripts.probe_mapper_adapter import event_similarity
    result=event_similarity([(1000,), (1020,)], [(1010,)], [30])
    assert result['matched']==1 and result['recall']==.5
    assert result['precision']==1
    # Match both even when the locally closest choice would leave one unmatched.
    result=event_similarity([(1000,), (1040,)], [(1020,), (980,)], [30])
    assert result['matched']==2


def test_hold_comparison_requires_both_endpoints_and_excludes_cut_holds():
    from manialab.domain import Note
    from scripts.probe_mapper_adapter import timing_comparison
    human=[Note(1000,2000,0),Note(3000,5000,1)]
    predicted=[Note(1001,2500,5),Note(3001,5000,7)]
    result=timing_comparison(human,predicted,0,4000)
    assert result['onset_groups']['matched']==2
    assert result['complete_hold_spans']['reference']==1
    assert result['complete_hold_spans']['matched']==0


def test_context_windows_cover_targets_once_and_retain_sustain_history():
    from scripts.mapper_training_context import target_windows
    windows=list(target_windows(9000,2047,1023))
    assert windows[0]==(0,0,1023)
    assert windows[-1][2]==9000
    assert all(a[2]==b[1] for a,b in zip(windows,windows[1:]))
    assert all(end-begin<=2047 for begin,target,end in windows)
    assert all((target-begin)*8>=8000 for begin,target,end in windows[2:])


def test_tail_time_metric_does_not_measure_only_easy_type_tokens():
    from scripts.mapper_training_context import typed_time_positions
    starts=dict(t=9,hold_note=100,hold_note_end=101,circle=102)
    ends=dict(t=90)
    labels=[-100,20,95,100,30,96,101,40,94,102]
    assert typed_time_positions(labels,starts,ends)==dict(ln_head_time=[1],ln_tail_time=[4],tap_time=[7])


def test_release_only_and_single_attack_windows_remain_training_candidates():
    from scripts.mapper_training_context import has_musical_target
    musical={4071,4087,4088,4090}
    assert has_musical_target([-100,80,4088],musical)  # A release without a new head.
    assert has_musical_target([-100,90,4090],musical)  # A continuing long hold.
    assert has_musical_target([-100,100,4071],musical) # One sparse attack.
    assert not has_musical_target([-100,3,4,5,6,2],musical)


def test_temporal_split_excludes_training_history_from_validation_audio():
    from scripts.mapper_training_context import window_in_ranges
    ranges = [[188286, 283840]]
    # The labels start inside validation but the preceding audio/history does not.
    assert not window_in_ranges(dict(start_ms=180000, target_ms=188300, end_ms=196500), ranges)
    assert window_in_ranges(dict(start_ms=188286, target_ms=196500, end_ms=204684), ranges)
    assert not window_in_ranges(dict(start_ms=275000, target_ms=283000, end_ms=291184), ranges)
    assert not window_in_ranges(dict(start_ms=0, target_ms=5000, end_ms=12000), [[0, 6000], [8000, 15000]])
    with pytest.raises(ValueError):
        window_in_ranges(dict(start_ms=0, target_ms=0, end_ms=100), [[200, 100]])


def test_source_training_schedule_never_silently_omits_a_keycount():
    from scripts.mapper_training_context import key_schedule
    assert key_schedule({10}, [10]) == [10]
    assert key_schedule({6, 7, 8, 9, 10}).count(10) == 4
    with pytest.raises(ValueError):
        key_schedule({10})
    with pytest.raises(ValueError):
        key_schedule({6, 10}, [10])


def test_cold_start_discards_orphan_release_groups_without_inventing_heads():
    from scripts.mapper_training_context import cold_start_group_indices
    groups=[('hold_note_sustain',0,[0,1]),('hold_note_end',0,[2,3]),
            ('circle',1,[4,5]),('hold_note',0,[6,7]),
            ('hold_note_sustain',0,[8,9]),('hold_note_end',0,[10,11])]
    assert cold_start_group_indices(groups)==list(range(4,12))


@pytest.mark.parametrize('types_first',[False,True])
def test_cold_start_sustain_cannot_attach_to_the_following_tap(types_first):
    from scripts.mapper_training_context import cold_start_event_indices
    groups=[('hold_note_sustain',0),('circle',1),('hold_note',2),
            ('hold_note_sustain',2),('hold_note_end',0),('hold_note_end',2)]
    events=[]
    for index,(kind,column) in enumerate(groups):
        body=[('t',1000+index*100),('column',column)]
        events.extend([(kind,0)]+body if types_first else body+[(kind,0)])
    keep=cold_start_event_indices(events,{'circle','hold_note','hold_note_end'},types_first)
    assert keep==list(range(3,12))+list(range(15,18))
