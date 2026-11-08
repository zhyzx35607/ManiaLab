import hashlib
import json

from scripts.train_layout_network import records_and_groups


def corpus(tmp_path, entries):
    (tmp_path/'maps').mkdir()
    records=[]
    for i,extra in enumerate(entries):
        raw=f'chart {i}\n'.encode()
        (tmp_path/'maps'/f'{i}.osu').write_bytes(raw)
        records.append(dict(eligible=True,approved=1,sha256=hashlib.sha256(raw).hexdigest(),beatmap_id=i,
            beatmapset_id=100+i,keys=10,song_key=f'song{i}',title='title',version='10K',**extra))
    (tmp_path/'manifest-test.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
    return records


def test_song_audio_and_set_connections_cannot_cross_training_splits(tmp_path):
    corpus(tmp_path,[{'audio_hash':'same'},{'audio_hash':'same'},{'audio_hash':'other'}])
    path=tmp_path/'manifest-test.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()]
    records[2]['beatmapset_id']=records[1]['beatmapset_id']
    path.write_text('\n'.join(json.dumps(r) for r in records))
    grouped,excluded=records_and_groups(tmp_path)
    assert len({r['group'] for r in grouped})==1 and excluded=={}


def test_wrong_hand_layout_and_overwritten_chart_are_excluded(tmp_path):
    corpus(tmp_path,[{},{}])
    path=tmp_path/'manifest-test.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()]
    records[0]['title']='4+6K Pack 1'
    path.write_text('\n'.join(json.dumps(r) for r in records))
    (tmp_path/'maps/1.osu').write_text('different revision')
    grouped,excluded=records_and_groups(tmp_path)
    assert not grouped
    assert excluded=={'asymmetric_10k':1,'chart_hash_mismatch':1}


def test_missing_or_unranked_status_cannot_enter_core_training(tmp_path):
    records=corpus(tmp_path,[{},{}])
    records[0].pop('approved');records[1]['approved']=-2
    (tmp_path/'manifest-test.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
    grouped,excluded=records_and_groups(tmp_path)
    assert not grouped and excluded['status_not_ranked_loved']==2
