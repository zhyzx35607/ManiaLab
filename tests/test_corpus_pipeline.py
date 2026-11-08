import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pytest

from manialab.domain import Note,export_osu
from scripts.sayo_charts import fetch_charts
from scripts.snapshot_corpus import build
from scripts.train_layout_network import ln_context,records_and_groups,samples
from scripts import collect_ranked_loved as collector

ROOT=Path(__file__).parents[1]


def test_mirror_ranges_read_exact_charts_and_skip_large_audio():
    stream=io.BytesIO()
    expected={'曲名 [LN].osu':b'chart-a'*5000,'another.osu':b'chart-b'*4000}
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('song.mp3',np.random.default_rng(1).bytes(1_000_000))
        for name,raw in expected.items():z.writestr(name,raw)
    archive=stream.getvalue();transferred=0
    def fetch(url,headers,include_headers):
        nonlocal transferred
        value=headers['Range'].removeprefix('bytes=')
        if value.startswith('-'):a,b=max(0,len(archive)-int(value[1:])),len(archive)-1
        else:a,b=map(int,value.split('-'))
        raw=archive[a:b+1];transferred+=len(raw)
        return raw,{'Content-Range':f'bytes {a}-{b}/{len(archive)}','ETag':'revision'},url
    charts,receipt=fetch_charts(1,fetch)
    assert dict(charts)==expected
    assert receipt['bytes_fetched']==transferred<200_000
    def no_range(*args,**kwargs):return archive,{},'url'
    with pytest.raises(ValueError,match='range'):fetch_charts(1,no_range)


def test_snapshot_new_exclusion_supersedes_old_and_layout_dedup_is_per_key(tmp_path):
    roots=[tmp_path/'old',tmp_path/'new']
    for root in roots:(root/'maps').mkdir(parents=True)
    notes=[Note(1000+i*500,1000+i*500,i%6) for i in range(100)]
    records=[]
    for bid,key in ((1,6),(2,6),(3,10)):
        content=export_osu(notes,key,ROOT/'vendor/mapperatorinator/template.osu').encode()
        (roots[0]/'maps'/f'{bid}.osu').write_bytes(content)
        records.append(dict(beatmap_id=bid,beatmapset_id=bid,keys=key,eligible=True,approved=1,
            sha256=hashlib.sha256(content).hexdigest(),song_key='song',title='x'))
    (roots[0]/'manifest-test.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
    (roots[1]/'manifest-test.jsonl').write_text(json.dumps(records[0]|dict(eligible=False)),encoding='utf-8')
    output=tmp_path/'snapshot';build(roots,output)
    result=[json.loads(line) for line in (output/'manifest-snapshot.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {r['beatmap_id'] for r in result}=={2,3}
    with pytest.raises(ValueError,match='new snapshot'):build(roots,output)


def test_snapshot_cannot_treat_legacy_loved_set_label_as_difficulty_status(tmp_path):
    (tmp_path/'maps').mkdir()
    raw=export_osu([Note(1000+i*500,1000+i*500,i%6) for i in range(100)],6,
                   ROOT/'vendor/mapperatorinator/template.osu').encode()
    (tmp_path/'maps/1.osu').write_bytes(raw)
    row=dict(beatmap_id=1,beatmapset_id=1,keys=6,approved=4,eligible=True,
             sha256=hashlib.sha256(raw).hexdigest(),song_key='fixture')
    (tmp_path/'manifest-test.jsonl').write_text(json.dumps(row))
    build([tmp_path],tmp_path/'snapshot')
    report=json.loads((tmp_path/'snapshot/audit.json').read_text())
    assert report['included']==0
    assert report['exclusions']=={'loved_difficulty_status_unverified':1}


def test_ln_training_samples_keep_actual_hold_state_and_cover_song():
    notes=[]
    for i in range(500):
        start=1000+i*500
        notes.append(Note(start,start+750 if i%4==0 else start,i%6))
    parsed=dict(keys=6,notes=notes,timing=['0,500,4,2,0,100,1,0'])
    a,b=samples(parsed,42,limit=40)
    assert len(a)==len(b)>100
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert ln_context(a).any() and (~ln_context(a)).any()
    for features in (a,b):
        lanes=features[:,:90].reshape(-1,10,9)
        assert not np.any((lanes[:,:,0]>0)&(lanes[:,:,5]>0))
        assert not np.any((lanes[:,:,0]>0)&(lanes[:,:,8]==0))


def test_older_audio_identity_still_groups_newer_metadata(tmp_path):
    (tmp_path/'maps').mkdir();records=[]
    for bid in (1,2,3):
        raw=export_osu([Note(1000,2000,0)],6,ROOT/'vendor/mapperatorinator/template.osu',title=f'song{bid}').encode()
        (tmp_path/'maps'/f'{bid}.osu').write_bytes(raw)
        records.append(dict(beatmap_id=bid,beatmapset_id=bid,keys=6,eligible=True,approved=1,
            sha256=hashlib.sha256(raw).hexdigest(),song_key=f'song{bid}',
            related_audio_hashes=['same-audio'] if bid in (1,2) else []))
    (tmp_path/'manifest-test.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
    grouped,_=records_and_groups(tmp_path)
    assert grouped[0]['group']==grouped[1]['group']!=grouped[2]['group']


@pytest.mark.parametrize('supplement',[False,True])
def test_official_status_refresh_preserves_transfer_source_and_retracts_guests(tmp_path,monkeypatch,supplement):
    for name in ('pages','sets','maps','records','official'):(tmp_path/name).mkdir()
    collector.save_json(tmp_path/'pages/0.json',dict(status=0,endid=0,data=[] if supplement else [dict(sid=9,approved=4)]))
    collector.save_json(tmp_path/'sets/9.json',dict(data=dict(approved=-2 if supplement else 4,bid_data=[])))
    difficulties=[]
    for bid,status in ((1,'loved'),(2,'pending')):
        raw=export_osu([Note(1000+i*500,1000+i*500,i%6) for i in range(100)],6,
                       ROOT/'vendor/mapperatorinator/template.osu').encode()
        (tmp_path/'maps'/f'{bid}.osu').write_bytes(raw)
        collector.save_json(tmp_path/'records'/f'{bid}.json',dict(beatmap_id=bid,beatmapset_id=9,keys=6,
            approved=1,eligible=True,suspected_conversion=False,audit_revision=4,status_scope='mirror_beatmapset',
            sha256=hashlib.sha256(raw).hexdigest(),source='sayobot-original-transfer'))
        difficulties.append(dict(id=bid,mode='mania',cs=6,status=status,checksum=hashlib.md5(raw).hexdigest()))
    collector.save_json(tmp_path/'official/9.json',dict(id=9,ranked=4,beatmaps=difficulties))
    def unexpected_request(*args,**kwargs):raise AssertionError('Cached refresh must not download')
    monkeypatch.setattr(collector,'request',unexpected_request)
    collector.collect(tmp_path,workers=1,include_sets=[9] if supplement else [])
    if supplement:
        collector.collect(tmp_path,workers=1)
        assert json.loads((tmp_path/'supplement-selection.json').read_text())==[9]
    rows={r['beatmap_id']:r for r in map(json.loads,(tmp_path/'manifest-ranked-loved.jsonl').read_text(encoding='utf-8').splitlines())}
    assert rows[1]['source']=='sayobot-original-transfer'
    assert rows[1]['approved']==4 and rows[1]['status_scope']=='beatmap'
    assert rows[2]['eligible'] is False and rows[2]['approved']==-1
    snapshot=tmp_path/'snapshot';build([tmp_path],snapshot)
    assert json.loads((snapshot/'manifest-snapshot.jsonl').read_text(encoding='utf-8'))['beatmap_id']==1
