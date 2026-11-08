import copy
import io
from pathlib import Path
import time

import pytest

from manialab.arrange import parameters
from manialab.domain import Note, parse_osu
from manialab.editing import edit_notes
from manialab.server import create_app
from manialab.storage import Projects

ROOT=Path(__file__).resolve().parents[1]


def candidate(mode='audio'):
    return dict(mode=mode,notes=[Note(1000,2000,0,id='a').data(),Note(1500,1500,1,id='b').data()],
        settings=parameters(),bpm=120,offset=0,division=4,audio_shift_ms=0,timing_revision=3,
        created=1.,report={},selection={},timing_confirmed=True)


def test_edit_snaps_both_endpoints_without_mutating_original():
    v=candidate();before=copy.deepcopy(v)
    result,log=edit_notes(v,dict(op='update',note_id='a',values=dict(start=1118,end=2376,lane=4)),5000)
    n=next(n for n in result if n.id=='a')
    assert (n.start,n.end,n.lane)==(1125,2375,4)
    assert log['before']['start']==1000 and log['after']['start']==1125
    assert v==before


@pytest.mark.parametrize('changes', [dict(lane=1),dict(start=-1),dict(end=999),dict(lane=10),dict(start=1000.1),dict(end=5001)])
def test_invalid_or_overlapping_edit_is_atomic(changes):
    v=candidate();before=copy.deepcopy(v)
    with pytest.raises(ValueError):
        edit_notes(v,dict(op='update',note_id='a',values=changes),5000)
    assert v==before


def test_remap_preserves_raw_sound_and_only_changes_lane():
    v=candidate('remap');v['notes'][0]['raw']='25,192,1000,128,2,2000:2:3:40:custom.wav'
    result,_=edit_notes(v,dict(op='update',note_id='a',values=dict(lane=5)))
    assert result[0].raw==v['notes'][0]['raw'] and result[0].lane==5
    for payload in [dict(op='delete',note_id='a'),dict(op='add',values=dict(start=3000,end=3000,lane=1)),
                    dict(op='update',note_id='a',values=dict(end=2500))]:
        with pytest.raises(ValueError,match='严格重定轨'):
            edit_notes(v,payload)


def test_add_delete_and_explicit_tap_are_distinct_operations():
    v=candidate()
    result,log=edit_notes(v,dict(op='add',values=dict(start=3001,end=3501,lane=2)),5000)
    assert len(result)==3 and (result[-1].start,result[-1].end)==(3000,3500)
    assert log['after']['source']=='manual'
    result,_=edit_notes(v,dict(op='update',note_id='a',values=dict(end=1000)),5000)
    assert result[0].hold is False
    with pytest.raises(ValueError,match='不足一格'):
        edit_notes(v,dict(op='update',note_id='a',values=dict(end=1002)),5000)
    result,_=edit_notes(v,dict(op='delete',note_id='a'))
    assert [n.id for n in result]==['b']


def test_http_edit_persists_new_candidate_and_exports_actual_correction(tmp_path):
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    project=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'audio.wav')},headers=headers).json
    project['analysis']=dict(revision=6,duration=5,candidates=[dict(sentinel='original')])
    project['variants']=[candidate()];project['active']=0
    store=Projects(tmp_path/'projects');store.save(project);base='/api/projects/'+project['id']
    project=client.get(base).json
    def run(payload):
        job=client.post(base+'/edit',json=payload,headers=headers).json
        for _ in range(200):
            result=client.get('/api/jobs/'+job['id']).json
            if result['status'] in ('done','error'):return result
            time.sleep(.01)
        raise AssertionError('edit timed out')
    payload=dict(op='update',note_id='a',values=dict(start=1118,end=2376,lane=4),base_created=1.)
    assert client.post(base+'/edit',json=payload).status_code==403
    try:
        assert run(payload)['status']=='done'
        saved=client.get(base).json
        assert saved['analysis']==project['analysis'] and saved['variants'][0]==project['variants'][0]
        assert saved['active']==1 and len(saved['variants'][1]['edits'])==1
        assert saved['variants'][1]['report']['ln_patterns']['counts']['releases']==1
        text=client.get(base+'/export/osu').data.decode()
        n=next(n for n in parse_osu(text)['notes'] if n.lane==4)
        assert (n.start,n.end)==(1125,2375)
        # A delayed edit from the old version must not touch the current one.
        failed=run(payload)
        assert failed['status']=='error' and '版本已经改变' in failed['message']
        assert client.get(base).json==saved
        restored=client.post(base+'/variant',json={'index':0},headers=headers).json
        assert restored['variants'][restored['active']]['notes']==project['variants'][0]['notes']
        # The corrected candidate is still available after reopening the project.
        assert store.load(project['id'])['variants'][1]['notes']==saved['variants'][1]['notes']
    finally:
        app.extensions['executor'].shutdown()
