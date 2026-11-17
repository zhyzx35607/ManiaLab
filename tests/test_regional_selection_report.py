"""Regional reports distinguish the whole-song proposal from saved output."""
from copy import deepcopy
from pathlib import Path

from manialab.arrange import parameters
from manialab.domain import Note, note_from, parse_osu, export_osu
from manialab.generation import generate_variant


TEMPO = dict(bpm=120, offset=0, division=4, audio_shift_ms=0)


def state_for(notes):
    return dict(analysis=dict(revision=13, duration=5, candidates=[n.data() for n in notes],
        phrase_candidates=[], timing_attacks=[], structure=dict(boundaries=[0,3000,5000],
            segments=[dict(start_ms=0,end_ms=3000),dict(start_ms=3000,end_ms=5000)])),
        variants=[], active=-1)


def generate(state, prefs=None, region=None):
    generate_variant(state,dict(mode='audio',timing_confirmed=True,**({'region':region} if region else {})),
        parameters(dict(ln_ratio=1,**(prefs or {}))),TEMPO,lambda *_:None,lambda:False)
    return state['variants'][-1]


def test_muted_locked_note_is_retained_and_not_reported_as_final_omission():
    state=state_for([Note(1000,1800,id='locked',source='pitch',voice='piano'),
                     Note(3500,4200,id='region',source='pitch',voice='bass')])
    old=deepcopy(generate(state))
    variant=generate(state,dict(piano_weight=0),[3000,5000])
    report=variant['selection']
    assert report['selected']==len(variant['notes'])==2
    assert report['scope']=='regional_output'
    assert report['locked']==1 and report['generated']==1
    assert report['selected_voices']=={'piano':1,'bass':1}
    assert report['actual_ln_ratio']==1
    records={r['id']:r for r in report['decisions']}
    assert records['locked']==dict(id='locked',voice='piano',decision='selected',
        reason='locked_existing',chart_start=1000,chart_end=1750)
    assert records['region']['reason']=='regional_regenerated'
    assert not any(r['decision']=='omitted' for r in report['decisions'])
    assert 'omission_reasons' not in report and 'selection_solver' not in report
    proposal=report['proposal']
    assert proposal['scope']=='full_song_proposal' and proposal['selected']==1
    assert proposal['omission_reasons']=={'voice_not_selected':1}
    assert next(r for r in proposal['decisions'] if r['id']=='locked')['decision']=='omitted'
    assert report['structure']['boundaries']==[0,3000,5000]
    assert state['variants'][0]==old
    before={n['id']:(n['start'],n['end'],n['lane']) for n in old['notes']}
    assert {n['id']:(n['start'],n['end'],n['lane']) for n in variant['notes']}==before
    content=export_osu([note_from(n) for n in variant['notes']],
        10,Path(__file__).parents[1]/'vendor/mapperatorinator/template.osu',bpm=120,offset=0)
    assert [(n.start,n.end) for n in parse_osu(content)['notes']]==[(1000,1750),(3500,4250)]


def test_manual_and_reanalyzed_locks_have_no_fresh_raw_decision_attached():
    state=state_for([Note(1000,1800,id='locked',source='pitch',voice='piano'),
                     Note(3500,4200,id='region',source='pitch',voice='bass')])
    old=generate(state)
    old['notes'].append(Note(2500,2500,id='manual',source='manual',lane=9).data())
    state['analysis']['candidates'][0]['end']=2200
    variant=generate(state,region=[3000,5000])
    report=variant['selection']
    assert report['selected']==3 and report['locked']==2 and report['generated']==1
    assert report['actual_ln_ratio']==round(2/3,4)
    records={r['id']:r for r in report['decisions']}
    assert records['manual']['reason']=='locked_existing'
    assert 'start' not in records['locked'] and 'end' not in records['locked']
    assert next(r for r in report['proposal']['decisions'] if r['id']=='locked')['end']==2200
    assert variant['source_evidence']['locked']['detected']['end']==1800
    assert 'manual' not in variant['source_evidence']
    assert 'proposal' not in old['selection']


def test_regional_remap_keeps_actual_counts_without_inventing_acoustic_decisions():
    notes=[Note(1000,1800,lane=0,id='left',source='import'),
           Note(3500,4200,lane=1,id='right',source='import')]
    state=dict(input_notes=[n.data() for n in notes],input_keys=10,variants=[],active=-1)
    prefs=parameters()
    for extra in ({},{'region':[3000,5000]}):
        generate_variant(state,dict(mode='remap',**extra),prefs,TEMPO,lambda *_:None,lambda:False)
    variant=state['variants'][-1]
    assert variant['selection']['selected']==2
    assert variant['selection']['locked']==1 and variant['selection']['generated']==1
    assert variant['selection']['scope']=='regional_output'
    assert 'source_evidence' not in variant
    assert [(n['start'],n['end']) for n in variant['notes']]==[(1000,1800),(3500,4200)]


def test_regional_report_survives_save_json_export_and_later_manual_edit(tmp_path):
    import io
    from manialab.server import create_app
    from tests.test_server import wait_job

    root=Path(__file__).parents[1]
    app=create_app(root,tmp_path)
    client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    try:
        state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
        state['analysis']=state_for([Note(1000,1800,id='locked',source='pitch',voice='piano'),
            Note(3500,4200,id='region',source='pitch',voice='bass')])['analysis']
        state['analysis']['tempo']=dict(bpm=120,offset=0,division=4,stable=True)
        app.extensions['projects'].save(state)
        base=f"/api/projects/{state['id']}"
        payload=dict(mode='audio',settings=dict(ln_ratio=1),timing_confirmed=True,**TEMPO)
        assert wait_job(client,client.post(base+'/generate',json=payload,headers=headers).json)['status']=='done'
        payload.update(settings=dict(ln_ratio=1,piano_weight=0),region=[3000,5000])
        assert wait_job(client,client.post(base+'/generate',json=payload,headers=headers).json)['status']=='done'
        saved=client.get(base).json
        current=saved['variants'][saved['active']]
        exported=client.get(base+'/export/json').json['variant']
        assert exported['selection']==current['selection']
        assert exported['selection']['selected']==2
        assert exported['selection']['proposal']['selected']==1
        text=client.get(base+'/export/osu').data.decode()
        assert 'Bookmarks: 3000' in text
        assert [(n.start,n.end) for n in parse_osu(text)['notes']]==[(1000,1750),(3500,4250)]
        changed=client.post(base+'/edit',json=dict(op='update',note_id='locked',
            base_created=current['created'],values={'end':2000}),headers=headers).json
        assert wait_job(client,changed)['status']=='done'
        edited=client.get(base+'/export/json').json['variant']
        assert edited['region'] is None and 'proposal' not in edited['selection']
        assert edited['source_evidence']==current['source_evidence']
        assert client.get(base).json['variants'][-2]==current
    finally:
        app.extensions['executor'].shutdown()
