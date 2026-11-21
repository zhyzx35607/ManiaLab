"""Regional reports distinguish the whole-song proposal from saved output."""
from copy import deepcopy
import hashlib
from pathlib import Path

from manialab.arrange import parameters
from manialab.domain import Note, note_from, parse_osu, export_osu
from manialab.generation import generate_variant
from manialab.selection_report import resolve_generation_origin


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
    assert old['generation_origin']==dict(kind='full_song_generation')
    assert variant['generation_origin']==dict(kind='regional_generation',region=[3000,5000])
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


def test_legacy_edited_candidate_origin_is_backfilled_from_available_parent():
    root=dict(created=1.0,region=[3000,5000])
    child=dict(created=2.0,region=None,edits=[dict(parent_created=1.0)])
    assert resolve_generation_origin(child,[root,child])==dict(
        kind='regional_generation',region=[3000,5000])
    assert resolve_generation_origin(child,[child])==dict(kind='unknown')


def test_variant_switch_restores_expression_for_regional_rewrite_and_json_export(tmp_path):
    import io
    from manialab.expression import propose_expression
    from manialab.recommendations import evidence_signature
    from manialab.server import create_app
    from tests.test_server import wait_job

    root=Path(__file__).parents[1]
    app=create_app(root,tmp_path)
    client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    try:
        imported=client.post('/api/import',headers=headers,
            data={'files':(io.BytesIO(b'placeholder'),'song.wav')})
        assert imported.status_code==200,imported.json
        base=f"/api/projects/{imported.json['id']}"
        prefs=parameters({'keys':10,'ln_ratio':.55,'min_ln':240})
        candidates=[
            Note(1000,1500,pitch=60,source='pitch',voice='vocals',id='vocal-early').data(),
            Note(2500,3000,pitch=64,source='pitch',voice='piano',id='piano-early').data(),
            Note(7000,7500,pitch=67,source='pitch',voice='vocals',id='vocal-late').data(),
            Note(8500,9000,pitch=72,source='pitch',voice='piano',id='piano-late').data(),
        ]
        analysis=dict(revision=3,duration=12.,tempo=dict(bpm=120,offset=0,division=4,stable=True),
            candidates=candidates,phrase_candidates=[],timing_attacks=[],
            structure=dict(segments=[dict(start_ms=0,end_ms=6000),dict(start_ms=6000,end_ms=12000)]))
        def expression(vocal_mode):
            result=propose_expression(analysis,prefs)
            result['source']='manual'
            for section in result['sections']:
                section['voices']['vocals']['mode']=vocal_mode
                section['voices']['piano']['mode']='onsets'
            return result

        omit=expression('omit')
        onsets=expression('onsets')
        state=imported.json
        state.update(settings=prefs,analysis=analysis,expression=omit,
            plan=dict(source='local-audio-recommendation',preferences=prefs,
                segments=[dict(start_ms=0,end_ms=6000,preferences={}),
                          dict(start_ms=6000,end_ms=12000,preferences={})],
                keys=10,bpm=120,evidence_signature=evidence_signature(analysis)))
        app.extensions['projects'].save(state)
        payload=dict(mode='audio',settings=prefs,timing_confirmed=True,**TEMPO)

        def generate(extra=None):
            submitted=client.post(base+'/generate',headers=headers,json=payload|(extra or {}))
            assert submitted.status_code==200,submitted.json
            result=wait_job(client,submitted.json)
            assert result['status']=='done',result
            return client.get(base).json

        first=generate()
        assert {note['id'] for note in first['variants'][0]['notes']}=={'piano-early','piano-late'}
        assert first['variants'][0]['selection']['expression']==omit

        state=app.extensions['projects'].load(imported.json['id'])
        state['expression']=onsets
        app.extensions['projects'].save(state)
        second=generate()
        assert {note['id'] for note in second['variants'][1]['notes']}=={
            'vocal-early','piano-early','vocal-late','piano-late'}
        assert second['variants'][1]['selection']['expression']==onsets

        selected=client.post(base+'/variant',headers=headers,json={'index':0})
        assert selected.status_code==200,selected.json
        assert selected.json['expression']==omit
        exported=client.get(base+'/export/json',headers=headers)
        assert exported.status_code==200
        assert exported.json['expression']==omit
        assert exported.json['variant']['selection']['expression']==omit

        rewritten=generate({'region':[0,6000]})
        regional=rewritten['variants'][rewritten['active']]
        assert regional['selection']['expression']==omit
        assert {note['voice'] for note in regional['notes']}=={'piano'}
        exported=client.get(base+'/export/json',headers=headers)
        assert exported.json['expression']==omit
        assert exported.json['variant']['selection']['expression']==omit
    finally:
        app.extensions['executor'].shutdown()


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


def test_regional_phrase_bookmark_and_osz_survive_edit_and_reopen(tmp_path):
    import io
    import zipfile

    from manialab.server import create_app
    from tests.test_server import wait_job

    root = Path(__file__).parents[1]
    audio = b'synthetic regional export audio'
    structure = dict(
        boundaries=[0, 2500, 5000],
        segments=[
            dict(start_ms=0, end_ms=2500, label='verse'),
            dict(start_ms=2500, end_ms=5000, label='chorus'),
        ],
    )
    candidates = [
        Note(1000, 1500, id='locked', source='pitch', voice='piano',
             release_reason='frame_decay'),
        Note(3000, 3125, id='anchor', source='pitch', voice='vocals',
             release_reason='frame_decay'),
        Note(3500, 3750, id='support', source='pitch', voice='vocals',
             release_reason='frame_decay'),
    ]
    phrase = Note(3000, 4125, id='anchor', source='pitch', voice='vocals',
        phrase=True, support=('support',), support_coverage=1,
        release_reason='vocal_activity_decay')
    analysis = dict(
        revision=13,
        duration=5,
        audio_sha256=hashlib.sha256(audio).hexdigest(),
        candidates=[note.data() for note in candidates],
        phrase_candidates=[phrase.data()],
        timing_attacks=[],
        structure=structure,
        tempo=dict(bpm=120, offset=0, division=4, stable=True),
    )
    settings = dict(ln_ratio=1, phrase_ln=1, min_ln=100)
    app = create_app(root, tmp_path)
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(audio), 'song.wav')})
        assert imported.status_code == 200, imported.json
        state = imported.json
        pid = state['id']
        base = '/api/projects/' + pid
        state['analysis'] = analysis
        app.extensions['projects'].save(state)

        automatic = client.post(base + '/expression', headers=headers,
            json=dict(action='automatic', settings=settings))
        assert automatic.status_code == 200, automatic.json
        manual_expression = automatic.json['expression']
        manual_expression['sections'][0]['voices']['piano']['mode'] = 'sustain'
        manual_expression['sections'][1]['voices']['vocals']['mode'] = 'phrase'
        saved_expression = client.post(base + '/expression', headers=headers,
            json={'expression': manual_expression})
        assert saved_expression.status_code == 200, saved_expression.json
        manual_expression = saved_expression.json['expression']
        assert manual_expression['source'] == 'manual'

        def generate(payload):
            response = client.post(base + '/generate', json=payload, headers=headers)
            assert response.status_code == 200, response.json
            job = wait_job(client, response.json)
            assert job['status'] == 'done', job
            return client.get(base).json

        payload = dict(mode='audio', settings=settings, timing_confirmed=True, **TEMPO)
        state = generate(payload)
        assert len(state['variants']) == 1
        full_song = state['variants'][state['active']]
        assert full_song['selection']['expression'] == manual_expression
        assert next(note for note in full_song['notes'] if note['id'] == 'anchor')['end'] == 4125

        state = client.get(base).json
        reused_lock = next(note for note in state['analysis']['candidates']
            if note['id'] == 'locked')
        reused_lock.update(start=2000, end=2300)
        app.extensions['projects'].save(state)

        automatic = client.post(base + '/expression', headers=headers,
            json=dict(action='automatic', settings=settings))
        assert automatic.status_code == 200, automatic.json
        regional_expression = automatic.json['expression']
        regional_expression['sections'][0]['voices']['piano']['mode'] = 'omit'
        regional_expression['sections'][1]['voices']['vocals']['mode'] = 'onsets'
        saved_expression = client.post(base + '/expression', headers=headers,
            json={'expression': regional_expression})
        assert saved_expression.status_code == 200, saved_expression.json
        regional_expression = saved_expression.json['expression']
        assert regional_expression['source'] == 'manual'
        state = generate(dict(payload, region=[2500, 5000]))
        regional = state['variants'][state['active']]
        assert regional['selection']['scope'] == 'regional_output'
        assert regional['selection']['structure'] == structure
        assert regional['selection']['expression'] == regional_expression
        assert regional['selection']['proposal']['scope'] == 'full_song_proposal'
        assert regional['selection']['proposal']['expression'] == regional_expression
        assert regional['selection']['proposal']['structure'] == structure
        assert regional['selection']['actual_ln_ratio'] == round(1 / 3, 4)
        assert regional['source_evidence']['anchor']['expressed']['start'] == 3000
        assert regional['source_evidence']['anchor']['expressed']['end'] == 3000
        assert regional['source_evidence']['anchor']['expressed']['phrase'] is False
        anchor = next(note for note in regional['notes'] if note['id'] == 'anchor')
        assert anchor['voice'] == 'vocals'
        assert anchor['end'] == anchor['start'] == 3000
        locked = next(note for note in regional['notes'] if note['id'] == 'locked')
        parent_locked = next(note for note in full_song['notes'] if note['id'] == 'locked')
        assert locked == parent_locked and locked['end'] > locked['start']
        assert regional['source_evidence']['locked'] == full_song['source_evidence']['locked']
        assert regional['source_evidence']['locked']['expressed']['end'] == locked['end']
        decisions = {item['id']: item for item in regional['selection']['decisions']}
        assert decisions['locked']['reason'] == 'locked_existing'
        assert decisions['locked']['chart_start'] == locked['start'] == 1000
        proposal_decisions = {
            item['id']: item for item in regional['selection']['proposal']['decisions']
        }
        assert proposal_decisions['locked']['decision'] == 'omitted'
        assert proposal_decisions['locked']['start'] == 2000
        assert regional['source_evidence']['locked']['detected']['start'] == 1000
        regional_notes = [note for note in regional['notes'] if 2500 <= note['start'] < 5000]
        assert {note['id'] for note in regional_notes} == {'anchor', 'support'}
        assert all(note['end'] == note['start'] for note in regional_notes)
        support_decisions = proposal_decisions
        assert support_decisions['support']['reason'] == 'section_onsets'
        assert support_decisions['support']['decision'] == 'selected'

        regional_before_restart = deepcopy(regional)
        app.extensions['executor'].shutdown()
        app = create_app(root, tmp_path)
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        reopened_state = client.get(base).json
        regional = reopened_state['variants'][reopened_state['active']]
        assert regional == regional_before_restart
        exported_regional = client.get(base + '/export/json')
        assert exported_regional.status_code == 200
        exported_variant = exported_regional.json['variant']
        assert exported_variant == regional_before_restart
        assert exported_variant['selection']['scope'] == 'regional_output'
        assert exported_variant['selection']['proposal']['scope'] == 'full_song_proposal'
        exported_decisions = {
            item['id']: item for item in exported_variant['selection']['decisions']
        }
        assert exported_decisions['locked']['reason'] == 'locked_existing'
        assert exported_decisions['locked']['chart_start'] == 1000
        exported_proposal = {
            item['id']: item for item in exported_variant['selection']['proposal']['decisions']
        }
        assert exported_proposal['locked']['decision'] == 'omitted'
        assert exported_proposal['locked']['start'] == 2000
        assert exported_variant['source_evidence'] == regional['source_evidence']
        assert exported_variant['source_evidence']['locked']['detected']['start'] == 1000

        edited_response = client.post(base + '/edit', headers=headers, json=dict(
            op='update',
            note_id='anchor',
            base_created=regional['created'],
            values={'lane': 2},
        ))
        assert edited_response.status_code == 200, edited_response.json
        edited_job = wait_job(client, edited_response.json)
        assert edited_job['status'] == 'done', edited_job
        edited_state = client.get(base).json
        edited = edited_state['variants'][edited_state['active']]
        edited_anchor = next(note for note in edited['notes'] if note['id'] == 'anchor')
        assert edited_anchor['lane'] == 2
        assert edited_anchor['start'] == edited_anchor['end'] == 3000
        assert edited['source_evidence'] == regional['source_evidence']
        assert edited['selection']['structure'] == structure
        assert edited['selection']['expression'] == regional_expression
        assert edited_state['variants'][0] != edited
    finally:
        app.extensions['executor'].shutdown()

    reopened = create_app(root, tmp_path)
    try:
        client = reopened.test_client()
        headers = {'X-ManiaLab-Token': reopened.config['TOKEN']}
        saved = client.get(base).json
        assert saved['variants'] == edited_state['variants']
        assert saved['active'] == edited_state['active']

        exported_json = client.get(base + '/export/json')
        assert exported_json.status_code == 200
        assert exported_json.json['variant'] == edited
        assert exported_json.json['analysis']['structure'] == structure
        assert exported_json.json['expression'] == regional_expression
        assert exported_json.json['variant']['selection']['structure'] == structure
        assert exported_json.json['variant']['selection']['expression'] == regional_expression
        assert exported_json.json['variant']['source_evidence']['anchor']['expressed'] == \
            regional['source_evidence']['anchor']['expressed']

        chart = client.get(base + '/export/osu')
        bundle = client.get(base + '/export/osz')
        assert chart.status_code == bundle.status_code == 200
        assert 'Bookmarks: 2500' in chart.get_data(as_text=True)
        chart_notes = parse_osu(chart.get_data(as_text=True))['notes']
        assert [(note.start, note.end, note.lane) for note in chart_notes] == [
            (note['start'], note['end'], note['lane']) for note in edited['notes']
        ]
        with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
            assert archive.read('ManiaLab.osu') == chart.data
            assert archive.read('song.wav') == audio
    finally:
        reopened.extensions['executor'].shutdown()
