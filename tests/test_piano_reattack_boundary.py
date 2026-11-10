"""Kong decoder restart boundaries must not become observed LN releases."""
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from manialab.arrange import choose_holds, parameters
from manialab.domain import Note, UNOBSERVED_RELEASES, parse_osu
from manialab.expression import expression_evidence, propose_expression
from manialab.expression_activity import section_activity
from manialab.expression_articulation import source_activity
from manialab.piano import OUTPUTS, decode
from tests.test_structure_expression import analysis


def decoded_notes():
    heads={key:np.zeros((1000,88),dtype=np.float32) for key in OUTPUTS}
    heads['velocity_output'][:]=.5
    heads['frame_output'][100:500,39]=.9
    for at in (100,300):
        heads['reg_onset_output'][at-1:at+2,39]=[.2,.8,.2]
    heads['reg_offset_output'][497:500,39]=[.2,.8,.2]
    events=decode(heads)
    # The actual PianoDetector bridge preserves release_reason in Note, while
    # release_observed is an event-level diagnostic rather than a Note field.
    notes=[Note(round(n['onset']*1000),round(n['offset']*1000),pitch=n['pitch'],
                strength=n['onset_score'],source='pitch',voice='piano',id='specialist:'+str(i),
                release_reason=n['release_reason']) for i,n in enumerate(events)]
    return events,notes


@pytest.mark.parametrize('ratio',[.45,.55,1.])
@pytest.mark.parametrize('manual_sustain',[False,True])
def test_decoder_reattack_keeps_attack_in_automatic_and_manual_expression(ratio,manual_sustain):
    events,notes=decoded_notes()
    assert events[0]['release_reason']=='reattack' and not events[0]['release_observed']
    assert events[1]['release_reason']=='release_regression' and events[1]['release_observed']
    assert (notes[0].start,notes[0].end)==(1000,2990)
    original=[n.data() for n in notes]
    a=analysis(notes,10000);settings=parameters({'ln_ratio':ratio})
    plan=propose_expression(a,settings)
    if manual_sustain:
        plan['source']='manual'
        plan['sections'][0]['voices']['piano']['mode']='sustain'
    selected,_,trace=expression_evidence(a,plan,settings)
    assert [(n.start,n.end) for n in selected]==[(1000,1000),(3000,4980)]
    assert trace[notes[0].id]['reason']=='unobserved_release'
    assert trace[notes[1].id]['reason']=='supported_sustain'
    assert a['candidates']==original


def test_reattack_does_not_claim_observed_activity_or_carry_into_later_section():
    _,notes=decoded_notes()
    activity=source_activity(notes[:1],1000,3000)
    assert activity['activity_coverage']==activity['attack_coverage']==.125
    assert activity['observed_interval_coverage']==0 and activity['uncertain_tails']==1
    carried=section_activity(notes[:1],2000,2500)
    assert not carried['present'] and carried['carried_candidate_count']==0


@pytest.mark.parametrize('reliable',['frame_decay','release_regression'])
def test_global_ln_selection_keeps_supported_tails_and_same_pitch_policy(reliable):
    _,notes=decoded_notes()
    p=parameters({'ln_ratio':1.})
    second=replace(notes[1],release_reason=reliable)
    result,_=choose_holds([notes[0],second],[p,p],p)
    assert [(n.start,n.end) for n in result]==[(1000,1000),(3000,4980)]
    basic=replace(notes[0],release_reason='same_pitch_reattack')
    assert 'reattack' in UNOBSERVED_RELEASES and basic.release_reason not in UNOBSERVED_RELEASES
    assert choose_holds([basic,second],[p,p],p)[0]==[basic,second]
    a=analysis([basic,second],10000)
    plan=propose_expression(a,p)
    selected,_,trace=expression_evidence(a,plan,p)
    assert selected==[basic,second]
    assert trace[basic.id]['reason']=='supported_sustain'


@pytest.mark.parametrize('manual_sustain',[False,True])
def test_flask_export_preserves_reattack_onsets_without_unsupported_ln(tmp_path,manual_sustain):
    from manialab.server import create_app
    from manialab.storage import Projects
    from tests.test_server import wait_job
    _,notes=decoded_notes()
    app=create_app(Path(__file__).resolve().parents[1],tmp_path)
    client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    state['analysis']=analysis(notes,10000)
    state['analysis']['revision']=10
    state['analysis']['tempo'].update(stable=True,division=4)
    original=json.loads(json.dumps([n.data() for n in notes]))
    Projects(tmp_path/'projects').save(state)
    url='/api/projects/'+state['id']
    try:
        if manual_sustain:
            response=client.post(url+'/expression',json={'action':'automatic'},headers=headers)
            assert response.status_code==200
            plan=response.json['expression']
            plan['sections'][0]['voices']['piano']['mode']='sustain'
            assert client.post(url+'/expression',json={'expression':plan},headers=headers).status_code==200
        payload=dict(settings={'keys':10,'density':12,'ln_ratio':1.},bpm=120,offset=0,
                     division=4,audio_shift_ms=0,timing_confirmed=True)
        result=wait_job(client,client.post(url+'/generate',json=payload,headers=headers).json)
        assert result['status']=='done'
        parsed=parse_osu(client.get(url+'/export/osu').data.decode())
        assert [(n.start,n.end) for n in parsed['notes']]==[(1000,1000),(3000,5000)]
        current=client.get(url).json
        assert current['analysis']['candidates']==original
        assert current['expression']['source']==('manual' if manual_sustain else 'automatic')
    finally:
        app.extensions['executor'].shutdown()
