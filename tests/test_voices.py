from dataclasses import replace
import numpy as np

from manialab.domain import Note, invariant
from manialab.voices import apply_phrase_evidence, vocal_phrases


def evidence(gap=0):
    times=np.arange(0,4,.01)
    frames=np.zeros((400,88),dtype=np.float32)
    frames[100:200,39]=.8
    frames[200+gap:300,43]=.8
    samples=np.zeros(22050*4,dtype=np.float32)
    samples[22050:44100]=.1
    samples[44100+gap*220:66150]=.1
    notes=[Note(1000,2000,pitch=60,source='pitch',id='a',voice='vocals'),
           Note(2000+gap*10,3000,pitch=64,source='pitch',id='b',voice='vocals')]
    return times,frames,samples,notes


def test_phrase_can_follow_pitch_changes_but_requires_continuous_voice():
    times,frames,samples,notes=evidence()
    phrases=vocal_phrases(frames,times,samples,notes)
    assert len(phrases)==1 and phrases[0].start==1000 and phrases[0].end==3000
    assert phrases[0].phrase and set(phrases[0].support)=={'a','b'}
    assert phrases[0].support_coverage==1
    silent=vocal_phrases(frames,times,np.zeros_like(samples),notes)
    assert silent==[]


def test_phrase_does_not_fill_a_breath_or_an_unpitched_held_sound():
    times,frames,samples,notes=evidence(gap=20)
    phrases=vocal_phrases(frames,times,samples,notes)
    assert len(phrases)==2
    assert phrases[0].end<=2000 and phrases[1].start>=2200
    assert vocal_phrases(np.zeros_like(frames),times,samples,notes)==[]


def test_phrase_amount_never_adds_an_attack_or_mutates_raw_evidence():
    times,frames,samples,notes=evidence()
    original=[n.data() for n in notes]
    phrases=vocal_phrases(frames,times,samples,notes)
    result,report=apply_phrase_evidence(notes,phrases,1)
    assert [(n.id,n.start) for n in result]==[(n.id,n.start) for n in notes]
    assert result[0].end==3000 and result[1]==notes[1]
    assert report['used']==1
    assert [n.data() for n in notes]==original
    result,report=apply_phrase_evidence(notes,phrases,0)
    assert result==notes and report['used']==0


def test_phrase_expression_can_change_at_a_segment_boundary():
    first=Note(1000,1400,source='pitch',voice='vocals',id='a')
    second=replace(first,start=5000,end=5400,id='b')
    phrases=[replace(n,end=n.start+2000,phrase=True,support=(n.id,),support_coverage=1) for n in (first,second)]
    result,report=apply_phrase_evidence([first,second],phrases,0,
        [dict(start_ms=4000,end_ms=7000,preferences={'phrase_ln':1})])
    assert result[0]==first and result[1].end==7000 and report['used']==1


def test_phrase_cap_cannot_shorten_an_original_long_sustain():
    note=Note(1000,12000,source='pitch',voice='vocals',id='long')
    phrase=replace(note,end=7000,phrase=True,support=('long',),support_coverage=1)
    result,report=apply_phrase_evidence([note],[phrase],1)
    assert result==[note] and report['used']==0
