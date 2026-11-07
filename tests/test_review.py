from manialab.domain import Note
from manialab.review import review_chart


def test_review_locates_releases_and_hand_coordination_without_mutation():
    notes=[Note(0,1000,0,id='hold'),Note(1020,1020,0,id='after'),Note(1090,1090,0,id='jack'),
           Note(2000,6000,1),Note(2000,6000,2),Note(2000,6000,3),Note(2000,6000,4),Note(3000,3000,0)]
    before=[n.data() for n in notes]
    report=review_chart(notes,10,500)
    assert report['counts']['short_release']==1
    assert report['counts']['fast_repeat']==1
    release=next(e for e in report['locations'] if e['kind']=='short_release')
    assert release['start_ms']==1020 and release['gap_ms']==20 and release['lane']==0
    hand=next(e for e in report['locations'] if e['kind']=='held_hand_load')
    assert hand['hand']==0 and hand['start_ms']==3000
    assert [n.data() for n in notes]==before


def test_long_rest_does_not_create_a_fast_repeat_warning():
    report=review_chart([Note(0,500,0),Note(1000,1000,0)],6)
    assert report['counts']=={} and report['locations']==[]


def test_review_uses_local_tempo_for_repeated_attacks():
    notes = [Note(0,0,0),Note(100,100,0),Note(2000,2000,0),Note(2100,2100,0)]
    points = ['0,500,4,2,0,70,1,0','2000,250,4,2,0,70,1,0']
    assert review_chart(notes,6)['counts']['fast_repeat'] == 2
    result = review_chart(notes,6,timing_points=points)
    assert result['counts']['fast_repeat'] == 1
    assert result['locations'][0]['start_ms'] == 100
