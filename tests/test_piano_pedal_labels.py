from scripts.piano_pedal_labels import sustain_labels


def test_sustain_targets_respect_retrigger_channel_and_key_release():
    def note(start, end, pitch=60, channel=0):
        return dict(onset=start, offset=end, pitch=pitch, channel=channel)
    notes=[note(1,2),note(3,4),note(1.5,2.5,64),note(1,2,channel=1),note(8,9)]
    controls=[dict(time=t,channel=0,controller=64,value=v) for t,v in [(0,127),(2,90),(5,0)]]
    result,audit=sustain_labels(notes,controls,10)
    assert sorted((n['onset'],n['pitch'],n['channel'],n['offset']) for n in result)==[
        (1,60,0,3),(1,60,1,2),(1.5,64,0,5),(3,60,0,5),(8,60,0,9)]
    assert audit['extended']==3 and audit['pedal_intervals']==1
    assert notes[0]['offset']==2  # Original MIDI evidence is immutable.


def test_sustain_targets_report_unclosed_pedal_and_do_not_extend_boundary_release():
    notes=[dict(onset=0,offset=1,pitch=60),dict(onset=2,offset=3,pitch=64)]
    controls=[dict(time=1,channel=0,controller=64,value=64)]
    result,audit=sustain_labels(notes,controls,8)
    assert [n['offset'] for n in result]==[1,8]
    assert audit['unclosed_pedals']==1
