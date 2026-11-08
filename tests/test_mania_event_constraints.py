import pytest

from assets.mania_event_constraints import ManiaEventState,context_state


STARTS=dict(t=10,snap=200,column=220,scroll_speed=240,hitsound=270,volume=290,
    circle=400,hold_note=401,hold_note_end=402,hold_note_sustain=403,scroll_speed_change=404)
ENDS=STARTS|dict(t=180,snap=217,column=238,scroll_speed=260,hitsound=280,volume=391)
ENDS.update({k:v+1 for k,v in STARTS.items() if v>=400})


def feed(state,sequence):
    for token in sequence:state.consume(token)


def test_constraints_keep_musical_times_free_and_require_complete_event():
    state=ManiaEventState(STARTS,ENDS,10,[2,6])
    assert 10 in state.allowed() and 179 in state.allowed() and 2 in state.allowed()
    feed(state,[20,201,222,270,310,401])  # 100ms LN head, third column.
    assert state.held=={2:100}
    feed(state,[30,201,222])
    assert state.allowed()==[402]
    with pytest.raises(ValueError):state.consume(401)  # Cannot overwrite its head.
    state.consume(402)
    assert not state.held and state.stage=='start'
    feed(state,[30,201,222,270,310,400])  # A new attack at a positive-length release is legal.
    feed(state,[30,201])
    assert 222 not in state.allowed()  # No duplicate attack at the same time.
    assert 229 in state.allowed() and 230 not in state.allowed()


def test_orphan_tail_eos_in_mid_event_and_short_hold_are_forbidden():
    state=ManiaEventState(STARTS,ENDS,6,[2,6])
    feed(state,[20,201,220])
    assert 402 not in state.allowed() and 2 not in state.allowed()
    feed(state,[270,310,401,20,201])
    assert 220 not in state.allowed()


def test_generated_history_supports_incoming_sustains_without_reading_sketch():
    held,attacks=context_state([('t',10),('column',1),('hold_note_sustain',0),
        ('t',20),('snap',1),('column',2),('circle',0)],100)
    assert held=={1:float('-inf')} and attacks=={2:-80}
    state=ManiaEventState(STARTS,ENDS,10,[2],held,attacks)
    feed(state,[30,221,403])
    assert 1 in state.held
    feed(state,[40,201,221,402])
    assert not state.held


def test_no_snapping_or_hitsound_configuration_and_eos_does_not_fake_tail():
    state=ManiaEventState(STARTS,ENDS,10,[2],snapping=False,hitsounds=False,sv=False)
    feed(state,[20,220,401,30,220,402,40,221,401,2])
    assert state.held=={1:300}
    assert state.stage=='start'


@pytest.mark.parametrize('hold',[False,True])
def test_full_chord_cannot_enter_dead_end_timestamp(hold):
    state=ManiaEventState(STARTS,ENDS,6,[2],sv=False)
    for column in range(6):
        feed(state,[20,201,220+column,270,310,401 if hold else 400])
    assert 20 not in state.allowed()
    assert 21 in state.allowed() and 2 in state.allowed()
    feed(state,[21,201,220])
    assert state.allowed()==([402] if hold else list(state.span('hitsound')))


def test_history_midpoint_is_reencoded_to_the_current_window_token():
    held,attacks=context_state([('t',1105),('column',2),('circle',0)],1000)
    state=ManiaEventState(STARTS,ENDS,6,[2],held,attacks,sv=False)
    feed(state,[20,201])
    assert 222 not in state.allowed() and 223 in state.allowed()
