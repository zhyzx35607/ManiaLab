from pathlib import Path
import subprocess
import sys

from manialab.domain import Note, upstream_events as compatible_upstream_events
from manialab.mapper_event_adapter import upstream_events


def run_isolated(code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_mapper_event_stream_keeps_timing_order_and_ln_end_precedence():
    notes = [
        Note(1000, 1000, lane=2),
        Note(1000, 2000, lane=1),
        Note(1500, 2000, lane=4),
        Note(2000, 2600, lane=0),
        Note(2000, 2000, lane=3),
    ]

    events = upstream_events(notes)

    assert [str(event) for event in events] == [
        "hold_note0", "t1000", "column1",
        "circle0", "t1000", "column2",
        "hold_note0", "t1500", "column4",
        "hold_note_end0", "t2000", "column1",
        "hold_note_end0", "t2000", "column4",
        "hold_note0", "t2000", "column0",
        "circle0", "t2000", "column3",
        "hold_note_end0", "t2600", "column0",
    ]
    assert len(events) == 24


def test_domain_upstream_events_remains_a_compatible_entry():
    notes = [Note(100, 500, lane=0), Note(200, 200, lane=1)]

    assert compatible_upstream_events(notes) == upstream_events(notes)


def test_domain_event_types_remain_lazy_compatibility_exports():
    code = (
        "import sys; import manialab.domain as domain; "
        "assert 'vendor.mapperatorinator.event' not in sys.modules; "
        "assert 'manialab.mapper_event_adapter' not in sys.modules; "
        "from manialab.domain import Event, EventType; "
        "assert 'vendor.mapperatorinator.event' in sys.modules; "
        "assert 'manialab.mapper_event_adapter' in sys.modules; "
        "from vendor.mapperatorinator.event import Event as ExpectedEvent, EventType as ExpectedEventType; "
        "assert Event is ExpectedEvent and EventType is ExpectedEventType"
    )
    run_isolated(code)


def test_importing_domain_does_not_load_vendor_event_protocol():
    code = (
        "import sys; import manialab.domain; "
        "assert 'vendor.mapperatorinator.event' not in sys.modules; "
        "assert 'manialab.mapper_event_adapter' not in sys.modules"
    )
    run_isolated(code)


def test_importing_mapper_event_adapter_does_not_load_domain():
    code = (
        "import sys; import manialab.mapper_event_adapter; "
        "assert 'manialab.domain' not in sys.modules"
    )
    run_isolated(code)
