from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from manialab import arrange
from manialab.domain import Note


def test_candidate_selection_import_does_not_load_arrange():
    code = (
        "import sys; import manialab.candidate_selection; "
        "assert 'manialab.arrange' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_arrange_selection_wrappers_preserve_legacy_replacement_points(monkeypatch):
    calls = []

    def solve(objective, **kwargs):
        calls.append("solver")
        return SimpleNamespace(x=np.ones(len(objective)), status=0)

    def choose(notes, local, settings, segment_preferences=()):
        calls.append("hold-selector")
        return notes, {"patched_hold_selector": True}

    monkeypatch.setattr(arrange, "_solve_binary", solve)
    monkeypatch.setattr(arrange, "choose_holds", choose)
    note = Note(0, 0, source="transient", id="tap")

    selected, report = arrange.select_notes([note])

    assert selected == [note]
    assert calls == ["solver", "hold-selector"]
    assert report["patched_hold_selector"] is True
