from pathlib import Path
import subprocess
import sys

from manialab import expression, expression_selection


def test_expression_module_reexports_selection_helpers():
    for name in (
        'resolve_phrase_decisions',
        'chord_layer_options',
        '_attack_value',
        '_deduplicate_pitches',
        'drum_group_anchor',
        '_group_fits',
    ):
        assert getattr(expression, name) is getattr(expression_selection, name)
    assert expression.select_expression.__module__ == 'manialab.expression'


def test_expression_selection_module_does_not_import_expression_or_audio():
    code = (
        "import sys; import manialab.expression_selection; "
        "assert 'manialab.expression' not in sys.modules; "
        "assert 'manialab.audio' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
