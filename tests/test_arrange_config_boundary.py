from pathlib import Path
import subprocess
import sys

from manialab import arrange, arrange_config


def test_arrange_reexports_config_objects_by_identity():
    for name in ("DEFAULTS", "BOUNDS", "INTEGER", "parameters"):
        assert getattr(arrange, name) is getattr(arrange_config, name)


def test_parameters_returns_an_independent_defaults_copy():
    defaults = arrange_config.DEFAULTS.copy()
    configured = arrange_config.parameters()

    configured["keys"] = 6

    assert arrange_config.DEFAULTS == defaults
    assert configured is not arrange_config.DEFAULTS


def test_arrange_config_import_does_not_load_solver_dependencies():
    code = (
        "import sys; import manialab.arrange_config; "
        "assert 'manialab.arrange' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert not any(name == 'scipy' or name.startswith('scipy.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_config_consumers_do_not_load_arrange_or_scipy():
    code = (
        "import sys; "
        "import manialab.planner, manialab.recommendations, manialab.project_routes; "
        "assert 'manialab.arrange' not in sys.modules; "
        "assert not any(name == 'scipy' or name.startswith('scipy.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
