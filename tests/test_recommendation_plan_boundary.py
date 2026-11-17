from pathlib import Path
import subprocess
import sys

from manialab import recommendation_plan, recommendations


def test_recommendations_reexports_plan_helpers_by_identity():
    for name in (
        'EVIDENCE_THRESHOLD',
        'adjusted_segments',
        'evidence_signature',
        'plan_evidence',
        'validate_plan',
    ):
        assert getattr(recommendations, name) is getattr(recommendation_plan, name)


def test_recommendation_plan_import_does_not_load_recommendations():
    code = (
        "import sys; import manialab.recommendation_plan; "
        "assert 'manialab.recommendations' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
