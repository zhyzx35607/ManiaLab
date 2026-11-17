from manialab import timing, timing_fitting


def test_timing_facade_preserves_fitting_helpers():
    for name in (
        "normalize_bpm",
        "_phase",
        "_fit",
        "_score",
        "refit_offset",
        "_precision_candidate",
        "beat_meter_support",
    ):
        assert getattr(timing, name) is getattr(timing_fitting, name)
