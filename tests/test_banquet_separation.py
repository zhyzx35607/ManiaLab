import numpy as np
import pytest
from scripts.check_banquet_separation import wave_metrics


def test_separation_scores_expose_gain_distortion_hidden_by_si_sdr():
    ref=np.sin(np.arange(1000)*.1)[None]
    exact=wave_metrics(ref,ref)
    scaled=wave_metrics(ref,.25*ref)
    assert exact['snr_db']>100 and scaled['snr_db']<3
    assert scaled['si_sdr_db']>100 and scaled['gain']==pytest.approx(.25)


def test_absent_target_has_residual_metrics_without_infinite_or_fabricated_sdr():
    score=wave_metrics(np.zeros((2,200)),np.full((2,200),.1))
    assert score['reason']=='silent_reference' and score['si_sdr_db'] is None and score['snr_db'] is None
    assert score['output_rms']==pytest.approx(.1)
    silent=wave_metrics(np.sin(np.arange(200)*.1)[None],np.zeros((1,200)))
    assert silent['si_sdr_db'] is None and silent['snr_db']==pytest.approx(0.)


def test_separation_scoring_refuses_misaligned_or_nonfinite_waveforms():
    with pytest.raises(ValueError,match='differ'):
        wave_metrics(np.zeros((2,200)),np.zeros((2,199)))
    with pytest.raises(ValueError,match='differ'):
        wave_metrics(np.zeros((2,200)),np.full((2,200),np.nan))


def test_joint_level_diagnostic_uses_one_mixture_query_gain_and_handles_silence():
    from scripts.check_banquet_input_level import joint_gain
    assert joint_gain(np.array([.1,-.2]),np.array([.5]))==pytest.approx(1.9)
    assert joint_gain(np.zeros(3),np.zeros(4))==1.
    with pytest.raises(ValueError,match='Nonfinite'):
        joint_gain(np.array([float('nan')]),np.ones(3))
