import numpy as np

from lewm.commanded_planar_motion_development import forecast
from lewm.yaw_source_ablation_development import choose_yaw, YawSourceMixin


def test_command_yaw_is_independent_of_neural_yaw_and_preserves_other_channels():
    rng = np.random.default_rng(5)
    original = rng.normal(size=(6,8,5)).astype(np.float32)
    alternative = original.copy(); alternative[:,:,2:4] = rng.normal(size=(6,8,2))*20
    snapshot = original.copy()
    for pulse in (False, True):
        command = forecast([[.16,0,.45]]*3, pulse=pulse)
        a = choose_yaw(original,command,'command'); b = choose_yaw(alternative,command,'command')
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,:,[0,1,4]],original[:,:,[0,1,4]])
        np.testing.assert_allclose(a[:,:,2:4],command[:,:,2:4],atol=1e-7,rtol=0)
        np.testing.assert_array_equal(choose_yaw(original,command,'learned'),original)
    np.testing.assert_array_equal(original,snapshot)


class Upstream:
    def __init__(self, **kwargs): self.planning_translation_pulse = True

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        return prediction, dict(forecast_xy_source='pose_command',contact_score_mode='disabled',
            learned_yaw_retained=True)


class Runtime(YawSourceMixin, Upstream): pass


def test_runtime_returns_and_records_the_applied_pulse_yaw():
    p = np.zeros((6,8,5)); p[:,:,4] = -1000
    result, receipt = Runtime(forecast_yaw_source='command')._correct_prediction(
        p,None,None,[[0,0,0]]*3)
    # Left arc commits one 100-ms pulse; a pure left turn still commits 400 ms.
    np.testing.assert_allclose(result[2,6,2:4],[np.sin(.045),np.cos(.045)],atol=1e-12)
    np.testing.assert_allclose(result[4,6,2:4],[np.sin(.18),np.cos(.18)],atol=1e-12)
    np.testing.assert_array_equal(result,receipt['applied_prediction_after_yaw_ablation'])
    assert not receipt['learned_yaw_retained'] and not receipt['neural_outcomes_used_for_scoring']
    assert receipt['pose_command_xy_remains_a_fitted_model']
