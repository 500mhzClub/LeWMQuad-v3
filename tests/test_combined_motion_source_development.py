import numpy as np
from lewm.combined_motion_source_development import CombinedMotionSourceMixin
from lewm.contact_score_ablation_development import contact_score_prediction
from lewm.pose_command_xy_control_development import choose_xy
from lewm.commanded_planar_motion_development import forecast


class Upstream:
    def __init__(self, *, forecast_xy_source, contact_score_mode):
        self.xy_source = forecast_xy_source; self.contact = contact_score_mode
        self.planning_translation_pulse = True

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        xy = forecast(prefix, pulse=True)[:,:,:2]
        selected = contact_score_prediction(choose_xy(prediction, xy, self.xy_source), self.contact)
        return selected, dict(forecast_xy_source=self.xy_source, contact_score_mode=self.contact)


class Runtime(CombinedMotionSourceMixin, Upstream): pass


def test_simple_control_has_no_neural_motion_channel_leakage():
    rng = np.random.default_rng(45); a = rng.normal(size=(6, 8, 5)); b = rng.normal(size=a.shape)
    original = a.copy(); prefix = [[.16, 0, .45]]*3
    runtime = Runtime(motion_prediction_source='pose_command')
    x, receipt = runtime._correct_prediction(a, None, None, prefix)
    y, _ = runtime._correct_prediction(b, None, None, prefix)
    np.testing.assert_array_equal(x, y)
    np.testing.assert_allclose(x[:,:,:4], forecast(prefix, pulse=True))
    np.testing.assert_array_equal(a, original)
    np.testing.assert_array_equal(x, receipt['applied_prediction_after_yaw_ablation'])
    assert not receipt['neural_outcomes_used_for_scoring'] and np.all(x[:,:,4] == -1000)


def test_learned_motion_channels_remain_applied_with_contact_disabled():
    p = np.random.default_rng(46).normal(size=(6, 8, 5))
    x, receipt = Runtime(motion_prediction_source='learned')._correct_prediction(
        p, None, None, [[0, 0, 0]]*3)
    np.testing.assert_array_equal(x[:,:,:4], p[:,:,:4])
    assert np.all(x[:,:,4] == -1000) and receipt['neural_outcomes_used_for_scoring']
