import numpy as np
import pytest

from lewm.framewise_visual_support_recovery_development import FramewiseSupportRegistration, FramewiseVisualSupportRuntime


class Registration:
    def observe(self, policy, primary, auxiliary, raw, *, now_ns):
        return {'current_pose': raw['current_pose']}


def packet(frame, features, angle):
    c, s = np.cos(angle), np.sin(angle)
    return dict(current_pose=dict(frame=frame, measured_ns=frame*100_000_000,
        position_initial_body_m=[0., 0., 0.],
        rotation_initial_body_from_current_body=[[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]),
        last_accepted_feature_witness={'selected_features': features},
        auxiliary_feature_witness={'selected_features': features})


def test_warning_between_plans_survives_transient_feature_recovery():
    observer = FramewiseSupportRegistration(Registration())
    for frame, count, angle in [(0, 100, 0.), (1, 30, .5), (2, 60, .6)]:
        raw = packet(frame, count, angle)
        result = observer.observe(None, None, None, raw, now_ns=frame*100_000_000)
    saved = result['visual_support']
    assert saved['recovery_state_at_observation']['trigger_ns'] == 100_000_000
    raw = packet(3, 100, 0.)
    observer.observe(None, None, None, raw, now_ns=300_000_000)
    # A later completion cannot change the evidence attached to an older plan.
    assert saved['recovery_state_at_observation']['trigger_ns'] == 100_000_000
    runtime = object.__new__(FramewiseVisualSupportRuntime)
    assert runtime._recovery_state(saved, None, None, 200_000_000) is not None
    with pytest.raises(ValueError):
        runtime._recovery_state(saved, None, None, 100_000_000)
