import numpy as np

from lewm.sparse_corner_completion_runtime_development import StrongCornerFramewiseRegistration
from lewm.visual_support_recovery_development import LocalSupportedView


class Passthrough:
    def observe(self, policy, primary, auxiliary, raw, *, now_ns):
        return raw


def raw(frame, strong, tracking, angle):
    c, s = np.cos(angle), np.sin(angle)
    pose = dict(frame=frame, measured_ns=frame*100_000_000,
        position_initial_body_m=[0., 0., 0.],
        rotation_initial_body_from_current_body=[[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return dict(current_pose=pose,
        last_accepted_feature_witness=dict(selected_features=tracking[0], original_selected_count=strong[0]),
        auxiliary_feature_witness=dict(selected_features=tracking[1], original_selected_count=strong[1]))


def test_faint_corners_do_not_suppress_existing_weak_view_recovery():
    registration = StrongCornerFramewiseRegistration(Passthrough())
    registration.views = LocalSupportedView(maximum_view_age_ns=None)
    states = (([110, 30], [150, 150], 0.), ([25, 40], [150, 150], .5),
        ([30, 35], [150, 150], 0.), ([50, 60], [150, 150], 0.))
    for frame, (strong, tracking, angle) in enumerate(states):
        evidence = raw(frame, strong, tracking, angle)
        result = registration.observe(None, None, None, evidence, now_ns=frame*100_000_000)
        receipt = result['visual_support']
        assert receipt['selected_features'] == strong
        assert receipt['tracking_selected_features'] == tracking
        assert evidence['last_accepted_feature_witness']['selected_features'] == tracking[0]
        assert (receipt['recovery_state_at_observation'] is not None) == (frame in (1, 2))
