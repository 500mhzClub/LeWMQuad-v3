"""Latch measured view recovery at camera cadence, before planning subsampling."""
import numpy as np

from lewm.prefix_aware_terminal_approach_development import PrefixAwareTerminalRuntime
from lewm.visual_support_recovery_development import LocalSupportedView, SupportRegistration


class FramewiseSupportRegistration(SupportRegistration):
    def __init__(self, original):
        super().__init__(original)
        self.views = LocalSupportedView()

    def observe(self, policy, primary, auxiliary, raw, *, now_ns):
        result = super().observe(policy, primary, auxiliary, raw, now_ns=now_ns)
        receipt = result['visual_support']
        pose = result.get('current_pose')
        if pose is None or receipt is None:
            return result
        if pose['measured_ns'] != now_ns or receipt['measured_ns'] != now_ns or pose['frame'] != receipt['frame']:
            raise ValueError('view recovery requires the same current registered pose')
        # Local visual references are geometric observations, independent of
        # which mission leg is active. Keep their original coordinate frame.
        active = self.views.advance(receipt['selected_features'],
            np.asarray(pose['position_initial_body_m']),
            np.asarray(pose['rotation_initial_body_from_current_body']), now_ns, 0)
        frozen = None if active is None else {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in active.items()}
        return result | dict(visual_support=receipt | dict(
            camera_cadence_recovery=True, recovery_state_at_observation=frozen,
            local_view_reference_independent_of_mission_leg=True))


class FramewiseVisualSupportRuntime(PrefixAwareTerminalRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registration = FramewiseSupportRegistration(self.registration.original)

    def _recovery_state(self, receipt, position, rotation, measured_ns):
        if receipt['measured_ns'] != measured_ns or receipt.get('camera_cadence_recovery') is not True:
            raise ValueError('plan must consume its own camera-time recovery state')
        active = receipt['recovery_state_at_observation']
        if active is None:
            return None
        if not active['measured_ns'] <= active['trigger_ns'] <= measured_ns:
            raise ValueError('future recovery references forbidden')
        return active | dict(rotation=np.asarray(active['rotation']), position=np.asarray(active['position']))
