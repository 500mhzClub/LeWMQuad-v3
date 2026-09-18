"""Settled round-trip controller with explicitly admitted paired-camera motion."""
from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion, current_dual_camera_pose
from lewm.settled_boundary_round_trip_development import SettledBoundaryRoundTripController


class DualCameraSettledController(SettledBoundaryRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = DualCameraVisualMotion(identity=(0, 0, 0))

    def observe(self, policy, depth, fast, *, now_ns, auxiliary_depth=None, auxiliary_rgb=None):
        evidence = None; raw = None; self.memory_receipt = None
        try:
            if self.terminal is None:
                raw = self.motion.observe(policy, depth, fast, now_ns=now_ns,
                    auxiliary_depth=auxiliary_depth, auxiliary_rgb=auxiliary_rgb)
                current_dual_camera_pose(raw, policy, auxiliary_rgb, auxiliary_depth,
                    identity=(0, 0, 0), now_ns=now_ns)
                evidence = self.registration.observe(policy, depth, auxiliary_depth, raw, now_ns=now_ns)
                self.memory_receipt = self.mapper.observe(policy, depth, evidence,
                    auxiliary_depth=auxiliary_depth, now_ns=now_ns)
            result = self.advance(policy, evidence, now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.terminal = 'SENSOR_OR_MODEL_FAILURE'; self.failure = str(error)
            self.previous_command = [0., 0., 0.]
            result = self._result([0., 0., 0.], None, None)
        return result | dict(evidence=evidence, original_visual_evidence=raw)

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='dual_camera_settled_round_trip_controller_v1',
            additional_auxiliary_rgb_for_motion=True)
