"""New sustained collection; native guards supervise but never inform a policy."""
import numpy as np

from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices
from lewm.sustained_motion_collection_development import validate_command
from scripts.rgbd_session_development import RGBDSession
from scripts.rgbd_shadow_motion_session_development import AppearanceRGBDSession
from scripts.sustained_motion_physical_init_development import SustainedPhysicalInit
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop


class SustainedMotionSession(RGBDSession, SustainedPhysicalInit):
    _build_contact_topology = AppearanceRGBDSession._build_contact_topology
    install_contact_identity = AppearanceRGBDSession.install_contact_identity
    capture_fixed_rgb = AppearanceRGBDSession.capture_fixed_rgb

    def __init__(self, *args, **kwargs):
        self.guard = None; self.guard_rows = []
        super().__init__(*args, **kwargs)

    def command_tick(self, requested):
        return super().command_tick(validate_command(requested))

    def _sample(self, requested, applied, timestamp_s):
        row = super()._sample(requested, applied, timestamp_s)
        if self.guard is not None:
            packet = {k: np.asarray(v)[0] for k, v in self.packets[-1].items()}
            nonfeet = nonfoot_ground_contact_indices(packet, **self.guard)
            speed = float(np.linalg.norm(row['base_twist_world'][:3]))
            in_domain = bool((np.abs(row['base_pose_world'][:2]) < 8.).all())
            self.guard_rows.append(dict(sample_index=len(self.samples)-1,
                measured_ns=int(round(timestamp_s*1e9)), nonfoot_ground_contact_indices=nonfeet,
                base_speed_m_s=speed, in_declared_capture_domain=in_domain,
                evaluator_supervision_not_policy_input=True))
            if nonfeet or speed > .3 or not in_domain:
                raise PhysicalStop('SUSTAINED_COLLECTION_NATIVE_CONTACT_SPEED_OR_DOMAIN_GUARD')
        return row
