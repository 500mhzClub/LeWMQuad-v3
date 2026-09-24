"""RGB-D acquisition with external native stops; no native controller input."""
import numpy as np
from PIL import Image

from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices
from lewm.longer_motion_collection_development import validate_command
from scripts.goal_hold_servo_physical_init_development import GoalHoldServoPhysicalInit
from scripts.rgbd_session_development import RGBDSession
from scripts.rgbd_shadow_motion_session_development import AppearanceRGBDSession
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop


class GoalHoldServoSession(RGBDSession,GoalHoldServoPhysicalInit):
    _build_contact_topology=AppearanceRGBDSession._build_contact_topology
    install_contact_identity=AppearanceRGBDSession.install_contact_identity
    capture_fixed_rgb=AppearanceRGBDSession.capture_fixed_rgb

    def __init__(self,*args,**kwargs):
        self.guard=None; self.guard_rows=[]
        super().__init__(*args,**kwargs)

    def command_tick(self,requested): return super().command_tick(validate_command(requested))

    def _sample(self,requested,applied,timestamp_s):
        row=super()._sample(requested,applied,timestamp_s)
        if self.guard is not None:
            packet={k:np.asarray(v)[0] for k,v in self.packets[-1].items()}
            indices=nonfoot_ground_contact_indices(packet,**self.guard)
            speed=float(np.linalg.norm(row['base_twist_world'][:3])); inside=bool((np.abs(row['base_pose_world'][:2])<8).all())
            self.guard_rows.append(dict(sample_index=len(self.samples)-1,nonfoot_ground_contact_indices=indices,
                base_speed_m_s=speed,in_domain=inside,evaluator_only=True))
            if indices or speed>.3 or not inside: raise PhysicalStop('SERVO_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP')
        return row

    def sensor_packets(self):
        index=self.capture_current(); now=int(self.model_manifest[index]['decision_ns'])
        with Image.open(self.output/f'rgb_{index:04d}.png') as image: pixels=np.array(image)
        return self.observations.packet(pixels,now),self.latest_depth,self.fast_buffer.packet(now_ns=now),now
