"""Goals in one uninterrupted visual frame; no place or clearance assertion."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from lewm.causal_sensor_state import SensorContractError, _identity, _ns
from lewm.joint_rgbd_rigid_pose_development import proper


def current_pose(evidence, *, identity, now_ns):
    now = _ns(now_ns, 'anchored goal clock')
    if (evidence['schema'] != 'visual_led_motion_evidence_development.v1'
            or _identity(evidence['identity']) != _identity(identity)
            or evidence['decision_ns'] != now or evidence['status'] != 'CURRENT_VISUAL_POSE'
            or evidence['terminal_failure'] is not None):
        raise SensorContractError('same-episode current visual evidence required')
    pose = evidence['current_pose']
    if (pose is None or pose['mode'] != 'gyro' or pose['measured_ns'] != now
            or pose['available_ns'] > now or type(pose['frame']) is not int or pose['frame'] < 0):
        raise SensorContractError('current gyro-conditioned visual frame required')
    p = np.asarray(pose['position_initial_body_m'], float)
    R = proper(pose['rotation_initial_body_from_current_body'])
    if p.shape != (3,) or not np.isfinite(p).all():
        raise SensorContractError('finite visual position required')
    for key in ('rgb_sha256', 'depth_sha256'):
        value = pose[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise SensorContractError('explicit RGB/depth binding required')
    return p.copy(), R.copy(), pose


@dataclass(frozen=True)
class AnchoredGoal:
    identity: tuple
    anchor_ns: int
    anchor_frame: int
    rgb_sha256: str
    depth_sha256: str
    anchor_position: tuple
    anchor_rotation: tuple
    requested_body_xy: tuple
    requested_yaw_delta_rad: float
    target_xy: tuple
    target_yaw_rad: float

    @classmethod
    def from_observation(cls, evidence, displacement_body_xy, yaw_delta_rad, *, identity, now_ns):
        p, R, pose = current_pose(evidence, identity=identity, now_ns=now_ns)
        d = np.asarray(displacement_body_xy, float)
        if (d.shape != (2,) or not np.isfinite(d).all() or np.linalg.norm(d) > .4+1e-12
                or isinstance(yaw_delta_rad, bool) or not isinstance(yaw_delta_rad, (float, int))
                or not math.isfinite(yaw_delta_rad) or abs(yaw_delta_rad) > math.pi):
            raise SensorContractError('bounded .4m local displacement and wrapped yaw request required')
        target = p + R @ np.r_[d, 0.]
        heading = R @ [math.cos(yaw_delta_rad), math.sin(yaw_delta_rad), 0.]
        if np.linalg.norm(heading[:2]) < .2:
            raise SensorContractError('nonvertical final heading required')
        return cls(_identity(identity), now_ns, pose['frame'], pose['rgb_sha256'], pose['depth_sha256'],
                   tuple(p), tuple(tuple(row) for row in R), tuple(d), float(yaw_delta_rad),
                   tuple(target[:2]), math.atan2(heading[1], heading[0]))

    def snapshot(self):
        return asdict(self) | dict(place_identity=None, clearance_qualified=False)
