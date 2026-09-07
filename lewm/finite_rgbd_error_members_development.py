"""Finite deterministic error members, each with its own terminal RGB-D owner.

No covariance, continuous error-set guarantee, model reset or action permission.
Error hypotheses are fixed before replay; perturbations preserve timestamp and
modality provenance. Failures are outcomes, not reasons to delete a member.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_shadow_motion_development import ShadowObserver


@dataclass(frozen=True)
class ErrorMember:
    name: str
    depth_gain: float = 0.
    depth_offset_m: float = 0.
    gyro_z_bias_rad_s: float = 0.
    post_anchor_force_y_bias_m_s2: float = 0.
    initial_velocity_y_error_m_s: float = 0.
    independent_depth_noise_m: float = 0.
    blank_rgb: bool = False

    def __post_init__(self):
        values = [self.depth_gain, self.depth_offset_m, self.gyro_z_bias_rad_s,
                  self.post_anchor_force_y_bias_m_s2, self.initial_velocity_y_error_m_s,
                  self.independent_depth_noise_m]
        if (not isinstance(self.name, str) or not self.name or type(self.blank_rgb) is not bool
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in values)):
            raise SensorContractError('named finite fixed error member required')


def fixed_members():
    rows = [ErrorMember('nominal')]
    for name, key, size in (
        ('depth_gain', 'depth_gain', .001), ('depth_offset', 'depth_offset_m', .001),
        ('gyro_z', 'gyro_z_bias_rad_s', .001), ('force_y', 'post_anchor_force_y_bias_m_s2', .01),
        ('prior_y', 'initial_velocity_y_error_m_s', .01), ('depth_noise', 'independent_depth_noise_m', .0001)):
        for sign in (1, -1): rows.append(ErrorMember(f'{name}_{sign:+d}', **{key: sign*size}))
    rows.append(ErrorMember('blank_rgb', blank_rgb=True))
    for sign in (1, -1):
        rows.append(ErrorMember(f'combined_{sign:+d}', depth_gain=sign*.001, depth_offset_m=sign*.001,
            gyro_z_bias_rad_s=sign*.001, post_anchor_force_y_bias_m_s2=sign*.01,
            initial_velocity_y_error_m_s=sign*.01, independent_depth_noise_m=sign*.0001))
    return tuple(rows)


def perturb_packets(member, policy, depth, fast, *, anchor_ns):
    if not isinstance(member, ErrorMember): raise SensorContractError('explicit error member required')
    if member == ErrorMember('nominal'): return policy, depth, fast
    p, d, f = deepcopy(policy), deepcopy(depth), deepcopy(fast)
    now = int(policy['sensor_state']['decision_ns'])
    if depth['depth_m'].dtype != np.float32: raise SensorContractError('actual float32 depth required')
    if member.depth_gain or member.depth_offset_m or member.independent_depth_noise_m:
        z = depth['depth_m'].astype(float)
        delta = member.depth_gain*z + member.depth_offset_m*depth['valid']
        if member.independent_depth_noise_m:
            rng = np.random.default_rng(np.random.SeedSequence([2026090621, now & 0xffffffff, now >> 32]))
            delta += member.independent_depth_noise_m*rng.uniform(-1., 1., z.shape)*depth['valid']
        d['depth_m'] = (z+delta).astype(np.float32)
    if member.gyro_z_bias_rad_s:
        slow = p['sensor_state']['sensed']['gyro']
        slow['values'][:, 2] += member.gyro_z_bias_rad_s*slow['valid'][:, 2]
        f['values'][:, 2] += member.gyro_z_bias_rad_s*f['valid'][:, 2]
    if member.post_anchor_force_y_bias_m_s2:
        force = p['sensor_state']['sensed']['specific_force']
        after = (force['measured_ns'] > anchor_ns) & force['valid'][:, 1]
        force['values'][after, 1] += member.post_anchor_force_y_bias_m_s2
    if member.blank_rgb:
        p['image']['rgb'] = np.full_like(p['image']['rgb'], 128)
        d['rgb_sha256'] = hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    return p, d, f


class FiniteRGBDMember:
    def __init__(self, member, prior):
        if not isinstance(member, ErrorMember): raise SensorContractError('explicit error member required')
        self.member = member; self.anchor_ns = prior.anchor_ns
        mean = list(prior.mean_initial_body_m_s); mean[1] += member.initial_velocity_y_error_m_s
        self.observer = ShadowObserver(replace(prior, mean_initial_body_m_s=tuple(mean)))
        self.last_packet = None

    def observe(self, policy, depth, fast):
        now = policy['sensor_state']['decision_ns']
        if self.observer.failure is not None:
            self.last_packet = None
            # This public predecessor path returns its recorded failure without
            # invoking the terminal estimator, reading new pixels or integrating.
            return self.observer.observe(policy, depth, fast, now_ns=now)
        p, d, f = perturb_packets(self.member, policy, depth, fast, anchor_ns=self.anchor_ns)
        result = self.observer.observe(p, d, f, now_ns=now)
        self.last_packet = (p, d, f) if result['state'] is not None else None
        return result
