"""Persistent ray memory consumes each new fused state exactly once."""
from copy import deepcopy

from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_inertial_fusion_development import RGBDInertialState
from lewm.uncertain_ray_memory_development import FusedRayEvidenceMemory


class _SingleUseFusion:
    def __init__(self):
        self.pending=None; self.gravity=None

    def stage(self, policy, relative, fusion, gravity):
        if self.pending is not None:raise SensorContractError('pending fusion cannot be replaced')
        self.pending=(policy,relative,deepcopy(fusion)); self.gravity=gravity.copy()

    def observe(self, policy, relative):
        pending=self.pending; self.pending=None
        if pending is None or policy is not pending[0] or relative is not pending[1]:
            raise SensorContractError('same synchronous single-use fusion observation required')
        return pending[2]


class RGBDInertialRayMemory:
    """No second integration, pose reset, raw-rank mutation, or stale query."""
    def __init__(self, *, prior, hypotheses):
        self.state=RGBDInertialState(prior=prior,hypotheses=hypotheses)
        self.rays=FusedRayEvidenceMemory(); self._reader=_SingleUseFusion()
        self.rays.integrator=self._reader; self.failed=False; self.last_ns=None

    def observe(self, policy, depth, fast, *, now_ns):
        if self.failed:raise SensorContractError('RGBD ray memory fault latched')
        try:
            row=self.state.observe(policy,depth,fast,now_ns=now_ns)
            self._reader.stage(policy,row['depth_state'],row['fusion'],self.state.integrator.gravity)
            memory=self.rays.observe(policy,depth,row['depth_state'],now_ns=now_ns)
            if self._reader.pending is not None:raise SensorContractError('fusion was not consumed')
            self.last_ns=now_ns
            return row|dict(ray_memory=memory)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.failed=True; self.rays.failed=True; self._reader.pending=None
            raise SensorContractError('RGBD ray memory unavailable; stop') from error

    def query(self, points_body, ground_support_allowed, *, now_ns, backend='compiled'):
        if self.failed or self.last_ns is None or now_ns!=self.last_ns:
            raise SensorContractError('current uninterrupted RGBD memory required')
        return self.rays.query(points_body,ground_support_allowed,now_ns=now_ns,backend=backend)
