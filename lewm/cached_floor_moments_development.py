"""Replay-only reuse of individual retained raw-floor cloud statistics.

Pair-dependent normal fitting, residuals, pruning and all acceptance limits
remain unchanged. Only full-cloud mean, covariance and second eigenvalue are
reused; every pruned subset is reduced anew in its original order.
"""
from collections import OrderedDict
from functools import partial
from types import FunctionType
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.robust_height_floor_candidates_development import (
    RESIDUAL_M, MAXIMUM_REFINEMENTS, MINIMUM_CLUSTER_FRACTION)
from lewm.batched_consensus_tracking_development import BatchedConsensusPose, BatchedConsensusMotion
from lewm.gyro_coherent_floor_tracking_development import GyroCoherentFloorPose


class FloorCloudMoments:
    def __init__(self, capacity=32):
        if type(capacity) is not int or capacity < 2:
            raise ValueError('at least two bounded cloud entries required')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = self.misses = self.pruned_reductions = 0

    def __call__(self, owner, selected, full_cloud):
        if owner.flags.writeable or not owner.flags.owndata:
            raise ValueError('owned read-only raw floor arrays required')
        key = id(owner)
        if full_cloud and key in self.entries:
            retained, statistics = self.entries.pop(key)
            assert retained is owner
            self.entries[key] = (retained, statistics)
            self.hits += 1
            return statistics
        mean = selected.mean(0)
        covariance = (selected-mean).T@(selected-mean)/len(selected)
        extent = np.linalg.eigvalsh(covariance)[1]
        mean.flags.writeable = covariance.flags.writeable = False
        statistics = (mean, covariance, extent)
        if full_cloud:
            self.misses += 1
            self.entries[key] = (owner, statistics)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        else:
            self.pruned_reductions += 1
        return statistics


def fit_pair_with_moments(reference, current, gyro, *, reference_pool_count, current_pool_count,
             reference_up, current_up, minimum_second_eigenvalue_m2, moments):
    G = proper(gyro)
    arrays = [np.asarray(p, float) for p in (reference, current)]
    pools = (reference_pool_count, current_pool_count)
    ups = [np.asarray(u, float) for u in (reference_up, current_up)]
    if any(p.ndim != 2 or p.shape[1:] != (3,) or len(p) > 38400 or
           not np.isfinite(p).all() for p in arrays):
        raise SensorContractError('bounded finite paired raw floor points required')
    if any(type(pool) is not int or not len(p) <= pool <= 38400
           for p, pool in zip(arrays, pools)):
        raise SensorContractError('original bounded floor pool counts required')
    if (any(u.shape != (3,) or not np.isfinite(u).all() or
            not np.isclose(np.linalg.norm(u), 1., atol=1e-10, rtol=0) for u in ups)
            or not np.isfinite(minimum_second_eigenvalue_m2)
            or minimum_second_eigenvalue_m2 < .0004):
        raise SensorContractError('measured up and original floor extent required')
    masks = [np.ones(len(p), bool) for p in arrays]
    for iteration in range(MAXIMUM_REFINEMENTS):
        selected = [p[m] for p, m in zip(arrays, masks)]
        if any(len(p) < max(100, MINIMUM_CLUSTER_FRACTION*pool)
               for p, pool in zip(selected, pools)):
            raise SensorContractError('insufficient paired floor support after pruning')
        statistics = [moments(p, chosen, bool(mask.all()))
            for p, chosen, mask in zip(arrays, selected, masks)]
        means = [s[0] for s in statistics]
        covs = [s[1] for s in statistics]
        if any(s[2] < minimum_second_eigenvalue_m2 for s in statistics):
            raise SensorContractError('paired floor patch lacks original two-axis extent')
        total = sum(map(len, selected))
        _, vectors = np.linalg.eigh((len(selected[0])*covs[0] +
            len(selected[1])*G@covs[1]@G.T)/total)
        n = vectors[:, 0]
        if n@ups[0] < 0: n = -n
        normals = (n, G.T@n)
        if any(normal@up < .97 for normal, up in zip(normals, ups)):
            raise SensorContractError('paired floor normal disagrees with measured up')
        residuals = [np.abs((p-m)@normal) for p, m, normal in zip(selected, means, normals)]
        if all(r.max() <= RESIDUAL_M for r in residuals):
            return dict(reference_normal_body=n.tolist(),
                reference_offset_m=-float(n@means[0]),
                current_normal_body=(G.T@n).tolist(),
                current_offset_m=-float((G.T@n)@means[1]),
                gyro_rotation=G.tolist(), refinement_steps=iteration+1,
                raw_input_counts=[len(p) for p in arrays], raw_pool_counts=list(pools),
                retained_counts=[int(m.sum()) for m in masks],
                excluded_input_indices=[np.flatnonzero(~m).tolist() for m in masks],
                maximum_residual_m=[float(r.max()) for r in residuals],
                rms_residual_m=[float(np.sqrt(np.mean(r*r))) for r in residuals],
                maximum_allowed_residual_m=RESIDUAL_M,
                minimum_second_eigenvalue_m2=minimum_second_eigenvalue_m2,
                common_normal_fitted_from_both_raw_patches=True,
                normal_transport_uses_public_gyro=True, native_pose_used=False,
                uncertainty_calibrated=False)
        for mask, residual in zip(masks, residuals):
            mask[np.flatnonzero(mask)[residual > RESIDUAL_M]] = False
    raise SensorContractError('bounded paired floor refinement did not converge')


class CachedFloorMomentsPose(BatchedConsensusPose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.floor_moments = FloorCloudMoments()
        original = GyroCoherentFloorPose._refine_candidate
        copied = FunctionType(original.__code__, original.__globals__ | dict(
            fit_pair=partial(fit_pair_with_moments, moments=self.floor_moments)),
            original.__name__, original.__defaults__, original.__closure__)
        copied.__kwdefaults__ = original.__kwdefaults__
        self._refine_candidate = copied.__get__(self, type(self))

    def _raw_floor(self, features, receipt):
        points = super()._raw_floor(features, receipt)
        points.flags.writeable = False
        return points


class CachedFloorMomentsMotion(BatchedConsensusMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = CachedFloorMomentsPose(activation_frame=activation_frame)
