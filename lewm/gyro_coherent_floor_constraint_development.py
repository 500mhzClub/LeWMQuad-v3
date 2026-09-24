"""Fit paired raw floor patches with one normal transported by measured gyro.

Independent plane offsets cannot be imposed with a different rotation without
also accounting for the change in their normals. This development alternative
fits the normals jointly; it retains the existing raw-point coherence limits.
"""
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.robust_height_floor_candidates_development import (
    RESIDUAL_M, MAXIMUM_REFINEMENTS, MINIMUM_CLUSTER_FRACTION)


def fit_pair(reference, current, gyro, *, reference_pool_count, current_pool_count,
             reference_up, current_up, minimum_second_eigenvalue_m2):
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
        means = [p.mean(0) for p in selected]
        covs = [(p-m).T@(p-m)/len(p) for p, m in zip(selected, means)]
        if any(np.linalg.eigvalsh(c)[1] < minimum_second_eigenvalue_m2 for c in covs):
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


def constrain_translation(t, constraint, gyro):
    """Apply only the height equation belonging to this exact gyro rotation."""
    if not np.array_equal(np.asarray(constraint['gyro_rotation']), gyro):
        raise SensorContractError('paired floor constraint gyro identity differs')
    normal = np.asarray(constraint['reference_normal_body'])
    return t + normal*(constraint['current_offset_m'] -
        constraint['reference_offset_m'] - normal@t)
