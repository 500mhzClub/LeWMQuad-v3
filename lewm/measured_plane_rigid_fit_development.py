"""Least-squares point registration constrained by two measured floor planes.

Pure prospective estimator component, not an admitted pose or observer change.
For x_a = R x_b + t and plane equations n_a.x_a+d_a = 0,
n_b.x_b+d_b = 0, impose R n_b = n_a and n_a.t = d_b-d_a.
The remaining yaw and tangent translation minimize the paired point residual.
No previous pose, command integration, history rewrite or rejection-gate change.
"""
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.floor_pose_registration_development import unit
from lewm.joint_measured_floor_plane_development import validate_joint_plane
from lewm.joint_rgbd_rigid_pose_development import proper, scatter


def fit(a, b, reference_plane, current_plane):
    """Fit supplied correspondences; robust consensus and pose admission are external.

Planes must carry the original joint-plane validation evidence. Caller retains
responsibility for actual packet identities and the same static floor identity.
No supplied correspondence is pruned or silently downweighted.
"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise SensorContractError('paired point shapes required')
    sa, sb = scatter(a), scatter(b)
    for plane in (reference_plane, current_plane):
        validate_joint_plane(plane, np.asarray(plane['up_body']))
    na, nb = unit(reference_plane['normal_body']), unit(current_plane['normal_body'])
    da, db = float(reference_plane['offset_body_m']), float(current_plane['offset_body_m'])
    # Deterministic tangent bases avoid a singular shortest-arc rotation when
    # normals oppose. This is a coordinate choice, not a gravity measurement.
    def basis(n):
        e = np.eye(3)[int(np.argmin(np.abs(n)))]
        u = e-n*float(n@e)
        u /= np.linalg.norm(u)
        return np.column_stack((u, np.cross(n, u), n))
    A, B = basis(na), basis(nb)
    ac, bc = a-a.mean(0), b-b.mean(0)
    x, y = ac@A[:, :2], bc@B[:, :2]
    cosine = float(np.sum(x*y))
    sine = float(np.sum(y[:, 0]*x[:, 1]-y[:, 1]*x[:, 0]))
    scale = float(np.linalg.norm(x)*np.linalg.norm(y))
    magnitude = float(np.hypot(cosine, sine))
    if not np.isfinite([scale, magnitude]).all() or scale <= 0 or magnitude <= 1e-12*scale:
        raise SensorContractError('paired tangent yaw must be identifiable')
    c, s = cosine/magnitude, sine/magnitude
    R = proper(A@np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])@B.T)
    unconstrained_t = a.mean(0)-R@b.mean(0)
    t = unconstrained_t+na*(db-da-float(na@unconstrained_t))
    residual = a-b@R.T-t
    return R, t, dict(method='measured_plane_constrained_point_least_squares_v1',
        points=len(a), reference_scatter_rms_m=sa.tolist(), current_scatter_rms_m=sb.tolist(),
        residual_rms_m=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        plane_normal_residual=float(np.linalg.norm(R@nb-na)),
        plane_offset_residual_m=float(na@t-(db-da)),
        tangent_yaw_identifiability=magnitude/scale,
        measured_plane_constraints_used=True, all_supplied_points_used=True,
        static_same_floor_hypothesis=True, floor_identity_certified=False,
        robust_consensus_performed=False, raw_packet_identity_admitted=False,
        gyro_consistency_checked=False, original_pose_gates_checked=False,
        previous_pose_reset=False, command_integration_used=False, native_pose_used=False,
        historical_map_rewritten=False, calibrated_uncertainty=False,
        pose_admitted=False, navigation_qualified=False)
