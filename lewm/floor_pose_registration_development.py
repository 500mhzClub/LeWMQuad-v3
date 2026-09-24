"""Measured-plane pose correction under an explicit static flat-floor hypothesis.

Pure candidate geometry: no command integration, scene query, classification,
history rewrite, or calibrated error bound. The original RGB-D pose must remain
separately available. These functions alone do not admit a controller pose.
"""
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.joint_rgbd_rigid_pose_development import proper

ROWS = np.arange(2, 480, 4)
COLUMNS = np.arange(2, 640, 4)


def unit(value):
    n = np.asarray(value, float)
    if n.shape != (3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1.) > 1e-8:
        raise ValueError('finite unit plane normal required')
    return n


def measured_candidates(depth, valid, body_from_optical, up_body):
    """All four surrounding measured quads and nine valid pixels, no height band."""
    E = np.asarray(body_from_optical, float)
    if (E.shape != (4, 4) or not np.isfinite(E).all()
            or not np.array_equal(E[3], [0., 0., 0., 1.])):
        raise ValueError('finite rigid camera mount required')
    proper(E[:3, :3]); up = unit(up_body)
    T = np.asarray(BODY_FROM_OPTICAL)
    Q = E[:3, :3]@T[:3, :3].T
    index = observed_floor_cell_index(depth, valid, Q.T@up)
    # The original mesh test includes reference-body height < -0.15 m.
    # Also require that bound in the actual body frame below.
    good = np.ones((len(ROWS), len(COLUMNS)), bool)
    for dr in (-1, 0):
        for dc in (-1, 0):
            good &= index['ground_cells'][np.ix_(ROWS+dr, COLUMNS+dc)]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            good &= valid[np.ix_(ROWS+dr, COLUMNS+dc)]
    yy, xx = np.meshgrid(ROWS+.5, COLUMNS+.5, indexing='ij')
    z = depth[np.ix_(ROWS, COLUMNS)]
    optical = np.stack((z*(xx-320)/FOCAL, z*(yy-240)/FOCAL, z), axis=-1)
    points = optical@E[:3, :3].T+E[:3, 3]
    good &= points@up < -.15
    return points[good], good


def fit_measured_plane(points, up_body):
    """Strict one-plane fit; rejected candidates are never silently trimmed."""
    points = np.asarray(points, float); up = unit(up_body)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) > 19200
            or not np.isfinite(points).all()):
        raise ValueError('bounded finite measured candidates required')
    receipt = dict(available=False, candidate_count=len(points), minimum_candidates=100,
        maximum_allowed_residual_m=.003, minimum_second_eigenvalue_m2=.05**2,
        minimum_up_alignment=.97, absolute_initial_height_band_used=False,
        candidates_trimmed=False, floor_identity_certified=False)
    if len(points) < 100: return receipt | dict(reason='insufficient_measured_candidates')
    center = points.mean(0); delta = points-center
    values, vectors = np.linalg.eigh(delta.T@delta/len(points))
    normal = vectors[:, 0]
    if normal@up < 0: normal = -normal
    offset = -float(normal@center); error = points@normal+offset
    receipt.update(normal_body=normal.tolist(), offset_body_m=offset,
        covariance_eigenvalues_m2=values.tolist(), maximum_residual_m=float(np.abs(error).max()),
        rms_residual_m=float(np.sqrt(np.mean(error**2))))
    if values[1] < .05**2: return receipt | dict(reason='insufficient_two_axis_extent')
    if normal@up < .97: return receipt | dict(reason='up_disagreement')
    if np.abs(error).max() > .003: return receipt | dict(reason='multiple_or_incoherent_planes')
    return receipt | dict(available=True, reason='measured_plane_under_flat_floor_hypothesis')


def paired_plane(primary, auxiliary):
    """Both cameras must independently admit compatible planes in BODY coordinates."""
    if not primary['available'] or not auxiliary['available']:
        raise ValueError('two current independently admitted measured planes required')
    a, b = unit(primary['normal_body']), unit(auxiliary['normal_body'])
    distance = abs(float(primary['offset_body_m'])-float(auxiliary['offset_body_m']))
    angle = float(np.arctan2(np.linalg.norm(np.cross(a, b)), a@b))
    if not np.isfinite(distance) or distance > .003 or angle > .01:
        raise ValueError('current primary/auxiliary plane disagreement')
    # Equal camera weighting, independent of how much of each image saw floor.
    n = a+b; length = np.linalg.norm(n)
    d = (float(primary['offset_body_m'])+float(auxiliary['offset_body_m']))/length
    return dict(normal_body=(n/length).tolist(), offset_body_m=d,
        intercamera_angle_rad=angle, intercamera_offset_difference_m=distance,
        static_flat_floor_hypothesis=True, physical_floor_identity_certified=False)


def register_pose(raw_position, raw_rotation, reference_plane, current_plane):
    """Constrain tilt/height; preserve raw in-plane position and forward azimuth.

    For x_initial = R*x_body+p, plane offset transforms as d_initial =
    d_body - n_initial.p. Minimal normal alignment is followed by a twist about
    the reference normal that restores the original projected forward heading.
    The 5 cm / 0.10 rad limits are development rejection gates, not error bounds.
    """
    p = np.asarray(raw_position, float); R = proper(raw_rotation)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('finite raw visual position required')
    n = unit(reference_plane['normal_body']); b = unit(current_plane['normal_body'])
    d0, d = float(reference_plane['offset_body_m']), float(current_plane['offset_body_m'])
    if not np.isfinite([d0, d]).all(): raise ValueError('finite plane offsets required')
    a = R@b; v = np.cross(a, n); c = float(a@n)
    tilt = float(np.arctan2(np.linalg.norm(v), c))
    height = d-d0-float(n@p)
    if tilt > .10 or abs(height) > .05:
        raise ValueError('floor registration exceeds fixed development correction gates')
    K = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    aligned = (np.eye(3)+K+K@K/(1.+c))@R
    old = R[:, 0]-n*(n@R[:, 0]); new = aligned[:, 0]-n*(n@aligned[:, 0])
    if min(np.linalg.norm(old), np.linalg.norm(new)) < .2:
        raise ValueError('nonvertical measured heading required')
    old /= np.linalg.norm(old); new /= np.linalg.norm(new)
    twist = float(np.arctan2(n@np.cross(new, old), new@old))
    N = np.array([[0., -n[2], n[1]], [n[2], 0., -n[0]], [-n[1], n[0], 0.]])
    corrected_R = proper((np.eye(3)+np.sin(twist)*N+(1.-np.cos(twist))*(N@N))@aligned)
    corrected_p = p+height*n
    return dict(position_initial_body_m=corrected_p.tolist(),
        rotation_initial_body_from_current_body=corrected_R.tolist(),
        raw_position_initial_body_m=p.tolist(), raw_rotation_initial_body_from_current_body=R.tolist(),
        normal_alignment_rad=tilt, heading_restoration_twist_rad=twist,
        normal_translation_correction_m=height, maximum_correction_m=.05, maximum_tilt_correction_rad=.10,
        in_plane_translation_preserved=True, in_plane_forward_azimuth_preserved=True,
        static_flat_floor_hypothesis=True, native_pose_used=False, command_integration_used=False,
        original_visual_witness_replaced=False, position_error_bound=None,
        orientation_error_bound=None, uncertainty_model_validated=False,
        floor_or_support_certified=False, navigation_qualified=False)
