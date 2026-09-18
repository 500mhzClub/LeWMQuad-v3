"""Confirm auxiliary floor patches using current primary measured floor seeds.

No pose is corrected. All samples remain measured; a plane does not establish
free space. This pure candidate helper is not installed in a live controller.
"""
import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.observed_geometry_refinement_development import sampled_floor_patch
from lewm.joint_rgbd_rigid_pose_development import proper

ROWS = np.arange(2, 480, 4)
COLUMNS = np.arange(2, 640, 4)


def measured_points(depth, rotation_map_from_reference, position_map):
    R = proper(rotation_map_from_reference); p = np.asarray(position_map, float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('finite observed reference position required')
    yy, xx = np.indices((480, 640)); T = np.asarray(BODY_FROM_OPTICAL)
    optical = np.stack((depth*(xx+.5-320)/FOCAL, depth*(yy+.5-240)/FOCAL, depth), axis=-1)
    return (optical@T[:3,:3].T+T[:3,3])@R.T+p


def primary_floor_plane(depth, valid, rotation_map_from_body, position_map, floor_height):
    # Seed only from the predecessor's complete measured primary floor patches.
    patch = sampled_floor_patch(depth, valid, rotation_map_from_body, position_map,
        floor_height, ROWS, COLUMNS)
    points = measured_points(depth, rotation_map_from_body, position_map)
    seeds = points[np.ix_(ROWS, COLUMNS)][patch['measured_floor_patch']]
    receipt = dict(available=False, seed_count=len(seeds), original_seed_band_m=.01,
        maximum_fit_residual_m=.003, minimum_second_covariance_eigenvalue_m2=.05**2,
        minimum_up_alignment=.97, native_pose_used=False, floor_or_support_certified=False)
    if len(seeds) < 100: return receipt | dict(reason='insufficient_original_floor_patches')
    center = seeds.mean(0); delta = seeds-center
    values, vectors = np.linalg.eigh(delta.T@delta/len(seeds)); normal = vectors[:, 0]
    if normal[2] < 0: normal = -normal
    offset = -float(normal@center); errors = seeds@normal+offset
    receipt.update(normal_map=normal.tolist(), offset_m=offset,
        seed_xy_lower_m=seeds[:, :2].min(0).tolist(), seed_xy_upper_m=seeds[:, :2].max(0).tolist(),
        covariance_eigenvalues_m2=values.tolist(), maximum_absolute_seed_residual_m=float(np.abs(errors).max()))
    if values[1] < .05**2: return receipt | dict(reason='insufficient_two_axis_floor_extent')
    if normal[2] < .97: return receipt | dict(reason='plane_up_disagreement')
    if np.abs(errors).max() > .003: return receipt | dict(reason='primary_floor_not_one_coherent_plane')
    return receipt | dict(available=True, reason='current_primary_measured_floor_plane')


def confirm_auxiliary_floor(depth, valid, rotation_map_from_reference, position_map,
        floor_height, rows, columns, plane):
    original = sampled_floor_patch(depth, valid, rotation_map_from_reference, position_map,
        floor_height, rows, columns)['measured_floor_patch']
    receipt = dict(original_floor_count=int(original.sum()), added_floor_count=0,
        measured_plane_band_m=.01, maximum_original_plane_height_difference_m=.03,
        primary_plane_extrapolation_requires_measured_auxiliary_patch=True,
        original_floor_classifications_preserved=True,
        original_ground_mesh_checks_preserved=True, unobserved_space_certified=False,
        ground_support_approved=False, non_foot_contact_exemption=False)
    if not plane['available']: return original.copy(), receipt
    R = proper(rotation_map_from_reference)
    points = measured_points(depth, R, position_map)
    normal = np.asarray(plane['normal_map'], float)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or abs(np.linalg.norm(normal)-1.) > 1e-8 or normal[2] < .97
            or not np.isfinite(plane['offset_m'])):
        raise ValueError('finite admitted primary plane required')
    index = observed_floor_cell_index(depth, valid, R[2])
    near = (valid & (np.abs(points@normal+plane['offset_m']) <= .01)
        & (np.abs(points[:, :, 2]-floor_height) <= .03))
    good = np.ones_like(original)
    for dr in (-1, 0):
        for dc in (-1, 0): good &= index['ground_cells'][np.ix_(rows+dr, columns+dc)]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1): good &= near[np.ix_(rows+dr, columns+dc)]
    revised = original | good
    receipt['added_floor_count'] = int((revised & ~original).sum())
    return revised, receipt
