"""Conditional finite-error footprint geometry; no navigation approval.

The plane family is n' dot (q-a) = b', with ||n'-n|| <= en,
||u'-u|| <= eu and |b'| <= eb. Nominal n and u are unit vectors.
These are supplied deterministic allowances, NOT covariance confidence bounds.
The measured-mesh check below is necessary evidence, not calibration of the
family or a guarantee about subpixel holes, camera errors, or real contacts.
"""
from itertools import product

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_floor_evidence_development import _sample, locate_floor_patches
from lewm.floor_footprint_bounds_development import observed_floor_cell_index, query_floor_bounds


def coupled_footprint_enclosure(lower, upper, anchors, normals, up, *,
                                normal_error, up_error, plane_offset_error):
    """Enclose all q-u'*[n' dot (q-a)-b']/(n' dot u') for q in boxes.

    The nominal footprint is affine in q, so its coordinate extrema occur at
    the eight box vertices. With d0=n dot u, D=en+eu+en*eu, L=max||q-a||,
    H=max|h0(q)|, the denominator is at least d0-D and
    |h'-h0| <= (en*L + eb + H*D)/(d0-D).
    Each footprint-coordinate error is at most |u_j|*dh+eu*(H+dh).
    This bound permits arbitrary dependencies between errors. It does not
    assume independent point and height extrema or a Gaussian distribution.
    """
    low, high, a, n, u = [np.asarray(x, dtype=float) for x in (lower, upper, anchors, normals, up)]
    errors = np.asarray([normal_error, up_error, plane_offset_error], dtype=float)
    if (low.ndim != 2 or low.shape[1:] != (3,) or high.shape != low.shape
            or a.shape != low.shape or n.shape != low.shape or u.shape != (3,)
            or errors.shape != (3,) or not all(np.isfinite(x).all() for x in (low, high, a, n, u, errors))
            or np.any(low > high) or np.any(errors < 0)
            or np.any(np.abs(np.linalg.norm(n, axis=1) - 1) > 1e-12)
            or abs(np.linalg.norm(u) - 1) > 1e-12):
        raise SensorContractError('ordered finite boxes, unit normals/up and nonnegative scalar error bounds required')
    en, eu, eb = errors
    d0 = n @ u
    # Use the actual floating-point norms, including their accepted numerical
    # unit-vector tolerance, in the Cauchy-Schwarz denominator bound.
    D = en * np.linalg.norm(u) + eu * np.linalg.norm(n, axis=1) + en * eu
    if np.any(d0 - D <= 1e-12) or not np.isfinite(D).all():
        raise SensorContractError('plane/up family must have a strictly positive denominator bound')
    corners = np.asarray(list(product((0, 1), repeat=3)), dtype=bool)
    vertices = np.where(corners[None], high[:, None], low[:, None])
    with np.errstate(over='ignore', invalid='ignore'):
        h0 = np.sum((vertices - a[:, None]) * n[:, None], axis=2) / d0[:, None]
        L = np.linalg.norm(vertices - a[:, None], axis=2).max(axis=1)
        H = np.abs(h0).max(axis=1)
        dh = (en * L + eb + H * D) / (d0 - D)
        feet = vertices - h0[..., None] * u
        error = dh[:, None] * np.abs(u) + eu * (H + dh)[:, None]
        foot_low, foot_high = feet.min(axis=1) - error, feet.max(axis=1) + error
        height_low, height_high = h0.min(axis=1) - dh, h0.max(axis=1) + dh
    if not all(np.isfinite(x).all() for x in (foot_low, foot_high, height_low, height_high)):
        raise SensorContractError('representable finite enclosure required')
    # Outward arithmetic allowance; not a physical error model.
    margin = 1e-12 + 128 * np.finfo(float).eps * np.maximum(np.abs(foot_low), np.abs(foot_high))
    hmargin = 1e-12 + 128 * np.finfo(float).eps * np.maximum(np.abs(height_low), np.abs(height_high))
    return {'footprint_lower_m': foot_low - margin, 'footprint_upper_m': foot_high + margin,
            'height_lower_m': height_low - hmargin, 'height_upper_m': height_high + hmargin,
            'nominal_plane_height_lower_m': h0.min(axis=1),
            'nominal_plane_height_upper_m': h0.max(axis=1),
            'height_within_existing_band': (height_low - hmargin >= -.06) & (height_high + hmargin <= .06),
            'height_error_bound_m': dh, 'denominator_lower_bound': d0 - D,
            'supplied_error_bounds_validated': False, 'ground_support_approved': False}


def query_observed_coupled_floor(depth, valid, lower, upper, up, ground_roles, *,
                                 normal_error, up_error, plane_offset_error):
    """Seed each plane from an observed patch; check ALL enclosed mesh cells.

    No plane from a missing/wall cell, no extrapolation across unchecked cells,
    no clipping the height band, and no search for a plane that makes a query
    pass. The seed is the first triangle of the located centre-footprint patch.
    Both triangles of every enclosed cell must fit the supplied plane family.
    Coverage remains conditional on that family describing the actual surface.
    This intentionally straightforward diagnostic is not a timed control path.
    """
    low, high, u = [np.asarray(x, dtype=float) for x in (lower, upper, up)]
    roles = np.asarray(ground_roles)
    # Validate bounds even if no seed is available; never let invalid arguments
    # bypass the contract through an empty/negative observation.
    if low.ndim != 2 or low.shape[1:] != (3,):
        raise SensorContractError('point boxes required')
    dummy_n = np.broadcast_to(u, low.shape) if u.shape == (3,) else np.zeros_like(low)
    coupled_footprint_enclosure(low, high, low, dummy_n, u, normal_error=normal_error,
                                up_error=up_error, plane_offset_error=plane_offset_error)
    if roles.shape != (len(low),) or roles.dtype != bool:
        raise SensorContractError('explicit boolean ground roles required')
    index = observed_floor_cell_index(depth, valid, u)
    seed = locate_floor_patches(depth, valid, low / 2 + high / 2, u)
    patches, _ = _sample(depth, valid, seed['cells_rc'])
    found = seed['observed_footprint'] & roles
    count = len(low)
    result = {'seed_observed': found, 'height_within_existing_band': np.zeros(count, bool),
              'all_projected_cells_observed_ground': np.zeros(count, bool),
              'measured_planes_within_supplied_family': np.zeros(count, bool),
              'conditional_mesh_coverage': np.zeros(count, bool),
              'invalid_cells': np.zeros(count, np.int64), 'covered_cells': np.zeros(count, np.int64),
              'lower_cells_xy': np.full((count, 2), -1, np.int64),
              'upper_cells_xy': np.full((count, 2), -1, np.int64),
              'height_lower_m': np.full(count, np.nan), 'height_upper_m': np.full(count, np.nan),
              'nominal_plane_height_lower_m': np.full(count, np.nan),
              'nominal_plane_height_upper_m': np.full(count, np.nan),
              'maximum_measured_normal_delta': np.full(count, np.nan),
              'maximum_measured_plane_offset_m': np.full(count, np.nan),
              'supplied_error_bounds_validated': False, 'continuous_surface_qualified': False,
              'ground_support_approved': False, 'navigation_qualified': False}
    for i in np.flatnonzero(found):
        a = patches[i, 0]
        n = np.cross(patches[i, 1] - a, patches[i, 2] - a)
        n /= np.linalg.norm(n)
        if n @ u < 0: n = -n
        bound = coupled_footprint_enclosure(low[i:i+1], high[i:i+1], a[None], n[None], u,
                                           normal_error=normal_error, up_error=up_error,
                                           plane_offset_error=plane_offset_error)
        # The feet have already been enclosed; zero extra height avoids the
        # independent Q x H expansion in the predecessor projection routine.
        row = query_floor_bounds(index, bound['footprint_lower_m'], bound['footprint_upper_m'],
                                 [0.], [0.], np.array([True]))
        for key in ('height_lower_m', 'height_upper_m', 'height_within_existing_band',
                    'nominal_plane_height_lower_m', 'nominal_plane_height_upper_m'):
            result[key][i] = bound[key][0]
        for key in ('lower_cells_xy', 'upper_cells_xy', 'invalid_cells', 'covered_cells',
                    'all_projected_cells_observed_ground'):
            result[key][i] = row[key][0]
        if not row['all_projected_cells_observed_ground'][0]: continue
        x0, y0 = row['lower_cells_xy'][0]; x1, y1 = row['upper_cells_xy'][0]
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        cells, _ = _sample(depth, valid, np.stack((yy.ravel(), xx.ravel()), axis=1))
        max_angle, max_offset = 0., 0.
        for b, c in ((1, 2), (2, 3)):
            tn = np.cross(cells[:, b] - cells[:, 0], cells[:, c] - cells[:, 0])
            tn /= np.linalg.norm(tn, axis=1)[:, None]
            tn[(tn @ u) < 0] *= -1
            max_angle = max(max_angle, float(np.linalg.norm(tn - n, axis=1).max()))
            max_offset = max(max_offset, float(np.abs(np.sum(tn * (cells[:, 0] - a), axis=1)).max()))
        result['maximum_measured_normal_delta'][i] = max_angle
        result['maximum_measured_plane_offset_m'][i] = max_offset
        # No tolerance beyond the supplied family: exact-zero hypotheses can
        # legitimately fail on quantized depth observations.
        agrees = max_angle <= normal_error and max_offset <= plane_offset_error
        result['measured_planes_within_supplied_family'][i] = agrees
        result['conditional_mesh_coverage'][i] = agrees and bound['height_within_existing_band'][0]
    return result
