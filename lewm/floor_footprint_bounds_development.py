"""Coverage of explicitly supplied floor-footprint bounds, not calibrated poses.

Checks EVERY image cell in a conservative projection rectangle. Small paired
perturbations are not substituted for this region. The caller must establish
the body-point, height and up bounds; this module does not derive them from a
covariance or approve navigation. Subpixel surface interpolation remains an
explicit assumption, not a continuous physical-scene certificate.
"""
from itertools import product
from types import MappingProxyType

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError


def observed_floor_cell_index(depth, valid, up):
    """Immutable summed invalid-cell index for the measured two-triangle mesh."""
    d, valid, up = np.asarray(depth), np.asarray(valid), np.asarray(up, dtype=float)
    if (d.shape != (480, 640) or valid.shape != d.shape or valid.dtype != bool
            or not np.isfinite(d).all() or np.any(d[~valid] != 0.)
            or np.any((d[valid] < .2) | (d[valid] > 5.)) or up.shape != (3,)
            or not np.isfinite(up).all() or abs(np.linalg.norm(up) - 1) > 1e-6):
        raise SensorContractError('measured depth grid and unit up required')
    transform = np.asarray(BODY_FROM_OPTICAL)
    u = (np.arange(640) + .5 - 320) / FOCAL
    v = (np.arange(480) + .5 - 240) / FOCAL
    optical = np.stack((d * u[None], d * v[:, None], d), axis=2)
    points = optical @ transform[:3, :3].T + transform[:3, 3]
    a, b, c, e = points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1]
    good = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1]
    for p in (a, b, c, e): good &= p @ up < -.15
    for left, right in ((b, e), (b, c), (c, e)):
        normal = np.cross(left - a, right - a)
        length = np.linalg.norm(normal, axis=2)
        good &= (length > 1e-10) & (np.abs(normal @ up) >= .97 * length)
        if left is b and right is e:
            # Same fourth-point planarity gate as patch_relation.
            good &= np.abs(np.sum((c - a) * normal, axis=2)) <= .003 * length
    prefix = np.zeros((480, 640), np.int64)
    prefix[1:, 1:] = (~good).cumsum(axis=0).cumsum(axis=1)
    result = {'ground_cells': good.copy(), 'invalid_cell_prefix': prefix, 'up': up.copy()}
    for array in result.values(): array.flags.writeable = False
    return MappingProxyType(result)


def projected_footprint_rectangles(lower, upper, height_lower, height_upper, up):
    """Enclose q - h*up for every point q in each supplied 3-D box.

Projection extrema of this convex prism occur at its vertices when all optical
depths are positive. The rectangle is deliberately conservative; an unobserved
rectangle corner cannot be ignored just because the actual region is smaller.
Up is fixed, so uncertainty in up must already be included by the caller's
enclosure construction. No such inclusion is assumed or certified here.
"""
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    hl, hh, up = [np.asarray(a, dtype=float) for a in (height_lower, height_upper, up)]
    if (low.ndim != 2 or low.shape[1:] != (3,) or high.shape != low.shape
            or hl.shape != (len(low),) or hh.shape != hl.shape or up.shape != (3,)
            or not all(np.isfinite(a).all() for a in (low, high, hl, hh, up))
            or np.any(low > high) or np.any(hl > hh) or np.any(hl < -.06) or np.any(hh > .06)
            or abs(np.linalg.norm(up) - 1.) > 1e-6):
        raise SensorContractError('finite ordered point/height bounds within the existing 6-cm band required')
    corners = np.asarray(list(product((0, 1), repeat=3)), dtype=bool)
    vertices = np.where(corners[None], high[:, None], low[:, None])
    feet = vertices[:, :, None, :] - np.stack((hl, hh), axis=1)[:, None, :, None] * up
    transform = np.asarray(BODY_FROM_OPTICAL)
    optical = (feet.reshape(len(low), 16, 3) - transform[:3, 3]) @ transform[:3, :3]
    if not np.isfinite(optical).all(): raise SensorContractError('representable footprint coordinates required')
    depth = optical[:, :, 2]
    in_front = ((depth >= .2) & (depth <= 5.)).all(axis=1)
    with np.errstate(over='ignore', invalid='ignore'):
        uv = FOCAL * optical[:, :, :2] / np.maximum(depth[:, :, None], 1e-12)
        uv += np.array([319.5, 239.5])
    if not np.isfinite(uv).all(): raise SensorContractError('representable footprint projection required')
    minimum, maximum = uv.min(axis=1), uv.max(axis=1)
    # Outward numerical allowance also includes cells on both sides of an
    # exact pixel-grid boundary; it must never shrink a requested region.
    margin = 1e-9 + 64 * np.finfo(float).eps * np.maximum(np.abs(minimum), np.abs(maximum))
    minimum -= margin; maximum += margin
    complete = (in_front & np.isfinite(minimum).all(axis=1) & np.isfinite(maximum).all(axis=1)
                & (minimum[:, 0] >= 0) & (maximum[:, 0] < 639)
                & (minimum[:, 1] >= 0) & (maximum[:, 1] < 479))
    lo_cell = np.floor(np.clip(minimum, -1, 640)).astype(np.int64)
    hi_cell = np.floor(np.clip(maximum, -1, 640)).astype(np.int64)
    return {'lower_cells_xy': lo_cell, 'upper_cells_xy': hi_cell,
            'projection_within_observed_camera': complete}


def query_floor_bounds(index, lower, upper, height_lower, height_upper, ground_roles):
    rectangles = projected_footprint_rectangles(lower, upper, height_lower, height_upper, index['up'])
    roles = np.asarray(ground_roles)
    if roles.shape != (len(lower),) or roles.dtype != bool:
        raise SensorContractError('explicit ground-role vector required')
    low, high = rectangles['lower_cells_xy'], rectangles['upper_cells_xy']
    complete = rectangles['projection_within_observed_camera'] & roles
    invalid = np.zeros(len(roles), np.int64); covered = np.zeros(len(roles), np.int64)
    ids = np.flatnonzero(complete)
    if len(ids):
        x0, y0 = low[ids].T; x1, y1 = (high[ids] + 1).T
        prefix = index['invalid_cell_prefix']
        invalid[ids] = prefix[y1, x1] - prefix[y0, x1] - prefix[y1, x0] + prefix[y0, x0]
        covered[ids] = (x1 - x0) * (y1 - y0)
    return rectangles | {'all_projected_cells_observed_ground': complete & (invalid == 0),
                          'invalid_cells': invalid, 'covered_cells': covered,
                          'coverage_conditional_on_supplied_bounds_and_fixed_up': True,
                          'supplied_bounds_validated': False, 'uncertain_surface_geometry_covered': False,
                          'ground_support_approved': False, 'navigation_qualified': False}
