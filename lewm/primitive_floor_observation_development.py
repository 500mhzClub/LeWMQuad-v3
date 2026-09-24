"""Whole physical-primitive floor footprints in one observed depth frame.

A supplied rigid transform is a mean relative sensor pose, not simulator pose.
Its errors must already be covered by each point-error allowance. This module
checks the measured mesh but does not calibrate those allowances or establish
non-floor clearance, contact, continuous surface coverage or future motion.
"""
import hashlib
from collections import OrderedDict
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_floor_evidence_development import _sample, patch_relation
from lewm.coupled_floor_enclosure_development import coupled_footprint_enclosure
from lewm.floor_footprint_bounds_development import observed_floor_cell_index, query_floor_bounds
from lewm.primitive_floor_relation_development import primitive_floor_gap_bounds


def plane_family_cells(frame, cells_rc, anchor, normal, *, normal_error, plane_offset_error):
    """Existing measured two-triangle predicate, including incident boundaries."""
    in_image = ((cells_rc[:, 0] >= 0) & (cells_rc[:, 0] < 480)
                & (cells_rc[:, 1] >= 0) & (cells_rc[:, 1] < 640))
    adjacent = np.clip(cells_rc, [0, 0], [478, 638])
    patch, _ = _sample(frame._depth, frame._valid, adjacent)
    good = in_image & frame._index['ground_cells'][adjacent[:, 0], adjacent[:, 1]]
    for b, c in ((1, 2), (2, 3)):
        n = np.cross(patch[:, b] - patch[:, 0], patch[:, c] - patch[:, 0])
        length = np.linalg.norm(n, axis=1)
        n = np.divide(n, length[:, None], out=np.zeros_like(n), where=length[:, None] > 1e-10)
        n[(n @ frame._up) < 0] *= -1
        good &= ((length > 1e-10) & (np.linalg.norm(n - normal, axis=1) <= normal_error)
                 & (np.abs(np.sum(n * (patch[:, 0] - anchor), axis=1)) <= plane_offset_error))
    return good


def cached_plane_family_data(frame, anchor, normal, *, normal_error, plane_offset_error):
    """Bounded exact-parameter cache: return mask plus summed invalid-cell index."""
    a, n = np.asarray(anchor, float), np.asarray(normal, float)
    key = (a.tobytes(), n.tobytes(), float(normal_error), float(plane_offset_error))
    cache = frame._plane_family_cache
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    mask = np.zeros((480, 640), bool)
    cells = np.argwhere(frame._index['ground_cells'])
    if len(cells):
        values = plane_family_cells(frame, cells, a, n, normal_error=normal_error,
                                     plane_offset_error=plane_offset_error)
        mask[cells[:, 0], cells[:, 1]] = values
    # Classify only available boundary returns; outside corners stay unknown.
    mask[479, :639] = mask[478, :639]; mask[:, 639] = mask[:, 638]
    prefix = np.zeros((480, 640), np.int64)
    prefix[1:, 1:] = (~mask[:479, :639]).cumsum(axis=0).cumsum(axis=1)
    mask.flags.writeable = False; prefix.flags.writeable = False
    cache[key] = (mask, prefix)
    if len(cache) > 2: cache.popitem(last=False)
    return mask, prefix


def cached_plane_family_mask(frame, anchor, normal, **kwargs):
    return cached_plane_family_data(frame, anchor, normal, **kwargs)[0]


class PreparedFloorFrame:
    """One immutable measured frame and index, shared by repeated queries."""

    def __init__(self, depth, valid, up):
        self._depth = np.asarray(depth).copy()
        self._valid = np.asarray(valid).copy()
        self._up = np.asarray(up, float).copy()
        self._index = observed_floor_cell_index(self._depth, self._valid, self._up)
        self._minimum_valid_depth = float(np.min(np.where(self._valid, self._depth, np.inf)))
        for array in (self._depth, self._valid, self._up): array.flags.writeable = False
        self.depth_sha256 = hashlib.sha256(self._depth.tobytes() + self._valid.tobytes()).hexdigest()
        self._plane_family_cache = OrderedDict()
        self._empty_floor_mask = np.zeros((480, 640), bool)
        self._empty_floor_mask.flags.writeable = False

    def query(self, geometry, joints, seed_cell_rc, **kwargs):
        return _observe_primitive_floor(geometry, joints, self._depth, self._valid, self._up,
                                         seed_cell_rc, _prepared=self, **kwargs)


def observe_primitive_floor(geometry, joints, depth, valid, up, seed_cell_rc, **kwargs):
    return PreparedFloorFrame(depth, valid, up).query(geometry, joints, seed_cell_rc, **kwargs)


def _observe_primitive_floor(geometry, joints, depth, valid, up, seed_cell_rc, *,
                            rotation_observation_from_body, translation_observation_from_body,
                            normal_error, up_error, plane_offset_error, point_error_by_shape, _prepared,
                            floor_backend='reference'):
    """Check each complete physical AABB's floor projection, not a point seed.

    The seed must itself be an observed eligible two-triangle floor cell. It
    selects one anchored plane family; every covered cell must independently
    agree with it. No ±6-cm proximity restriction is imposed on the primitive's
    top: physical separation is computed separately using its exact minimum.
    """
    R = np.asarray(rotation_observation_from_body, float)
    t = np.asarray(translation_observation_from_body, float)
    up = np.asarray(up, float); seed = np.asarray(seed_cell_rc)
    if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all()
            or not np.isfinite(t).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-12, rtol=0)
            or abs(np.linalg.det(R) - 1) > 1e-12 or seed.shape != (2,)
            or seed.dtype.kind not in 'iu' or not (0 <= seed[0] < 479 and 0 <= seed[1] < 639)):
        raise SensorContractError('proper finite sensor-frame transform and in-image integer seed cell required')
    if floor_backend not in ('reference', 'cached'): raise SensorContractError('supported floor backend required')
    index = _prepared._index
    patch, mask = _sample(depth, valid, seed[None])
    eligible = patch_relation(patch.mean(axis=1), patch, mask, up)
    if not eligible['ground_patch_eligible'][0] or not index['ground_cells'][tuple(seed)]:
        raise SensorContractError('seed must be an observed eligible floor cell; no plane through missing/wall data')
    a = patch[0, 0]; n = np.cross(patch[0, 1] - a, patch[0, 2] - a); n /= np.linalg.norm(n)
    if n @ up < 0: n *= -1
    a_body, n_body = (a - t) @ R, n @ R
    gaps = primitive_floor_gap_bounds(geometry, joints, a_body, n_body, normal_error=normal_error,
                                      plane_offset_error=plane_offset_error, point_error_by_shape=point_error_by_shape)
    shapes = geometry.supports(joints, R)['shapes']
    low = np.array([r['lower'] for r in shapes]) + t
    high = np.array([r['upper'] for r in shapes]) + t
    radius = np.array([point_error_by_shape[r['shape_id']] for r in shapes])[:, None]
    count = len(shapes)
    bound = coupled_footprint_enclosure(low - radius, high + radius,
                                        np.broadcast_to(a, low.shape), np.broadcast_to(n, low.shape), up,
                                        normal_error=normal_error, up_error=up_error, plane_offset_error=plane_offset_error)
    coverage = query_floor_bounds(index, bound['footprint_lower_m'], bound['footprint_upper_m'],
                                  np.zeros(count), np.zeros(count), np.ones(count, bool))
    family = np.zeros(count, bool)
    family_prefix = None
    if floor_backend == 'cached' and coverage['all_projected_cells_observed_ground'].any():
        _, family_prefix = cached_plane_family_data(_prepared, a, n, normal_error=normal_error,
                                                     plane_offset_error=plane_offset_error)
    for i in np.flatnonzero(coverage['all_projected_cells_observed_ground']):
        x0, y0 = coverage['lower_cells_xy'][i]; x1, y1 = coverage['upper_cells_xy'][i]
        if family_prefix is not None:
            x1 += 1; y1 += 1
            family[i] = (family_prefix[y1, x1] - family_prefix[y0, x1]
                         - family_prefix[y1, x0] + family_prefix[y0, x0]) == 0
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        cells, _ = _sample(depth, valid, np.stack((yy.ravel(), xx.ravel()), axis=1))
        agrees = True
        for b, c in ((1, 2), (2, 3)):
            normals = np.cross(cells[:, b] - cells[:, 0], cells[:, c] - cells[:, 0])
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            normals[(normals @ up) < 0] *= -1
            agrees &= bool((np.linalg.norm(normals - n, axis=1) <= normal_error).all()
                            and (np.abs(np.sum(normals * (cells[:, 0] - a), axis=1)) <= plane_offset_error).all())
        family[i] = agrees
    floor_coverage = {r['shape_id']: bool(coverage['all_projected_cells_observed_ground'][i] and family[i])
                      for i, r in enumerate(shapes)}
    return {'gap_bounds': gaps, 'floor_coverage': floor_coverage,
            'plane_anchor_observation_m': a.copy(), 'plane_normal_observation': n.copy(),
            'lower_cells_xy': coverage['lower_cells_xy'], 'upper_cells_xy': coverage['upper_cells_xy'],
            'invalid_cells': coverage['invalid_cells'], 'covered_cells': coverage['covered_cells'],
            'measured_planes_within_supplied_family': family,
            'non_floor_clearance_established': False, 'contact_permitted': False,
            'supplied_error_bounds_validated': False, 'navigation_qualified': False}
