"""Fixed-patch measured plane; no uncertainty calibration or motion permission.

All 18x18 observed pixels contribute once to a total-least-squares fit. Every
one of its 17x17 cells must also satisfy the existing measured triangle gates.
There is no outlier deletion, patch search, hole filling, or world-floor input.
The tolerances are hypotheses about measured consistency, NOT sensor bounds.
"""
from dataclasses import dataclass

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.coupled_floor_enclosure_development import coupled_footprint_enclosure
from lewm.floor_footprint_bounds_development import query_floor_bounds
from lewm.measured_plane_obstacle_memory_development import MeasuredPlaneHypothesis
from lewm.primitive_floor_observation_development import (
    PreparedFloorFrame, cached_plane_family_data, plane_family_cells)
from lewm.primitive_floor_relation_development import primitive_floor_gap_bounds


@dataclass(frozen=True)
class PatchRules:
    radius_cells: int = 8
    minimum_tangent_rms_m: float = .005
    minimum_tangent_ratio: float = .02
    maximum_pixel_residual_m: float = .001
    maximum_triangle_normal_error: float = .002
    maximum_triangle_offset_m: float = .001

    def __post_init__(self):
        values = np.asarray([self.minimum_tangent_rms_m, self.minimum_tangent_ratio,
                             self.maximum_pixel_residual_m, self.maximum_triangle_normal_error,
                             self.maximum_triangle_offset_m], float)
        if (type(self.radius_cells) is not int or not 1 <= self.radius_cells <= 32
                or values.shape != (5,) or not np.isfinite(values).all() or np.any(values <= 0)
                or self.minimum_tangent_ratio > 1 or self.maximum_triangle_normal_error >= 1):
            raise SensorContractError('finite positive fixed-patch consistency rules required')


@dataclass(frozen=True)
class MultiPixelPlane:
    depth_sha256: str
    up: tuple
    seed_cell_rc: tuple | None
    rules: PatchRules
    status: str
    anchor: tuple | None = None
    normal: tuple | None = None
    tangent_rms_m: tuple | None = None
    maximum_pixel_residual_m: float | None = None
    policy: str = 'median-eligible-fixed-patch-tls-v1'

    def for_frame(self, frame):
        if (not isinstance(frame, PreparedFloorFrame) or self.depth_sha256 != frame.depth_sha256
                or self.up != tuple(frame._up) or self.policy != 'median-eligible-fixed-patch-tls-v1'):
            raise SensorContractError('fixed-patch observation identity mismatch')
        if self.status != 'MEASURED_PATCH_CONSISTENT':
            raise SensorContractError('fixed-patch plane unavailable; retain unknown')
        return np.asarray(self.anchor, float), np.asarray(self.normal, float)


def fit_multipixel_plane(frame, *, rules=PatchRules()):
    """Select median eligible cell ONCE, then accept or reject its fixed patch."""
    if not isinstance(frame, PreparedFloorFrame) or not isinstance(rules, PatchRules):
        raise SensorContractError('prepared frame and explicit fixed-patch rules required')
    seed = MeasuredPlaneHypothesis.from_frame(frame).cell_for(frame)
    common = dict(depth_sha256=frame.depth_sha256, up=tuple(float(x) for x in frame._up),
                  seed_cell_rc=seed, rules=rules)
    if seed is None:
        return MultiPixelPlane(**common, status='NO_ELIGIBLE_CELL')
    r, c = seed; k = rules.radius_cells
    if r-k < 0 or c-k < 0 or r+k >= 479 or c+k >= 639:
        return MultiPixelPlane(**common, status='FIXED_PATCH_OUTSIDE_IMAGE')
    if not frame._index['ground_cells'][r-k:r+k+1, c-k:c+k+1].all():
        return MultiPixelPlane(**common, status='FIXED_PATCH_MISSING_OR_INELIGIBLE')
    yy, xx = np.mgrid[r-k:r+k+2, c-k:c+k+2]
    z = frame._depth[yy, xx].astype(float)
    optical = np.stack((z*(xx+.5-320)/FOCAL, z*(yy+.5-240)/FOCAL, z), axis=-1)
    transform = np.asarray(BODY_FROM_OPTICAL)
    points = (optical @ transform[:3, :3].T + transform[:3, 3]).reshape(-1, 3)
    a = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points-a, full_matrices=False)
    rms = singular / np.sqrt(len(points))
    if (not np.isfinite(singular).all() or rms[1] < rules.minimum_tangent_rms_m
            or singular[1] < rules.minimum_tangent_ratio*singular[0]):
        return MultiPixelPlane(**common, status='FIXED_PATCH_ILL_CONDITIONED',
                               tangent_rms_m=tuple(float(x) for x in rms[:2]))
    n = vectors[-1]
    if n @ frame._up < 0: n = -n
    residual = float(np.max(np.abs((points-a) @ n)))
    diagnostics = dict(tangent_rms_m=tuple(float(x) for x in rms[:2]),
                       maximum_pixel_residual_m=residual)
    if n @ frame._up < .97 or residual > rules.maximum_pixel_residual_m:
        return MultiPixelPlane(**common, **diagnostics, status='FIXED_PATCH_RESIDUAL_OR_ORIENTATION')
    y, x = np.mgrid[r-k:r+k+1, c-k:c+k+1]
    agrees = plane_family_cells(frame, np.column_stack((y.ravel(), x.ravel())), a, n,
        normal_error=rules.maximum_triangle_normal_error,
        plane_offset_error=rules.maximum_triangle_offset_m)
    if not agrees.all():
        return MultiPixelPlane(**common, **diagnostics, status='FIXED_PATCH_TRIANGLE_DISAGREEMENT')
    return MultiPixelPlane(**common, **diagnostics, status='MEASURED_PATCH_CONSISTENT',
                           anchor=tuple(float(x) for x in a), normal=tuple(float(x) for x in n))


def query_multipixel_floor(frame, plane, geometry, joints, *,
                           rotation_observation_from_body, translation_observation_from_body,
                           normal_error, up_error, plane_offset_error, point_error_by_shape,
                           floor_backend='reference'):
    """Use the fitted plane explicitly, but independently check EVERY footprint cell.

    Supplied errors must cover sensor, fit, kinematic and relative-pose errors.
    Fit residuals are deliberately never substituted for those unknown bounds.
    """
    if not isinstance(plane, MultiPixelPlane):
        raise SensorContractError('explicit multipixel measured-plane hypothesis required')
    a, n = plane.for_frame(frame)
    R = np.asarray(rotation_observation_from_body, float)
    t = np.asarray(translation_observation_from_body, float)
    if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all()
            or not np.isfinite(t).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-12, rtol=0)
            or abs(np.linalg.det(R)-1) > 1e-12 or floor_backend not in ('reference', 'cached')):
        raise SensorContractError('proper finite relative transform and supported floor backend required')
    gaps = primitive_floor_gap_bounds(geometry, joints, (a-t)@R, n@R,
        normal_error=normal_error, plane_offset_error=plane_offset_error,
        point_error_by_shape=point_error_by_shape)
    shapes = geometry.supports(joints, R)['shapes']
    low = np.array([s['lower'] for s in shapes])+t
    high = np.array([s['upper'] for s in shapes])+t
    radius = np.array([point_error_by_shape[s['shape_id']] for s in shapes])[:, None]
    bounds = coupled_footprint_enclosure(low-radius, high+radius,
        np.broadcast_to(a, low.shape), np.broadcast_to(n, low.shape), frame._up,
        normal_error=normal_error, up_error=up_error, plane_offset_error=plane_offset_error)
    count = len(shapes)
    coverage = query_floor_bounds(frame._index, bounds['footprint_lower_m'], bounds['footprint_upper_m'],
                                   np.zeros(count), np.zeros(count), np.ones(count, bool))
    family = np.zeros(count, bool)
    complete = np.flatnonzero(coverage['all_projected_cells_observed_ground'])
    prefix = None
    if floor_backend == 'cached' and len(complete):
        _, prefix = cached_plane_family_data(frame, a, n, normal_error=normal_error,
                                              plane_offset_error=plane_offset_error)
    for i in complete:
        x0, y0 = coverage['lower_cells_xy'][i]; x1, y1 = coverage['upper_cells_xy'][i]
        if prefix is not None:
            family[i] = (prefix[y1+1, x1+1]-prefix[y0, x1+1]
                         -prefix[y1+1, x0]+prefix[y0, x0]) == 0
        else:
            yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
            family[i] = plane_family_cells(frame, np.column_stack((yy.ravel(), xx.ravel())), a, n,
                normal_error=normal_error, plane_offset_error=plane_offset_error).all()
    return dict(gap_bounds=gaps,
        floor_coverage={s['shape_id']: bool(coverage['all_projected_cells_observed_ground'][i] and family[i])
                        for i, s in enumerate(shapes)},
        plane_anchor_observation_m=a, plane_normal_observation=n,
        lower_cells_xy=coverage['lower_cells_xy'], upper_cells_xy=coverage['upper_cells_xy'],
        invalid_cells=coverage['invalid_cells'], covered_cells=coverage['covered_cells'],
        measured_planes_within_supplied_family=family, plane_fit_status=plane.status,
        non_floor_clearance_established=False, contact_permitted=False,
        supplied_error_bounds_validated=False, navigation_qualified=False, future_gait_qualified=False)
