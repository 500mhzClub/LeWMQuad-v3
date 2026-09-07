"""Range-bounded measured mesh around a FIXED reference plane.

The true surface need not be planar. Every admissible measured vertex lies in
the declared residual tube; linear interpolation keeps each triangle there.
This is conditional on range bounds, exact camera geometry and piecewise-linear
surface interpolation. It is not a calibration, contact or motion certificate.
"""
from itertools import product

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.coupled_floor_enclosure_development import coupled_footprint_enclosure
from lewm.floor_footprint_bounds_development import query_floor_bounds
from lewm.measured_plane_obstacle_memory_development import MeasuredPlaneHypothesis
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.primitive_floor_relation_development import primitive_floor_gap_bounds


def triangle_orientation_bounds(rays, lower, upper, up, up_error):
    """Exact endpoint extrema of cross-products projected along nominal up.

    With a fixed camera centre the normal is multiaffine in the three ranges.
    Its projected extrema occur among eight range endpoints. The maximum norm
    is also attained at an endpoint (coordinatewise convex). Subtract eu*maxnorm
    for arbitrary dependent up error. Floating-point allowance is numerical only.
    All triangles use the same image-winding orientation; folds cannot relabel
    their own sign to pass.
    """
    rays, lo, hi, u = [np.asarray(x, float) for x in (rays, lower, upper, up)]
    if (rays.ndim != 3 or rays.shape[1:] != (3, 3) or lo.shape != rays.shape[:2]
            or hi.shape != lo.shape or u.shape != (3,) or not np.isfinite(up_error)
            or up_error < 0 or not all(np.isfinite(x).all() for x in (rays, lo, hi, u))
            or np.any(lo <= 0) or np.any(lo > hi) or abs(np.linalg.norm(u)-1) > 1e-12):
        raise SensorContractError('finite ordered three-ray intervals and unit up required')
    smallest = np.full(len(lo), np.inf); biggest = np.full(len(lo), -np.inf)
    maximum_norm = np.zeros(len(lo)); arithmetic = np.zeros(len(lo))
    for choices in product((False, True), repeat=3):
        z = np.where(np.asarray(choices)[None], hi, lo)
        p = z[..., None]*rays
        left, right = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]
        n = np.cross(left, right); projection = n@u
        smallest = np.minimum(smallest, projection); biggest = np.maximum(biggest, projection)
        maximum_norm = np.maximum(maximum_norm, np.linalg.norm(n, axis=1))
        # Account for subtraction of nearby points, not just the small final cross.
        scale = np.linalg.norm(p, axis=2).max(axis=1)
        arithmetic = np.maximum(arithmetic, 1e-12+512*np.finfo(float).eps*scale**2)
    allowance = up_error*maximum_norm + arithmetic*(1+up_error)
    return dict(projected_normal_lower=smallest-allowance,
                projected_normal_upper=biggest+allowance, maximum_normal_norm=maximum_norm)


class BoundedDepthSurface:
    """Immutable source observation with one query-independent reference plane.

    No raw tiny-triangle-normal agreement requirement. Instead, all admitted
    vertices are bounded in a common tube and all admitted triangles retain
    their image winding for every allowed range/up endpoint. No plane is fitted
    through missing data, and no different patch is tried after rejection.
    """
    def __init__(self, depth, valid, up, *, range_error_m, surface_tube_m, up_error):
        numbers = np.asarray([range_error_m, surface_tube_m, up_error], float)
        if (numbers.shape != (3,) or not np.isfinite(numbers).all() or np.any(numbers < 0)
                or surface_tube_m <= 0 or up_error >= 1):
            raise SensorContractError('explicit finite nonnegative range/up and positive tube bounds required')
        if np.asarray(depth).dtype != np.float32:
            raise SensorContractError('actual float32 range representation required')
        self.frame = PreparedFloorFrame(depth, valid, up)
        if abs(np.linalg.norm(self.frame._up)-1) > 1e-12:
            raise SensorContractError('unit observed up required')
        self.range_error_m = float(range_error_m)
        self.surface_tube_m = float(surface_tube_m)
        self.up_error = float(up_error)
        self.seed = MeasuredPlaneHypothesis.from_frame(self.frame).cell_for(self.frame)
        self.status = 'NO_ELIGIBLE_SEED'; self.anchor = None; self.normal = None
        self._mask = np.zeros((479, 639), bool); self._prefix = np.zeros((480, 640), np.int64)
        self.diagnostics = {}
        self._prepare()
        for a in (self._mask, self._prefix, self.anchor, self.normal):
            if a is not None: a.flags.writeable = False

    def _prepare(self):
        if self.seed is None: return
        r, c = self.seed; k = 8
        if r-k < 0 or r+k >= 479 or c-k < 0 or c+k >= 639:
            self.status = 'FIXED_PATCH_OUTSIDE_IMAGE'; return
        f = self.frame; d, valid, up = f._depth, f._valid, f._up
        rows, cols = np.mgrid[:480, :640]; T = np.asarray(BODY_FROM_OPTICAL)
        rays = np.stack(((cols+.5-320)/FOCAL, (rows+.5-240)/FOCAL, np.ones_like(cols)), axis=-1)@T[:3, :3].T
        z = d.astype(float)
        previous = np.nextafter(d, np.float32(-np.inf)).astype(float)
        following = np.nextafter(d, np.float32(np.inf)).astype(float)
        epsilon = self.range_error_m + .5*np.maximum(z-previous, following-z) + 1e-12
        lower, upper = z-epsilon, z+epsilon
        usable = valid & (lower >= .2) & (upper <= 5.)
        patch = np.s_[r-k:r+k+2, c-k:c+k+2]
        if not usable[patch].all():
            self.status = 'FIXED_PATCH_MISSING_OR_RANGE_CROSSING'; return
        points = z[..., None]*rays + T[:3, 3]
        patch_points = points[patch].reshape(-1, 3)
        a = patch_points.mean(axis=0)
        _, s, vectors = np.linalg.svd(patch_points-a, full_matrices=False)
        if s[1]/np.sqrt(len(patch_points)) < .005 or s[1] < .02*s[0]:
            self.status = 'FIXED_PATCH_ILL_CONDITIONED'; return
        n = vectors[-1]
        if n@up < 0: n = -n
        if n@up-self.up_error < .97 or n@(T[:3, 3]-a) <= self.surface_tube_m:
            self.status = 'FIXED_PATCH_ORIENTATION'; return
        # Reference plane is deterministic; tube bounds ALL possible true vertices.
        vertex_bound = np.abs((points-a)@n) + np.abs(rays@n)*epsilon
        vertex_bound += 1e-12+128*np.finfo(float).eps*(np.linalg.norm(points-a, axis=2)+epsilon)
        height_upper = (points@up + np.abs(rays@up)*epsilon
                        + self.up_error*(np.linalg.norm(points, axis=2)+epsilon*np.linalg.norm(rays, axis=2)))
        pixels = usable & (vertex_bound <= self.surface_tube_m) & (height_upper < -.15)
        self.diagnostics = dict(maximum_patch_vertex_residual_bound_m=float(vertex_bound[patch].max()),
            smaller_patch_tangent_rms_m=float(s[1]/np.sqrt(len(patch_points))))
        if not pixels[patch].all():
            self.status = 'FIXED_PATCH_OUTSIDE_SURFACE_TUBE'; return
        candidate = pixels[:-1, :-1] & pixels[:-1, 1:] & pixels[1:, 1:] & pixels[1:, :-1]
        cells = np.argwhere(candidate)
        corners = cells[:, None]+np.array([[0, 0], [0, 1], [1, 1], [1, 0]])[None]
        yy, xx = corners[..., 0], corners[..., 1]
        ray = rays[yy, xx]; lo = lower[yy, xx]; hi = upper[yy, xx]
        # Choose ONE winding from the nominal seed's first triangle, not per cell.
        seed_points = points[r:r+2, c:c+2]
        cross = np.cross(seed_points[0, 1]-seed_points[0, 0], seed_points[1, 1]-seed_points[0, 0])
        sign = 1 if cross@up > 0 else -1
        good = np.ones(len(cells), bool)
        for triangle in ((0, 1, 2), (0, 2, 3)):
            indices = list(triangle)
            bounds = triangle_orientation_bounds(ray[:, indices], lo[:, indices], hi[:, indices], up, self.up_error)
            good &= ((bounds['projected_normal_lower'] > 0) if sign > 0
                     else (bounds['projected_normal_upper'] < 0))
        candidate[tuple(cells.T)] = good
        self.diagnostics['tube_consistent_cells'] = int(len(cells))
        self.diagnostics['robust_nonfolding_cells'] = int(good.sum())
        if not candidate[r-k:r+k+1, c-k:c+k+1].all():
            self.status = 'FIXED_PATCH_ORIENTATION_UNRESOLVED'; return
        self.anchor, self.normal = a, n
        self._mask = candidate
        self._prefix[1:, 1:] = (~candidate).cumsum(axis=0).cumsum(axis=1)
        self.status = 'BOUNDED_MEASURED_SURFACE_AVAILABLE'

    def query(self, geometry, joints, *, rotation_observation_from_body,
              translation_observation_from_body, point_error_by_shape, backend='prefix'):
        if self.status != 'BOUNDED_MEASURED_SURFACE_AVAILABLE':
            raise SensorContractError('bounded surface unavailable; retain unknown')
        R, t = np.asarray(rotation_observation_from_body, float), np.asarray(translation_observation_from_body, float)
        if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all()
                or not np.allclose(R.T@R, np.eye(3), atol=1e-12, rtol=0)
                or abs(np.linalg.det(R)-1) > 1e-12 or backend not in ('prefix', 'reference')):
            raise SensorContractError('proper finite relative transform and supported backend required')
        a, n = self.anchor, self.normal
        # Normal error is ZERO because this is a fixed reference, not an unknown
        # physical plane. Surface variation is bounded by tube_m everywhere used.
        gaps = primitive_floor_gap_bounds(geometry, joints, (a-t)@R, n@R, normal_error=0.,
            plane_offset_error=self.surface_tube_m, point_error_by_shape=point_error_by_shape)
        shapes = geometry.supports(joints, R)['shapes']
        low, high = [np.array([s[key] for s in shapes])+t for key in ('lower', 'upper')]
        radius = np.array([point_error_by_shape[s['shape_id']] for s in shapes])[:, None]
        bound = coupled_footprint_enclosure(low-radius, high+radius, np.broadcast_to(a, low.shape),
            np.broadcast_to(n, low.shape), self.frame._up, normal_error=0., up_error=self.up_error,
            plane_offset_error=self.surface_tube_m)
        count = len(shapes)
        index = dict(up=self.frame._up, invalid_cell_prefix=self._prefix)
        coverage = query_floor_bounds(index, bound['footprint_lower_m'], bound['footprint_upper_m'],
            np.zeros(count), np.zeros(count), np.ones(count, bool))
        if backend == 'reference':
            for i in np.flatnonzero(coverage['projection_within_observed_camera']):
                x0, y0 = coverage['lower_cells_xy'][i]; x1, y1 = coverage['upper_cells_xy'][i]
                invalid = int((~self._mask[y0:y1+1, x0:x1+1]).sum())
                coverage['invalid_cells'][i] = invalid
                coverage['all_projected_cells_observed_ground'][i] = invalid == 0
        return dict(gap_bounds=gaps,
            floor_coverage={s['shape_id']: bool(coverage['all_projected_cells_observed_ground'][i])
                            for i, s in enumerate(shapes)},
            lower_cells_xy=coverage['lower_cells_xy'], upper_cells_xy=coverage['upper_cells_xy'],
            invalid_cells=coverage['invalid_cells'], covered_cells=coverage['covered_cells'],
            surface_tube_m=self.surface_tube_m, range_error_hypothesis_m=self.range_error_m,
            reference_plane_not_assumed_true_surface=True, piecewise_linear_surface_assumed=True,
            exact_camera_geometry_assumed=True, true_surface_identity_established=False,
            supplied_error_bounds_validated=False, non_floor_clearance_established=False,
            contact_permitted=False, navigation_qualified=False, future_gait_qualified=False)
