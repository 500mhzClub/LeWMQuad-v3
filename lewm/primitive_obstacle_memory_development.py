"""Observation-bound physical-primitive floor/non-floor evidence consumer.

Current posture only. No contact permission, future-gait certificate, calibrated
sensor bounds or navigation qualification. Native scene/contact labels are not
inputs. Historical evidence assumes a static environment.
"""
from itertools import product

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_floor_evidence_development import locate_floor_patches
from lewm.floor_footprint_bounds_development import projected_footprint_rectangles
from lewm.primitive_floor_observation_development import (
    PreparedFloorFrame, plane_family_cells, cached_plane_family_mask)
from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.primitive_beam_kernel_development import reduce_primitive_beams, warm_primitive_beam_kernel
from lewm.uncertain_ray_memory_development import FusedRayEvidenceMemory, transport_radius


def pixel_cell_box_depth_intervals(optical_lower, optical_upper, cells_rc):
    """Exact box intersection with the beam between four pixel-centre rays.

    At positive optical depth z, a cell's x interval is [a*z,b*z]. It intersects
    a box iff a*z<=Xhi and -b*z<=-Xlo, likewise for y. These four linear
    inequalities intersect the box's z interval. Unlike rectangle x global
    depth, this preserves angular/depth coupling. No ray sampling is used.
    """
    low, high = np.asarray(optical_lower, float), np.asarray(optical_upper, float)
    cells = np.asarray(cells_rc)
    if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
            or np.any(low > high) or cells.ndim != 2 or cells.shape[1:] != (2,)
            or cells.dtype.kind not in 'iu' or np.any(cells < 0)
            or np.any(cells[:, 0] >= 480) or np.any(cells[:, 1] >= 640)):
        raise SensorContractError('finite optical box and in-image integer cells required')
    left = (cells[:, 1] - 319.5) / FOCAL; right = (cells[:, 1] + 1 - 319.5) / FOCAL
    top = (cells[:, 0] - 239.5) / FOCAL; bottom = (cells[:, 0] + 1 - 239.5) / FOCAL
    near = np.full(len(cells), max(0., low[2])); far = np.full(len(cells), high[2])
    feasible = np.ones(len(cells), bool)
    for coefficient, limit in ((left, high[0]), (-right, -low[0]), (top, high[1]), (-bottom, -low[1])):
        positive, negative = coefficient > 0, coefficient < 0
        far[positive] = np.minimum(far[positive], limit / coefficient[positive])
        near[negative] = np.maximum(near[negative], limit / coefficient[negative])
        feasible &= (coefficient != 0) | (limit >= 0)
    # Outward arithmetic allowance, not sensor or calibration uncertainty.
    margin = 1e-10 + 128 * np.finfo(float).eps * np.maximum(np.abs(near), np.abs(far))
    near -= margin; far += margin
    return near, far, feasible & (near <= far) & (far > 0)


def non_floor_box_evidence(frame, lower, upper, *, plane=None, normal_error,
                           plane_offset_error, range_error_m, backend='reference'):
    """Every intersecting pixel cell must be floor-family or beyond the box.

    Boxes already contain the intended geometry padding and pose errors. Range
    error enlarges each return interval. Near non-floor returns veto clearance,
    even in incomplete views; nearer occluders remain unknown. A floor-family
    mask is NOT standalone free space: the consumer must separately establish
    physical floor separation or leave an exact foot contact as a candidate.
    """
    if not isinstance(frame, PreparedFloorFrame):
        raise SensorContractError('immutable prepared measured frame required')
    if backend not in ('reference', 'compiled'):
        raise SensorContractError('explicit supported beam backend required')
    low, high = np.asarray(lower, float), np.asarray(upper, float)
    errors = np.asarray([normal_error, plane_offset_error, range_error_m], float)
    if (errors.shape != (3,) or not np.isfinite(errors).all() or np.any(errors < 0)):
        raise SensorContractError('explicit finite nonnegative plane/range errors required')
    range_error_m = float(range_error_m)
    rect = projected_footprint_rectangles(low, high, np.zeros(len(low)), np.zeros(len(low)), frame._up)
    if plane is not None:
        a, n = [np.asarray(x, float) for x in plane]
        if (a.shape != (3,) or n.shape != (3,) or not np.isfinite([a, n]).all()
                or abs(np.linalg.norm(n) - 1) > 1e-12 or n @ frame._up <= 0):
            raise SensorContractError('finite oriented unit floor plane required')
    corners = np.asarray(list(product((0, 1), repeat=3)), dtype=bool)
    vertices = np.where(corners[None], high[:, None], low[:, None])
    transform = np.asarray(BODY_FROM_OPTICAL)
    optical = (vertices - transform[:3, 3]) @ transform[:3, :3]
    near, far = optical[:, :, 2].min(axis=1), optical[:, :, 2].max(axis=1)
    optical_low, optical_high = optical.min(axis=1), optical.max(axis=1)
    count = len(low)
    if backend == 'compiled':
        floor_mask = frame._empty_floor_mask
        eligible = ((far + range_error_m >= .2)
                    & (rect['projection_within_observed_camera'] | (far + range_error_m >= frame._minimum_valid_depth)))
        if plane is not None and eligible.any():
            floor_mask = cached_plane_family_mask(frame, a, n, normal_error=normal_error,
                                                   plane_offset_error=plane_offset_error)
        clear, conflict, scanned, floor_pixels = reduce_primitive_beams(
            frame._depth, frame._valid, floor_mask, optical_low, optical_high,
            rect['lower_cells_xy'], rect['upper_cells_xy'], rect['projection_within_observed_camera'],
            range_error_m, frame._minimum_valid_depth)
        return {'non_floor_clearance': clear, 'non_floor_conflict': conflict, 'unknown_or_blocked': ~clear,
                'scanned_pixels': scanned, 'floor_family_pixels': floor_pixels,
                'projection_within_observed_camera': rect['projection_within_observed_camera'],
                'floor_classification_calibrated': False, 'navigation_qualified': False}
    clear = np.zeros(count, bool); conflict = clear.copy(); unknown = clear.copy()
    scanned = np.zeros(count, np.int64); floor_pixels = scanned.copy()
    for i in range(count):
        x0, y0 = rect['lower_cells_xy'][i]; x1, y1 = rect['upper_cells_xy'][i]
        if near[i] <= 0 <= far[i]: x0, y0, x1, y1 = 0, 0, 639, 479
        # Last-row/column returns have no complete floor cell, but remain
        # obstacle evidence. Incomplete views must not discard those pixels.
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(639, x1), min(479, y1)
        complete = bool(rect['projection_within_observed_camera'][i])
        if x0 > x1 or y0 > y1 or far[i] + range_error_m < .2:
            unknown[i] = True
            continue
        if not complete and far[i] + range_error_m < frame._minimum_valid_depth:
            # Exact rejection of possible near returns using immutable global
            # depth minimum. This incomplete view remains UNKNOWN, never free.
            # In particular this avoids scanning whole-camera beams when an
            # old camera origin lies inside a now-nearby primitive box.
            unknown[i] = True
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        cells = np.stack((yy.ravel(), xx.ravel()), axis=1)
        entry, exit, intersects = pixel_cell_box_depth_intervals(optical_low[i], optical_high[i], cells)
        cells = cells[intersects]; entry, exit = entry[intersects], exit[intersects]
        if not len(cells):
            unknown[i] = True
            continue
        offsets = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        corners_rc = cells[:, None] + offsets
        inside = (corners_rc[:, :, 0] < 480) & (corners_rc[:, :, 1] < 640)
        rr = np.clip(corners_rc[:, :, 0], 0, 479); cc = np.clip(corners_rc[:, :, 1], 0, 639)
        values = frame._depth[rr, cc]; valid = frame._valid[rr, cc] & inside
        # Explicit float64 interval arithmetic, independent of NumPy's weak
        # Python-scalar promotion of float32 depth arrays.
        minimum = np.where(valid, values, np.inf).min(axis=1).astype(np.float64)
        maximum = np.where(valid, values, -np.inf).max(axis=1).astype(np.float64)
        floor = np.zeros(len(cells), bool) if plane is None else plane_family_cells(
            frame, cells, a, n, normal_error=normal_error, plane_offset_error=plane_offset_error)
        beyond = minimum - range_error_m > exit
        possible_near = (maximum + range_error_m >= entry) & (minimum - range_error_m <= exit)
        conflict[i] = bool(np.any(valid.any(axis=1) & ~floor & possible_near))
        clear[i] = bool(complete and np.all(valid.all(axis=1) & (floor | beyond)) and not conflict[i])
        unknown[i] = not clear[i]
        scanned[i] = len(cells); floor_pixels[i] = int(floor.sum())
    return {'non_floor_clearance': clear, 'non_floor_conflict': conflict, 'unknown_or_blocked': unknown,
            'scanned_pixels': scanned, 'floor_family_pixels': floor_pixels,
            'projection_within_observed_camera': rect['projection_within_observed_camera'],
            'floor_classification_calibrated': False, 'navigation_qualified': False}


def combine_primitive_views(shape_ids, views):
    """A complete coherent view may support; ANY retained contradiction vetoes."""
    ids = tuple(shape_ids)
    if not ids or len(set(ids)) != len(ids): raise SensorContractError('unique nonempty shape IDs required')
    clear = np.zeros(len(ids), bool); candidates = clear.copy(); conflict = clear.copy(); penetration = clear.copy()
    for row in views:
        for key in ('clear', 'contact_candidate', 'non_floor_conflict', 'floor_penetration'):
            array = np.asarray(row[key])
            if array.shape != (len(ids),) or array.dtype != bool:
                raise SensorContractError('view evidence must align exact primitive population')
        if tuple(row['shape_ids']) != ids: raise SensorContractError('view primitive identities changed')
        if np.any(row['contact_candidate'] & np.array([sid not in FOOT_SHAPES for sid in ids])):
            raise SensorContractError('only exact foot primitives may carry contact candidates')
        clear |= row['clear']; candidates |= row['contact_candidate']
        conflict |= row['non_floor_conflict']; penetration |= row['floor_penetration']
    veto = conflict | penetration
    return {'shape_ids': ids, 'conditional_clearance': clear & ~veto,
            'foot_contact_candidate': candidates & ~veto,
            'non_floor_conflict': conflict, 'floor_penetration': penetration,
            'all_primitives_conditionally_clear': bool((clear & ~veto).all()),
            'contact_permitted': False, 'future_gait_qualified': False, 'navigation_qualified': False}


class PrimitiveObstacleMemory:
    """Own current posture, causal fused poses and immutable retained depth frames."""

    def __init__(self, geometry, *, normal_error, up_error, plane_offset_error, range_error_m, beam_backend='reference'):
        errors = np.asarray([normal_error, up_error, plane_offset_error, range_error_m], float)
        if (not isinstance(geometry, ArticulatedCollisionGeometry) or errors.shape != (4,)
                or not np.isfinite(errors).all() or np.any(errors < 0) or normal_error >= 1):
            raise SensorContractError('verified geometry and explicit nonnegative error allowances required')
        self._geometry = geometry
        if beam_backend not in ('reference', 'compiled'): raise SensorContractError('supported beam backend required')
        self._beam_backend = beam_backend
        if beam_backend == 'compiled': warm_primitive_beam_kernel()
        self._errors = dict(normal_error=float(normal_error), up_error=float(up_error),
                            plane_offset_error=float(plane_offset_error))
        self._range_error = float(range_error_m)
        self._rays = FusedRayEvidenceMemory()
        self._prepared = {}; self._joints = None; self._last_ns = None; self.failed = False

    def observe(self, policy, depth, relative, *, now_ns):
        if self.failed: raise SensorContractError('primitive observation fault latched')
        try:
            if depth['measured_ns'] != now_ns: raise SensorContractError('current depth required')
            self._rays.observe(policy, depth, relative, now_ns=now_ns)
            joints = policy['sensor_state']['sensed']['joints']
            if joints['measured_ns'][-1] != now_ns or not joints['valid'][-1, :12].all():
                raise SensorContractError('current valid measured joint positions required')
            self._joints = joints['values'][-1, :12].copy(); self._joints.flags.writeable = False
            frame = self._rays.latest_frame
            prepared = PreparedFloorFrame(depth['depth_m'], depth['valid'], frame['evidence']['up'])
            if prepared.depth_sha256 != relative['local_surfaces']['depth_sha256']:
                raise SensorContractError('floor and nominal observer depth binding mismatch')
            keep = {f['measured_ns'] for f in self._rays.frames} | {now_ns}
            self._prepared = {ns: p for ns, p in self._prepared.items() if ns in keep}
            self._prepared[now_ns] = prepared; self._last_ns = now_ns
            return {'measured_ns': now_ns, 'retained_observations': len(self._prepared),
                    'depth_sha256': prepared.depth_sha256, 'navigation_qualified': False}
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True; self._prepared.clear(); self._joints = None
            raise SensorContractError('primitive observation unavailable; stop') from error

    def query_current_primitives(self, *, now_ns, beam_backend=None):
        if self.failed or self._last_ns is None or now_ns != self._last_ns:
            raise SensorContractError('fresh active primitive observation required')
        backend = self._beam_backend if beam_backend is None else beam_backend
        if backend not in ('reference', 'compiled'): raise SensorContractError('supported beam backend required')
        rays = self._rays
        boxes = self._geometry.supports(self._joints, np.eye(3))['shapes']
        ids = tuple(r['shape_id'] for r in boxes)
        foot_centres = np.asarray([r['center_body_m'] for r in boxes if r['shape_id'] in FOOT_SHAPES])
        frames = list(rays.frames)
        if frames[-1]['measured_ns'] != now_ns: frames.append(rays.latest_frame)
        corners = np.asarray(list(product((0, 1), repeat=3)), dtype=bool)
        views = []; bindings = []
        for frame in frames:
            prepared = self._prepared[frame['measured_ns']]
            R = frame['rotation'].T @ rays.rotation
            t = (rays.position - frame['position']) @ frame['rotation']
            point_errors = {r['shape_id']: float(transport_radius(np.where(corners, r['upper'], r['lower']),
                                                                 rays.latest_frame, frame).max()) for r in boxes}
            transformed = self._geometry.supports(self._joints, R)['shapes']
            error = np.array([point_errors[sid] for sid in ids])[:, None]
            # Isotropic 4-cm geometry padding, once, with explicit point errors.
            low = np.array([r['lower'] for r in transformed]) + t - error - .04
            high = np.array([r['upper'] for r in transformed]) + t + error + .04
            seed = locate_floor_patches(prepared._depth, prepared._valid, foot_centres @ R.T + t, prepared._up)
            found = np.flatnonzero(seed['observed_footprint'])
            covered = np.zeros(len(ids), bool); separated = covered.copy(); candidate = covered.copy(); penetration = covered.copy()
            plane = None
            if len(found):
                cell = seed['cells_rc'][found[0]]
                floor = prepared.query(self._geometry, self._joints, cell,
                                       rotation_observation_from_body=R, translation_observation_from_body=t,
                                       point_error_by_shape=point_errors,
                                       floor_backend='cached' if backend == 'compiled' else 'reference', **self._errors)
                for i, gap in enumerate(floor['gap_bounds']['primitives']):
                    covered[i] = floor['floor_coverage'][ids[i]]
                    separated[i] = gap['minimum_gap_lower_m'] > 0
                    candidate[i] = (ids[i] in FOOT_SHAPES and gap['minimum_gap_lower_m'] <= 0 <= gap['minimum_gap_upper_m'])
                    penetration[i] = covered[i] and gap['minimum_gap_upper_m'] < 0
                # Same plane family for floor footprints and ray-return roles.
                # Preserve the exact observed plane identity; no unnecessary
                # body-frame round trip before the shared mask/cache lookup.
                plane = (floor['plane_anchor_observation_m'], floor['plane_normal_observation'])
            other = non_floor_box_evidence(prepared, low, high, plane=plane,
                                            normal_error=self._errors['normal_error'],
                                            plane_offset_error=self._errors['plane_offset_error'],
                                            range_error_m=self._range_error, backend=backend)
            views.append({'shape_ids': ids, 'clear': covered & separated & other['non_floor_clearance'],
                          'contact_candidate': covered & candidate & other['non_floor_clearance'],
                          'non_floor_conflict': other['non_floor_conflict'], 'floor_penetration': penetration})
            bindings.append({'measured_ns': frame['measured_ns'], 'depth_sha256': prepared.depth_sha256,
                             'seed_observed': bool(len(found)), 'floor_covered_primitives': int(covered.sum()),
                             'non_floor_clear_primitives': int(other['non_floor_clearance'].sum()),
                             'non_floor_conflict_primitives': int(other['non_floor_conflict'].sum()),
                             'scanned_pixels': int(other['scanned_pixels'].sum())})
        return combine_primitive_views(ids, views) | {'measured_ns': now_ns, 'identity': rays.identity,
                    'views': bindings, 'current_measured_posture_only': True, 'static_scene_assumed': True,
                    'supplied_error_bounds_validated': False}
