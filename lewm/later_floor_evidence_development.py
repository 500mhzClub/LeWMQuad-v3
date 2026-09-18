"""Resolve old ambiguous nominal-foot bounds using later complete measured views.

All source returns remain in their original partitions. This is conditional
geometry under the fixed observed pose/floor model, not calibrated uncertainty
or a physical support certificate.
"""
from copy import deepcopy
import re
import numpy as np
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.joint_visual_surface_memory_development import MAX_FRAMES, VOXEL_M
from lewm.observed_floor_waypoint_development import CELL_M
from lewm.causal_depth_observation_development import CALIBRATION_ID as PRIMARY_CALIBRATION
from lewm.auxiliary_downward45_depth_observation_development import CALIBRATION_ID as AUXILIARY_CALIBRATION
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose


def map_box(bounds, rotation):
    """Outward arithmetic enclosure for the given binary64 box and matrix."""
    box = np.asarray(bounds, float); B = proper(rotation)
    if (box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] > box[1])
            or np.any(np.abs(box) > 50.)):
        raise ValueError('finite bounded ordered measured enclosure required')
    lower = np.zeros(3); upper = np.zeros(3)
    for j in range(3):
        a = B[:, j]*box[0, j]; b = B[:, j]*box[1, j]
        low = np.nextafter(np.minimum(a, b), -np.inf)
        high = np.nextafter(np.maximum(a, b), np.inf)
        lower = np.nextafter(lower+low, -np.inf)
        upper = np.nextafter(upper+high, np.inf)
    return np.stack((lower, upper))


def floor_squares(bounds_map, floor_height):
    box = np.asarray(bounds_map, float)
    if (box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] > box[1])
            or not np.isfinite(floor_height)):
        raise ValueError('finite map enclosure and fixed floor height required')
    # Inward thresholds avoid enlarging the declared 10-mm band by rounding.
    inside = bool(box[0, 2] >= np.nextafter(floor_height-.01, np.inf)
        and box[1, 2] <= np.nextafter(floor_height+.01, -np.inf))
    if not inside: return None, 'enclosure_outside_original_floor_height_band'
    if np.any(box[:, :2] < -5.) or np.any(box[:, :2] >= 5.):
        return None, 'outside_observed_floor_grid_domain'
    a = np.floor(np.nextafter(box[0, :2]/CELL_M, -np.inf)).astype(np.int64)
    b = np.floor(np.nextafter(box[1, :2]/CELL_M, np.inf)).astype(np.int64)
    if np.any(a < -100) or np.any(b >= 100): return None, 'outside_observed_floor_grid_domain'
    if int(np.prod(b-a+1)) > 16: return None, 'enclosure_spans_more_than_sixteen_floor_squares'
    return [(i, j) for i in range(int(a[0]), int(b[0])+1)
        for j in range(int(a[1]), int(b[1])+1)], None


class LaterFloorEvidence:
    def __init__(self):
        self.frame = -1; self.now_ns = None
        self.rotation = None; self.floor_height = None
        self._records = []; self._cell_observations = {}

    def record_pair(self, frame, now_ns, map_from_initial, floor_height, observations):
        B = proper(map_from_initial)
        if (type(frame) is not int or frame != self.frame+1 or frame >= MAX_FRAMES
                or type(now_ns) is not int or now_ns != 1_500_000_000+100_000_000*frame
                or not np.isfinite(floor_height) or len(observations) != 2):
            raise ValueError('bounded consecutive paired floor observations required')
        if self.rotation is not None and (not np.array_equal(B, self.rotation) or floor_height != self.floor_height):
            raise ValueError('unchanged initial map and floor hypothesis required')
        checked = []
        for camera, observation in zip(('primary', 'auxiliary'), observations, strict=True):
            witness = deepcopy(observation['witness'])
            if (witness['camera'] != camera or type(witness['frame']) is not int or witness['frame'] != frame
                    or type(witness['measured_ns']) is not int or witness['measured_ns'] != now_ns
                    or witness['calibration_id'] != (PRIMARY_CALIBRATION if camera == 'primary' else AUXILIARY_CALIBRATION)):
                raise ValueError('paired current camera witnesses required')
            for name in ('rgb_sha256', 'depth_sha256'):
                if not isinstance(witness[name], str) or re.fullmatch('[0-9a-f]{64}', witness[name]) is None:
                    raise ValueError('exact camera byte identities required')
            proper(witness['rotation_map_from_reference'])
            p = np.asarray(witness['position_map_m'], float)
            if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError('finite observed camera reference position required')
            cells = np.asarray(observation['cells'])
            if (cells.ndim != 2 or cells.shape[1:] != (2,) or cells.dtype.kind not in 'iu'
                    or len(cells) > 40000 or np.any(cells < -100) or np.any(cells >= 100)):
                raise ValueError('bounded actually covered floor cells required')
            keys = {tuple(map(int, row)) for row in cells}
            if len(keys) != len(cells): raise ValueError('unique measured floor cells required')
            checked.append((witness, keys))
        primary, auxiliary = checked[0][0], checked[1][0]
        Q, q = reference_pose(primary['rotation_map_from_reference'], primary['position_map_m'])
        if (primary['rgb_sha256'] != auxiliary['rgb_sha256']
                or not np.array_equal(Q, auxiliary['rotation_map_from_reference'])
                or not np.array_equal(q, auxiliary['position_map_m'])):
            raise ValueError('same primary image and original auxiliary reference transform required')
        # Validate both cameras before publishing any evidence for this frame.
        if self.rotation is None:
            self.rotation = B.copy(); self.floor_height = float(floor_height)
        for witness, cells in checked:
            bit = 1 << len(self._records)
            self._records.append(witness)
            for cell in cells:
                self._cell_observations[cell] = self._cell_observations.get(cell, 0) | bit
        self.frame = frame; self.now_ns = now_ns

    def resolve(self, bounds_initial, latest_sample_frame, *, now_ns):
        if self.now_ns != now_ns or self.rotation is None:
            raise ValueError('current completed paired floor evidence required')
        if type(latest_sample_frame) is not int or not 0 <= latest_sample_frame <= self.frame:
            raise ValueError('actual nonfuture last ambiguous sample frame required')
        bounds = map_box(bounds_initial, self.rotation)
        cells, reason = floor_squares(bounds, self.floor_height)
        receipt = dict(resolved=False, map_bounds_m=bounds.tolist(), original_floor_height_band_m=.01,
            latest_ambiguous_sample_frame=latest_sample_frame,
            covered_map_cells=None if cells is None else [list(k) for k in cells],
            later_single_view=None, reason=reason, original_return_classification_changed=False,
            pose_and_flat_floor_hypotheses_required=True, calibrated_pose_or_sensor_bounds=False,
            ground_support_approved=False, unobserved_space_certified=False)
        if cells is None: return receipt
        candidates = self._cell_observations.get(cells[0], 0)
        for cell in cells[1:]: candidates &= self._cell_observations.get(cell, 0)
        # Both records at the source frame are excluded. A newer ambiguous
        # sample automatically invalidates every older resolution witness.
        candidates >>= 2*(latest_sample_frame+1)
        if not candidates:
            return receipt | dict(reason='no_strictly_later_single_view_covering_entire_enclosure')
        record_id = 2*(latest_sample_frame+1)+(candidates & -candidates).bit_length()-1
        witness = self._records[record_id]
        assert latest_sample_frame < witness['frame'] <= self.frame and witness['measured_ns'] <= now_ns
        return receipt | dict(resolved=True, reason='later_complete_measured_floor_view',
            later_single_view=deepcopy(witness))


def sphere_hits(index, center, radius):
    """Enumerate every hit using the original bound-index sphere calculation."""
    c = np.asarray(center, float)
    if c.shape != (3,) or not np.isfinite(c).all() or not np.isfinite(radius) or not 0 < radius <= 1.:
        raise ValueError('bounded finite sphere required')
    low, high = c-radius, c+radius
    a = np.ceil(low/VOXEL_M).astype(int)-1; b = np.floor(high/VOXEL_M).astype(int)
    if int(np.prod(b-a+1)) <= len(index.cells):
        keys = ((x,y,z) for x in range(a[0],b[0]+1) for y in range(a[1],b[1]+1) for z in range(a[2],b[2]+1))
    else:
        keys = (k for k in index.cells if all(a[i] <= k[i] <= b[i] for i in range(3)))
    hits = [k for k in keys if k in index.bounds and (index.bounds[k][0] <= high).all()
        and (index.bounds[k][1] >= low).all()]
    return sorted(k for k in hits if np.linalg.norm(np.maximum(
        np.maximum(index.bounds[k][0]-c, c-index.bounds[k][1]), 0.)) <= radius+1e-12)


def resolve_sphere_query(index, original, center, radius, ledger, *, now_ns):
    """Retain original index; filter only bounds with complete later evidence."""
    expected = index.intersect_sphere(center, radius)
    if original != expected: raise ValueError('exact original ambiguous sphere query required')
    hits = sphere_hits(index, center, radius)
    if len(hits) != original['intersecting_voxels']:
        raise ValueError('every original intersecting ambiguous enclosure required')
    resolved = []; remaining = []; examined = []
    for key in hits:
        evidence = ledger.resolve(index.bounds[key], index.latest_frames[key], now_ns=now_ns)
        row = dict(cell=list(key), bounds_initial_body_m=index.bounds[key].tolist(),
            contributing_samples=index.sample_counts[key], first_sample_witness=deepcopy(index.cells[key]),
            **evidence)
        examined.append(row)
        (resolved if evidence['resolved'] else remaining).append(key)
    key = remaining[0] if remaining else None
    revised = original | dict(status='POSSIBLE_MEASURED_SAMPLE_BOUNDS_INTERSECTION' if remaining else 'UNKNOWN',
        intersecting_voxels=len(remaining), first_cell=None if key is None else list(key),
        witness=None if key is None else deepcopy(index.cells[key]),
        first_bounds_m=None if key is None else index.bounds[key].tolist(),
        first_bounds_sample_count=None if key is None else index.sample_counts[key],
        first_bounds_latest_frame=None if key is None else index.latest_frames[key])
    assert len(resolved)+len(remaining) == len(hits)
    return revised, dict(original_intersections=len(hits), resolved_intersections=len(resolved),
        remaining_intersections=len(remaining), enclosures=examined,
        all_original_returns_and_partitions_retained=True, unresolved_contacts_exempted=False,
        resolution_scope='ambiguous_nominal_foot_returns_with_strictly_later_complete_floor_evidence')
