"""Fixed measured plane hypotheses with per-primitive geometric exemptions.

Development-only current-posture evidence. A selected upward plane is NOT
traversable ground. No simulator labels, contact permission, future motion,
calibrated sensor error, or navigation qualification is supplied here.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.primitive_obstacle_memory_development import (
    PrimitiveObstacleMemory, non_floor_box_evidence, combine_primitive_views)
from lewm.uncertain_ray_memory_development import transport_radius


@dataclass(frozen=True)
class MeasuredPlaneHypothesis:
    """One query-independent identity; no dominant-floor or semantic assertion."""

    depth_sha256: str
    up: tuple
    cell_rc: tuple | None
    policy: str = 'median-row-major-eligible-cell-v1'

    @classmethod
    def from_frame(cls, frame):
        if not isinstance(frame, PreparedFloorFrame):
            raise SensorContractError('immutable prepared observation required')
        cells = np.argwhere(frame._index['ground_cells'])
        cell = tuple(int(x) for x in cells[len(cells) // 2]) if len(cells) else None
        return cls(frame.depth_sha256, tuple(float(x) for x in frame._up), cell)

    def cell_for(self, frame):
        if (self.depth_sha256 != frame.depth_sha256 or self.up != tuple(frame._up)
                or self.policy != 'median-row-major-eligible-cell-v1'):
            raise SensorContractError('measured plane hypothesis observation identity mismatch')
        return self.cell_rc


def relation_gated_non_floor_evidence(frame, lower, upper, *, plane, shape_ids,
                                      gap_lower, gap_upper, **kwargs):
    """Exempt observed family returns only for physically compatible shapes.

    Separation uses the unpadded physical primitive and supplied uncertainty.
    An exact foot straddling the plane remains only a contact CANDIDATE.
    A non-foot intersection, definitely submerged shape, or absent plane keeps
    all near returns as possible obstacles, even in an incomplete view.
    Whole footprint coverage is a separate prerequisite for positive evidence.
    """
    ids = tuple(shape_ids)
    low, high = np.asarray(lower, float), np.asarray(upper, float)
    gl, gh = np.asarray(gap_lower, float), np.asarray(gap_upper, float)
    if (not ids or len(set(ids)) != len(ids) or low.shape != (len(ids), 3)
            or high.shape != low.shape or gl.shape != (len(ids),) or gh.shape != gl.shape
            or not np.isfinite([gl, gh]).all() or np.any(gl > gh)):
        raise SensorContractError('aligned unique shapes and finite ordered physical gap bounds required')
    feet = np.array([sid in FOOT_SHAPES for sid in ids])
    exemption = (plane is not None) & ((gl > 0) | (feet & (gl <= 0) & (gh >= 0)))
    result = {}
    for exempt in (False, True):
        selected = np.flatnonzero(exemption == exempt)
        if not len(selected):
            continue
        part = non_floor_box_evidence(frame, low[selected], high[selected],
                                      plane=plane if exempt else None, **kwargs)
        for key, value in part.items():
            if isinstance(value, np.ndarray):
                if key not in result:
                    result[key] = np.empty(len(ids), dtype=value.dtype)
                result[key][selected] = value
            else:
                result[key] = value
    return result | {'plane_exemption': exemption}


class MeasuredPlaneObstacleMemory(PrimitiveObstacleMemory):
    """Retain one immutable hypothesis with each inherited causal depth frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hypotheses = {}

    def observe(self, policy, depth, relative, *, now_ns):
        try:
            row = super().observe(policy, depth, relative, now_ns=now_ns)
            keep = {ns: h for ns, h in self._hypotheses.items() if ns in self._prepared}
            keep[now_ns] = MeasuredPlaneHypothesis.from_frame(self._prepared[now_ns])
            self._hypotheses = keep
            return row | {'hypothesis_cell_rc': keep[now_ns].cell_rc,
                          'hypothesis_policy': keep[now_ns].policy}
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True; self._prepared.clear(); self._hypotheses.clear(); self._joints = None
            raise SensorContractError('measured plane observation unavailable; stop') from error

    def query_current_primitives(self, *, now_ns, beam_backend=None):
        if self.failed or self._last_ns is None or now_ns != self._last_ns:
            raise SensorContractError('fresh active primitive observation required')
        backend = self._beam_backend if beam_backend is None else beam_backend
        if backend not in ('reference', 'compiled'): raise SensorContractError('supported beam backend required')
        rays = self._rays
        boxes = self._geometry.supports(self._joints, np.eye(3))['shapes']
        ids = tuple(r['shape_id'] for r in boxes)
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
            hypothesis = self._hypotheses[frame['measured_ns']]
            cell = hypothesis.cell_for(prepared)
            gap_low = np.zeros(len(ids)); gap_high = np.zeros(len(ids))
            covered = np.zeros(len(ids), bool); separated = covered.copy(); candidate = covered.copy(); penetration = covered.copy()
            plane = None
            if cell is not None:
                floor = prepared.query(self._geometry, self._joints, cell,
                                       rotation_observation_from_body=R, translation_observation_from_body=t,
                                       point_error_by_shape=point_errors,
                                       floor_backend='cached' if backend == 'compiled' else 'reference', **self._errors)
                for i, gap in enumerate(floor['gap_bounds']['primitives']):
                    if gap['shape_id'] != ids[i]:
                        raise SensorContractError('floor primitive ordering changed')
                    gap_low[i] = gap['minimum_gap_lower_m']; gap_high[i] = gap['minimum_gap_upper_m']
                    covered[i] = floor['floor_coverage'][ids[i]]
                    separated[i] = gap['minimum_gap_lower_m'] > 0
                    candidate[i] = (ids[i] in FOOT_SHAPES and gap['minimum_gap_lower_m'] <= 0 <= gap['minimum_gap_upper_m'])
                    penetration[i] = covered[i] and gap['minimum_gap_upper_m'] < 0
                # Same plane family for floor footprints and ray-return roles.
                # Preserve the exact observed plane identity; no unnecessary
                # body-frame round trip before the shared mask/cache lookup.
                plane = (floor['plane_anchor_observation_m'], floor['plane_normal_observation'])
            other = relation_gated_non_floor_evidence(prepared, low, high, plane=plane,
                                            shape_ids=ids, gap_lower=gap_low, gap_upper=gap_high,
                                            normal_error=self._errors['normal_error'],
                                            plane_offset_error=self._errors['plane_offset_error'],
                                            range_error_m=self._range_error, backend=backend)
            views.append({'shape_ids': ids, 'clear': covered & separated & other['non_floor_clearance'],
                          'contact_candidate': covered & candidate & other['non_floor_clearance'],
                          'non_floor_conflict': other['non_floor_conflict'], 'floor_penetration': penetration})
            bindings.append({'measured_ns': frame['measured_ns'], 'depth_sha256': prepared.depth_sha256,
                             'seed_observed': cell is not None, 'hypothesis_cell_rc': cell,
                             'hypothesis_policy': hypothesis.policy,
                             'floor_covered_primitives': int(covered.sum()),
                             'plane_exempt_primitives': int(other['plane_exemption'].sum()),
                             'non_floor_clear_primitives': int(other['non_floor_clearance'].sum()),
                             'non_floor_conflict_primitives': int(other['non_floor_conflict'].sum()),
                             'scanned_pixels': int(other['scanned_pixels'].sum())})
        return combine_primitive_views(ids, views) | {'measured_ns': now_ns, 'identity': rays.identity,
                    'views': bindings, 'current_measured_posture_only': True, 'static_scene_assumed': True,
                    'supplied_error_bounds_validated': False}
