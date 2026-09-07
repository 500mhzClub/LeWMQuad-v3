"""Exact registration with distance-certified nearest-neighbour reuse.

Recomputes weights, eigenspaces and least-squares updates for every perturbed
input. Nominal neighbours are reused only when a conservative competitor bound
proves the same nearest point; ambiguous queries run the ordinary tree search.
This is a computational optimization, not a sensor-uncertainty certificate.
"""
import numpy as np
from scipy.spatial import cKDTree
from copy import deepcopy
import hashlib

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import body_points, CausalDepthHistory
from lewm.depth_local_surfaces_development import LocalSurfaceHistory, observe_local_surfaces
from lewm.depth_relative_motion_development import DepthRelativeState
from lewm.correlated_moment_sensitivity_development import RawRgbdMomentSensitivity


def _ids(ids, count):
    ids = np.asarray(ids)
    if ids.shape != (count,) or ids.dtype.kind not in 'iu' or np.any(ids < 0):
        raise SensorContractError('unique explicit point-lineage IDs required')
    # Actual pixel-lineage IDs are already strictly increasing. Keep the full
    # uniqueness check for reordered inputs; unsigned subtraction is not safe.
    if not np.all(ids[1:] > ids[:-1]) and len(np.unique(ids)) != count:
        raise SensorContractError('unique explicit point-lineage IDs required')
    return ids


class NearestQueryReference:
    """Immutable nominal targets/queries and nearest-competitor distances."""

    def __init__(self, target, queries, target_ids, query_ids, tree):
        target, queries = np.asarray(target), np.asarray(queries)
        if (target.ndim != 2 or target.shape[1:] != (3,) or not len(target)
                or queries.ndim != 2 or queries.shape[1:] != (3,)
                or not np.isfinite(target).all() or not np.isfinite(queries).all()
                or not np.array_equal(tree.data, target)):
            raise SensorContractError('finite nonempty targets and matching search tree required')
        self.target, self.queries = target.copy(), queries.copy()
        self.target_ids = _ids(target_ids, len(target)).copy()
        self.query_ids = _ids(query_ids, len(queries)).copy()
        distances, indices = tree.query(queries, k=2)
        if not np.isfinite(distances[:, 0]).all():
            raise SensorContractError('representable nearest distances required')
        self.nearest = indices[:, 0].copy()
        self.competitor_distance = distances[:, 1].copy()
        for array in (self.target, self.queries, self.target_ids, self.query_ids,
                      self.nearest, self.competitor_distance):
            array.flags.writeable = False

    def query(self, target, queries, target_ids, query_ids, tree):
        target, queries = np.asarray(target), np.asarray(queries)
        if (target.ndim != 2 or target.shape[1:] != (3,) or not len(target)
                or queries.ndim != 2 or queries.shape[1:] != (3,)
                or not np.isfinite(target).all() or not np.isfinite(queries).all()
                or not np.array_equal(tree.data, target)):
            raise SensorContractError('finite targets/queries and matching search tree required')
        target_ids, query_ids = _ids(target_ids, len(target)), _ids(query_ids, len(queries))
        if not np.array_equal(target_ids, self.target_ids) or not np.array_equal(query_ids, self.query_ids):
            distance, index = tree.query(queries, distance_upper_bound=.2)
            return distance, index, {'certified': 0, 'searched': len(queries), 'lineage_fallback': True}
        displacement = np.linalg.norm(queries - self.queries, axis=1)
        target_bound = float(np.linalg.norm(target - self.target, axis=1).max())
        nearest_distance = np.linalg.norm(queries - target[self.nearest], axis=1)
        # For EVERY competing target j, triangle inequality gives
        # d'_j >= d_second - ||q'-q|| - max_j ||t'_j-t_j||.
        # The strict numerical guard deliberately sends ties back to SciPy.
        lower = self.competitor_distance - displacement - target_bound
        coordinate_scale = max(float(np.max(np.abs(a), initial=0.))
                               for a in (target, queries, self.target, self.queries))
        numerical_guard = 1e-10 + 64 * np.finfo(float).eps * coordinate_scale
        certified = nearest_distance + numerical_guard < lower
        distance = np.where(nearest_distance < .2, nearest_distance, np.inf)
        index = np.where(nearest_distance < .2, self.nearest, len(target)).copy()
        ambiguous = np.flatnonzero(~certified)
        if len(ambiguous):
            distance[ambiguous], index[ambiguous] = tree.query(queries[ambiguous], distance_upper_bound=.2)
        return distance, index, {'certified': int(certified.sum()), 'searched': len(ambiguous), 'lineage_fallback': False}


def surface_cloud_with_ids(depth, policy, *, now_ns):
    """Same cloud construction as the frozen observer, with pixel lineage."""
    cloud = body_points(depth, policy, now_ns=now_ns, stride=8)
    p, valid = cloud['points_body_m'], cloud['valid']
    centre = p[1:-1, 1:-1]
    left, right, top, bottom = p[1:-1, :-2], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]
    good = (valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
            & valid[:-2, 1:-1] & valid[2:, 1:-1])
    for neighbour in (left, right, top, bottom):
        good &= np.linalg.norm(neighbour - centre, axis=-1) <= .25
    normal = np.cross(right - left, bottom - top)
    length = np.linalg.norm(normal, axis=-1)
    good &= np.isfinite(length) & (length > 1e-7)
    normal = np.divide(normal, length[..., None], out=np.zeros_like(normal), where=length[..., None] > 1e-7)
    for neighbour in (left, right, top, bottom):
        good &= np.abs(np.sum((neighbour - centre) * normal, axis=-1)) <= .003
    row, col = np.meshgrid(cloud['rows'][1:-1], cloud['columns'][1:-1], indexing='ij')
    return (centre[good].copy(), normal[good].copy()), (row * 640 + col)[good].copy()


def register_translation_reuse(previous, current, rotation_previous_from_current,
                               previous_ids, current_ids, *, references=None):
    """Frozen registration equations with exact certified/query fallback.

With references=None, collect nominal query references at each iteration.
With references supplied, recompute the perturbed estimator, reusing only
certified nearest indices. Changed lineage, iteration count or nearest matches
does not force the nominal rank, convergence or output onto the perturbed run.
"""
    target, normals = previous
    source, source_normals = current
    rotation = np.asarray(rotation_previous_from_current, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(rotation) - 1) > 1e-7):
        raise SensorContractError('proper observed relative rotation required')
    for points, ns in (previous, current):
        if (points.ndim != 2 or points.shape[1:] != (3,) or ns.shape != points.shape
                or not np.isfinite(points).all() or not np.isfinite(ns).all()
                or not np.allclose(np.linalg.norm(ns, axis=1), 1., atol=1e-6, rtol=0)):
            raise SensorContractError('finite unit-normal surface observations required')
    previous_ids, current_ids = _ids(previous_ids, len(target)), _ids(current_ids, len(source))
    collected = []
    stats = {'certified_queries': 0, 'searched_queries': 0, 'lineage_fallback_iterations': 0,
             'reference_queries': 0}
    base = {'translation_previous_body_m': None, 'observable_projection_previous_body_m': None,
        'rank': 0, 'normal_eigenvalues': [0., 0., 0.], 'weak_directions_previous_body': np.eye(3).tolist(),
        'matched_points': 0, 'residual_rms_m': None, 'converged': False,
        'status': 'INSUFFICIENT_SURFACE_SUPPORT', 'hardware_calibrated': False,
        'calibrated_uncertainty': False, 'assumption': 'static_scene_local_correspondences_and_gyro_rotation'}
    if min(len(target), len(source)) < 100: return base, collected, stats
    source = source @ rotation.T; source_normals = source_normals @ rotation.T
    tree = cKDTree(target)
    translation = np.zeros(3)
    for iteration in range(15):
        queries = source + translation
        if references is not None and iteration < len(references):
            distance, indices, row = references[iteration].query(target, queries, previous_ids, current_ids, tree)
            stats['certified_queries'] += row['certified']; stats['searched_queries'] += row['searched']
            stats['lineage_fallback_iterations'] += row['lineage_fallback']
        else:
            distance, indices = tree.query(queries, distance_upper_bound=.2)
            stats['searched_queries'] += len(queries)
        if references is None:
            collected.append(NearestQueryReference(target, queries, previous_ids, current_ids, tree))
            stats['reference_queries'] += len(queries)
        good = np.isfinite(distance)
        clipped = np.minimum(indices, len(target) - 1)
        good &= np.abs(np.sum(source_normals * normals[clipped], axis=1)) >= .95
        if good.sum() < 100: return {**base, 'matched_points': int(good.sum())}, collected, stats
        a = normals[indices[good]]
        b = np.sum(a * (target[indices[good]] - source[good]), axis=1)
        residual = a @ translation - b
        weights = np.minimum(1., .01 / np.maximum(np.abs(residual), 1e-12))
        gram = (a.T * weights) @ a / weights.sum()
        values, vectors = np.linalg.eigh(gram)
        constrained = values > max(.005, .01 * values[-1])
        basis = vectors[:, constrained]
        updated = basis @ np.linalg.lstsq((a @ basis) * np.sqrt(weights[:, None]),
                                          b * np.sqrt(weights), rcond=None)[0]
        change = np.linalg.norm(updated - translation)
        translation = updated
        if change < 1e-5: break
    residual = a @ translation - b
    rms = float(np.sqrt(np.mean(residual ** 2)))
    rank = int(constrained.sum())
    accepted = change < 1e-5 and rms <= .01 and np.linalg.norm(translation) <= .15
    status = ('OBSERVED_TRANSLATION' if rank == 3 else 'PARTIALLY_OBSERVED_TRANSLATION') if accepted else 'REGISTRATION_REJECTED'
    result = {**base, 'translation_previous_body_m': translation.tolist() if accepted and rank == 3 else None,
        'observable_projection_previous_body_m': translation.tolist() if accepted else None,
        'rank': rank, 'normal_eigenvalues': np.maximum(values, 0.).tolist(),
        'weak_directions_previous_body': vectors[:, ~constrained].T.tolist(),
        'matched_points': int(good.sum()), 'residual_rms_m': rms,
        'converged': bool(change < 1e-5), 'iterations': iteration + 1, 'status': status}
    return result, collected, stats


class _SharedObservations:
    def __init__(self):
        self.now = None; self.ready = False

    def begin(self, now):
        self.now = now; self.ready = False; self.references = None
        self.surfaces = {}; self.clouds = {}
        self.surface_hits = self.cloud_hits = 0


class _SharedSurfaceHistory(LocalSurfaceHistory):
    def __init__(self, group):
        super().__init__(); self.group = group

    def observe(self, depth, policy, *, now_ns):
        # Each observer still validates and advances its own causal history.
        self.depth_history.push(depth, policy, now_ns=now_ns)
        key = (tuple(depth['identity']), depth['calibration_id'], depth['rgb_sha256'],
               hashlib.sha256(depth['depth_m'].tobytes() + depth['valid'].tobytes()).hexdigest())
        if key in self.group.surfaces:
            result = deepcopy(self.group.surfaces[key]); self.group.surface_hits += 1
        else:
            result = observe_local_surfaces(depth, policy, now_ns=now_ns)
            self.group.surfaces[key] = deepcopy(result)
        self.frames = [*self.frames[-3:], deepcopy(result)]
        return result


class CertifiedDepthRelativeState(DepthRelativeState):
    """Exact original output schema, with per-pair clocks and error guards."""

    def __init__(self, group, *, nominal):
        super().__init__()
        self.group, self.nominal = group, nominal
        self.surfaces = _SharedSurfaceHistory(group)
        self.previous_ids = None; self.statistics = {}; self.nominal_result = None

    def observe(self, policy, depth, fast, *, now_ns):
        if self.failed: raise SensorContractError('relative depth state fault latched')
        try:
            if self.nominal: self.group.begin(now_ns)
            elif now_ns != self.group.now or not self.group.ready:
                raise SensorContractError('completed co-timed nominal registration required')
            surface = self.surfaces.observe(depth, policy, now_ns=now_ns)
            attitude = (self.orientation.begin(policy, fast, now_ns=now_ns) if self.previous is None
                        else self.orientation.step(policy, fast, now_ns=now_ns))
            rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
            key = surface['depth_sha256']
            if key in self.group.clouds:
                current, current_ids = self.group.clouds[key]; self.group.cloud_hits += 1
            else:
                current, current_ids = surface_cloud_with_ids(depth, policy, now_ns=now_ns)
                for array in (*current, current_ids): array.flags.writeable = False
                self.group.clouds[key] = current, current_ids
            motion = None; self.statistics = {}
            if self.previous is not None:
                motion, collected, self.statistics = register_translation_reuse(
                    self.previous, current, self.previous_rotation.T @ rotation,
                    self.previous_ids, current_ids,
                    references=None if self.nominal else self.group.references)
                if self.nominal: self.group.references = collected
                delta = motion['translation_previous_body_m']
                if delta is None: self.position = None
                elif self.position is not None: self.position += self.previous_rotation @ np.asarray(delta)
            self.previous, self.previous_rotation, self.previous_ids = current, rotation.copy(), current_ids
            if self.nominal: self.group.ready = True
            result = {'measured_ns': now_ns, 'local_surfaces': surface, 'relative_orientation': attitude,
                'motion': motion, 'surface_points': len(current[0]),
                'position_initial_body_m': self.position.tolist() if self.position is not None else None,
                'position_is_observed_anchor': motion is None,
                'arrival_verified': False, 'turn_clearance_qualified': False,
                'scope': 'development relative observation, not a calibrated motion or clearance certificate'}
            if self.nominal: self.nominal_result = deepcopy(result)
            return result
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            self.nominal_result = None
            raise SensorContractError('relative depth observation failed; do not use stale state') from error


class CertifiedRgbdMomentSensitivity(RawRgbdMomentSensitivity):
    """Same paired sensor-error reference with exact shared-work acceleration."""

    def __init__(self, source_ids, *, difference_step=1e-3):
        super().__init__(source_ids, difference_step=difference_step)
        self.group = _SharedObservations()
        for index, model in enumerate(self.models):
            model.depth = CertifiedDepthRelativeState(self.group, nominal=index == 0)

    def _observe(self, policy, depth, fast, loadings):
        result = super()._observe(policy, depth, fast, loadings)
        return result | {'registration_query_accounting': [deepcopy(m.depth.statistics) for m in self.models],
                         'shared_surface_hits': self.group.surface_hits,
                         'shared_cloud_hits': self.group.cloud_hits,
                         'uncertainty_calibration_unchanged': True}


class _RegistrationOnlyDepthHistory:
    """Validated metadata for a perturbation branch, NOT a wall/portal report.

Moment fusion reads only these identity/time/hash fields. Its translation still
comes from the complete dense surface-cloud registration. Never substitute this
object for a nominal navigation surface observation.
"""

    def __init__(self):
        self.depth_history = CausalDepthHistory()

    def observe(self, depth, policy, *, now_ns):
        self.depth_history.push(depth, policy, now_ns=now_ns)
        return {'measured_ns': depth['measured_ns'], 'identity': tuple(depth['identity']),
                'rgb_sha256': depth['rgb_sha256'],
                'depth_sha256': hashlib.sha256(depth['depth_m'].tobytes() + depth['valid'].tobytes()).hexdigest()}


class LeanCertifiedRgbdMomentSensitivity(CertifiedRgbdMomentSensitivity):
    """Elide unused pair-only wall reports, preserve nominal navigation report.

The original fusion equations do not consume wall segments or portal summaries.
All raw sensor validation, depth history, surface-cloud selection, registration,
gyro integration, gravity and fusion still run in each perturbation branch.
"""

    def __init__(self, source_ids, *, difference_step=1e-3):
        super().__init__(source_ids, difference_step=difference_step)
        for model in self.models[1:]: model.depth.surfaces = _RegistrationOnlyDepthHistory()

    def _observe(self, policy, depth, fast, loadings):
        result = super()._observe(policy, depth, fast, loadings)
        return result | {'omitted_unused_pair_wall_reports': len(self.models) - 1,
                         'nominal_full_surface_report_preserved': True}

    def nominal_depth_state(self, *, now_ns):
        """Full current depth evidence, without executing another estimator."""
        nominal = self.models[0].depth
        if (self.failed or self.current is None or nominal.failed or nominal.nominal_result is None
                or now_ns != self.current['measured_ns'] or now_ns != nominal.nominal_result['measured_ns']):
            raise SensorContractError('fresh valid nominal depth observation required')
        return deepcopy(nominal.nominal_result)
