"""Observed floor-patch/pose relations with shared sensor perturbations.

Offline diagnostics only. A fitted infinite plane is never treated as observed
floor coverage. Each relation must project along measured up into an actual
adjacent-pixel triangle; missing or non-ground pixels remain unknown. Neither
finite perturbation pairs nor their variance certify a continuous error set.
"""
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_moment_sensitivity_development import RawRgbdMomentSensitivity


def patch_relation(points, patches, valid, up):
    """Signed height and observed footprint of adjacent TL/TR/BR/BL patches.

Two measured triangles, not an extrapolated plane, define the footprint.
Height is measured along up; the fourth-point 3-mm planarity and 0.97 normal
tests retain the existing ground-evidence tolerances. This is still a local
surface interpolation assumption, not proof about subpixel holes or contacts.
"""
    q, p, up = [np.asarray(x, dtype=float) for x in (points, patches, up)]
    valid = np.asarray(valid)
    if (q.ndim != 2 or q.shape[1:] != (3,) or p.shape != (len(q), 4, 3)
            or valid.shape != (len(q), 4) or valid.dtype != bool or up.shape != (3,)
            or not all(np.isfinite(x).all() for x in (q, p, up))
            or abs(np.linalg.norm(up) - 1.) > 1e-6):
        raise SensorContractError('finite points, adjacent patches, validity and unit up required')
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
    length = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, length[:, None], out=np.zeros_like(normal), where=length[:, None] > 1e-10)
    eligible = (valid.all(axis=1) & (length > 1e-10) & (np.abs(normal @ up) >= .97)
                & (np.abs(np.sum((p[:, 2] - p[:, 0]) * normal, axis=1)) <= .003)
                & ((p @ up) < -.15).all(axis=1))
    height = np.full(len(q), np.nan)
    inside = np.zeros(len(q), bool)
    for triangle in ((0, 1, 2), (0, 2, 3)):
        a, b, c = [p[:, i] for i in triangle]
        normal_t = np.cross(b - a, c - a)
        norm_t = np.linalg.norm(normal_t, axis=1)
        good = eligible & (norm_t > 1e-10) & (np.abs(normal_t @ up) >= .97 * norm_t)
        ids = np.flatnonzero(good)
        if not len(ids): continue
        matrix = np.stack((b[ids] - a[ids], c[ids] - a[ids], np.broadcast_to(up, (len(ids), 3))), axis=2)
        uvh = np.linalg.solve(matrix, (q[ids] - a[ids])[..., None])[..., 0]
        contained = (uvh[:, 0] >= -1e-8) & (uvh[:, 1] >= -1e-8) & (uvh[:, :2].sum(axis=1) <= 1. + 1e-8)
        # Retain the first plane's height outside its footprint solely to locate
        # the projected footprint. Outside values are not observed support.
        replace = ~np.isfinite(height[ids]) | (contained & ~inside[ids])
        height[ids[replace]] = uvh[replace, 2]
        inside[ids] |= contained
    return {'height_m': height, 'observed_footprint': inside, 'ground_patch_eligible': eligible,
            'ground_support_approved': False, 'continuous_surface_qualified': False}


def _project(points):
    transform = np.asarray(BODY_FROM_OPTICAL)
    optical = (points - transform[:3, 3]) @ transform[:3, :3]
    valid = (optical[:, 2] >= .2) & (optical[:, 2] <= 5.)
    denominator = np.maximum(optical[:, 2], 1e-12)
    rc = np.stack((FOCAL * optical[:, 1] / denominator + 239.5,
                   FOCAL * optical[:, 0] / denominator + 319.5), axis=1)
    valid &= (rc[:, 0] >= 0) & (rc[:, 0] < 479) & (rc[:, 1] >= 0) & (rc[:, 1] < 639)
    return np.floor(np.clip(rc, -1, 640)).astype(int), valid


def _sample(depth, valid, cells):
    corners = cells[:, None, :] + np.array([[0, 0], [0, 1], [1, 1], [1, 0]])[None]
    in_image = ((corners[:, :, 0] >= 0) & (corners[:, :, 0] < 480)
                & (corners[:, :, 1] >= 0) & (corners[:, :, 1] < 640))
    row = np.clip(corners[:, :, 0], 0, 479); col = np.clip(corners[:, :, 1], 0, 639)
    z = depth[row, col]
    optical = np.stack((z * (col + .5 - 320) / FOCAL, z * (row + .5 - 240) / FOCAL, z), axis=2)
    transform = np.asarray(BODY_FROM_OPTICAL)
    return optical @ transform[:3, :3].T + transform[:3, 3], valid[row, col] & in_image


def locate_floor_patches(depth, valid, points, up):
    """Find measured triangles under points within the existing 6-cm band.

Follow at most four local plane projections to adjacent observed patches. No
wall/missing patch is crossed to infer floor, and no search expands the height
band. Failure to locate a patch is unknown, not proof that floor is absent.
"""
    points = np.asarray(points, dtype=float); up = np.asarray(up, dtype=float)
    depth, valid = np.asarray(depth), np.asarray(valid)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or depth.shape != (480, 640) or valid.shape != depth.shape or valid.dtype != bool
            or not np.isfinite(depth).all() or np.any(depth[~valid] != 0.)
            or np.any((depth[valid] < .2) | (depth[valid] > 5.))):
        raise SensorContractError('finite queries and measured depth/validity grid required')
    cells, active = _project(points)
    found = np.zeros(len(points), bool); heights = np.full(len(points), np.nan)
    for _ in range(4):
        patch, mask = _sample(depth, valid, cells)
        relation = patch_relation(points, patch, mask, up)
        active &= relation['ground_patch_eligible'] & np.isfinite(relation['height_m'])
        active &= np.abs(relation['height_m']) <= .06
        accepted = active & relation['observed_footprint']
        heights[accepted] = relation['height_m'][accepted]
        found |= accepted
        moving = active & ~found
        if not moving.any(): break
        ids = np.flatnonzero(moving)
        feet = points[ids] - relation['height_m'][ids, None] * up
        next_cells, visible = _project(feet)
        cells[ids] = next_cells
        active[ids] &= visible
    return {'cells_rc': cells, 'observed_footprint': found, 'height_m': heights,
            'ground_support_approved': False}


class PairedFloorEvidence:
    """Raw sensor-pair reference plus bounded stored raw floor observations."""

    def __init__(self, source_ids, *, difference_step=1e-3):
        self.observer = RawRgbdMomentSensitivity(source_ids, difference_step=difference_step)
        self.current = None
        self.retained = {}
        self.failed = False

    def observe(self, policy, depth, fast, loadings):
        if self.failed: raise SensorContractError('paired floor evidence fault latched')
        try:
            result = self.observer.observe(policy, depth, fast, loadings)
            poses = []
            for model in self.observer.models:
                integrator = model.integrator
                poses.append({'position': integrator.position.copy(), 'rotation': integrator.rotation.copy(),
                              'up': integrator.rotation.T @ (integrator.gravity / 9.81)})
            self.current = {'measured_ns': depth['measured_ns'], 'poses': poses,
                            'depth': depth['depth_m'].copy(), 'valid': depth['valid'].copy(),
                            'depth_loadings': np.asarray(loadings['depth_m'], dtype=float).copy()}
            for key in ('depth', 'valid', 'depth_loadings'): self.current[key].flags.writeable = False
            return result
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True; self.current = None
            raise SensorContractError('paired floor history invalid') from error

    def retain(self, label):
        if (self.failed or self.current is None or type(label) is not str or not label
                or label in self.retained or len(self.retained) >= 64):
            raise SensorContractError('active observation and fresh label within 64-view limit required')
        # observe() replaces, never mutates, this privately owned snapshot.
        self.retained[label] = self.current

    def _patch(self, frame, index, cells):
        depth = frame['depth']
        if index == 0: return _sample(depth, frame['valid'], cells)
        source, sign = (index - 1) // 2, (1. if index % 2 else -1.)
        # Only materialize tapped range values, not a whole perturbed image.
        corners = cells[:, None, :] + np.array([[0, 0], [0, 1], [1, 1], [1, 0]])[None]
        rows = np.clip(corners[:, :, 0], 0, 479); cols = np.clip(corners[:, :, 1], 0, 639)
        z = (depth[rows, cols] + sign * self.observer.step * frame['depth_loadings'][rows, cols, source]).astype(depth.dtype)
        transform = np.asarray(BODY_FROM_OPTICAL)
        optical = np.stack((z * (cols + .5 - 320) / FOCAL, z * (rows + .5 - 240) / FOCAL, z), axis=2)
        mask = ((corners[:, :, 0] >= 0) & (corners[:, :, 0] < 480)
                & (corners[:, :, 1] >= 0) & (corners[:, :, 1] < 640) & frame['valid'][rows, cols])
        return optical @ transform[:3, :3].T + transform[:3, 3], mask

    def query(self, label, points_body, ground_roles, *, now_ns):
        if (self.failed or self.current is None or label not in self.retained
                or now_ns != self.current['measured_ns']):
            raise SensorContractError('fresh active paired current/stored observations required')
        points = np.asarray(points_body, dtype=float); roles = np.asarray(ground_roles)
        if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
                or roles.shape != (len(points),) or roles.dtype != bool):
            raise SensorContractError('finite queries and explicit ground roles required')
        old = self.retained[label]
        transported = []
        for current_pose, stored_pose in zip(self.current['poses'], old['poses'], strict=True):
            transported.append((points @ current_pose['rotation'].T + current_pose['position']
                                - stored_pose['position']) @ stored_pose['rotation'])
        nominal = locate_floor_patches(old['depth'], old['valid'], transported[0], old['poses'][0]['up'])
        nominal_found = nominal['observed_footprint'] & roles
        cells = nominal['cells_rc']
        joint, pose_only, floor_only = [], [], []
        joint_observed, split_observed = [], []
        base_patch, base_mask = self._patch(old, 0, cells)
        for index, (point, stored_pose) in enumerate(zip(transported, old['poses'], strict=True)):
            patch, mask = self._patch(old, index, cells)
            both = patch_relation(point, patch, mask, stored_pose['up'])
            just_pose = patch_relation(point, base_patch, base_mask, old['poses'][0]['up'])
            just_floor = patch_relation(transported[0], patch, mask, stored_pose['up'])
            joint.append(both['height_m']); pose_only.append(just_pose['height_m']); floor_only.append(just_floor['height_m'])
            joint_observed.append(both['observed_footprint'])
            split_observed.append(just_pose['observed_footprint'] & just_floor['observed_footprint'])
        usable = nominal_found & np.asarray(joint_observed).all(axis=0)
        comparable = usable & np.asarray(split_observed).all(axis=0)
        factors = []
        for values, supported in zip((joint, pose_only, floor_only), (usable, comparable, comparable), strict=True):
            array = np.asarray(values)
            factor = ((array[1::2] - array[2::2]) / (2 * self.observer.step)).T
            factor[~supported] = np.nan
            factors.append(factor)
        return {'nominal_height_m': np.where(nominal_found, nominal['height_m'], np.nan),
                'nominal_observed_footprint': nominal_found, 'cells_rc': cells,
                'paired_footprints_observed': usable, 'split_comparison_supported': comparable,
                'joint_height_factor_m': factors[0], 'pose_only_height_factor_m': factors[1],
                'floor_only_height_factor_m': factors[2],
                'joint_height_variance_m2': np.sum(factors[0] ** 2, axis=1),
                'incorrect_independent_height_variance_m2': np.sum(factors[1] ** 2 + factors[2] ** 2, axis=1),
                'source_ids': self.observer.source_ids, 'difference_step': self.observer.step,
                'current_measured_ns': now_ns, 'stored_measured_ns': old['measured_ns'],
                'sensor_model_calibrated': False, 'linearization_validated': False,
                'whole_error_envelope_covered': False, 'ground_support_approved': False,
                'navigation_qualified': False, 'hardware_qualified': False}
