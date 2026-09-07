"""Gyro-rotated point-to-plane relative translation from actual depth.

Development estimator, not calibrated odometry. Weak translation directions
remain unobserved; neither commands nor true pose fill a missing component.
Correspondences assume a static scene and at most0.2 m frame-to-frame motion.
"""
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree

from lewm.causal_depth_observation_development import body_points
from lewm.causal_sensor_state import SensorContractError
from lewm.depth_local_surfaces_development import LocalSurfaceHistory
from lewm.fast_gyro_development import FastRelativeOrientation


def surface_cloud(depth, policy, *, now_ns):
    cloud = body_points(depth, policy, now_ns=now_ns, stride=8)
    p, valid = cloud['points_body_m'], cloud['valid']
    centre = p[1:-1, 1:-1]
    left, right, top, bottom = p[1:-1, :-2], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]
    good = (valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
            & valid[:-2, 1:-1] & valid[2:, 1:-1])
    for neighbour in (left, right, top, bottom):
        good &= np.linalg.norm(neighbour-centre, axis=-1) <= .25
    normal = np.cross(right-left, bottom-top)
    length = np.linalg.norm(normal, axis=-1)
    good &= np.isfinite(length) & (length > 1e-7)
    normal = np.divide(normal, length[..., None], out=np.zeros_like(normal), where=length[..., None] > 1e-7)
    # Opposing one-sided tangents must also be locally planar; edge normals
    # computed across a depth jump must not become a new motion constraint.
    for neighbour in (left, right, top, bottom):
        good &= np.abs(np.sum((neighbour-centre)*normal, axis=-1)) <= .003
    return centre[good].copy(), normal[good].copy()


def register_translation(previous, current, rotation_previous_from_current):
    """Return current body-origin displacement expressed in previous body axes."""
    target, normals = previous
    source, source_normals = current
    rotation = np.asarray(rotation_previous_from_current, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(rotation)-1) > 1e-7):
        raise SensorContractError('proper observed relative rotation required')
    for points, ns in (previous, current):
        if (points.ndim != 2 or points.shape[1:] != (3,) or ns.shape != points.shape
                or not np.isfinite(points).all() or not np.isfinite(ns).all()
                or not np.allclose(np.linalg.norm(ns, axis=1), 1., atol=1e-6, rtol=0)):
            raise SensorContractError('finite unit-normal surface observations required')
    base = {'translation_previous_body_m': None, 'observable_projection_previous_body_m': None,
        'rank': 0, 'normal_eigenvalues': [0., 0., 0.], 'weak_directions_previous_body': np.eye(3).tolist(),
        'matched_points': 0, 'residual_rms_m': None, 'converged': False,
        'status': 'INSUFFICIENT_SURFACE_SUPPORT', 'hardware_calibrated': False,
        'calibrated_uncertainty': False, 'assumption': 'static_scene_local_correspondences_and_gyro_rotation'}
    if min(len(target), len(source)) < 100: return base
    source = source@rotation.T; source_normals = source_normals@rotation.T
    tree = cKDTree(target)
    translation = np.zeros(3)
    for iteration in range(15):
        distance, indices = tree.query(source+translation, distance_upper_bound=.2)
        good = np.isfinite(distance)
        clipped = np.minimum(indices, len(target)-1)
        good &= np.abs(np.sum(source_normals*normals[clipped], axis=1)) >= .95
        if good.sum() < 100: return {**base, 'matched_points': int(good.sum())}
        a = normals[indices[good]]
        b = np.sum(a*(target[indices[good]]-source[good]), axis=1)
        residual = a@translation-b
        weights = np.minimum(1., .01/np.maximum(np.abs(residual), 1e-12))
        gram = (a.T*weights)@a/weights.sum()
        values, vectors = np.linalg.eigh(gram)
        constrained = values > max(.005, .01*values[-1])
        basis = vectors[:, constrained]
        updated = basis@np.linalg.lstsq((a@basis)*np.sqrt(weights[:, None]),
                                       b*np.sqrt(weights), rcond=None)[0]
        change = np.linalg.norm(updated-translation)
        translation = updated
        if change < 1e-5: break
    residual = a@translation-b
    rms = float(np.sqrt(np.mean(residual**2)))
    rank = int(constrained.sum())
    accepted = change < 1e-5 and rms <= .01 and np.linalg.norm(translation) <= .15
    status = ('OBSERVED_TRANSLATION' if rank == 3 else 'PARTIALLY_OBSERVED_TRANSLATION') if accepted else 'REGISTRATION_REJECTED'
    return {**base, 'translation_previous_body_m': translation.tolist() if accepted and rank == 3 else None,
        'observable_projection_previous_body_m': translation.tolist() if accepted else None,
        'rank': rank, 'normal_eigenvalues': np.maximum(values, 0.).tolist(),
        'weak_directions_previous_body': vectors[:, ~constrained].T.tolist(),
        'matched_points': int(good.sum()), 'residual_rms_m': rms,
        'converged': bool(change < 1e-5), 'iterations': iteration+1, 'status': status}


class DepthRelativeState:
    """Live/replay observer; a missing component invalidates cumulative position."""
    def __init__(self):
        self.surfaces = LocalSurfaceHistory()
        self.orientation = FastRelativeOrientation()
        self.previous = self.previous_rotation = None
        self.position = np.zeros(3)
        self.failed = False

    def observe(self, policy, depth, fast, *, now_ns):
        if self.failed: raise SensorContractError('relative depth state fault latched')
        try:
            surface = self.surfaces.observe(depth, policy, now_ns=now_ns)
            attitude = (self.orientation.begin(policy, fast, now_ns=now_ns) if self.previous is None
                        else self.orientation.step(policy, fast, now_ns=now_ns))
            rotation = np.asarray(attitude['rotation_initial_body_from_current_body'])
            current = surface_cloud(depth, policy, now_ns=now_ns)
            motion = None
            if self.previous is not None:
                motion = register_translation(self.previous, current, self.previous_rotation.T@rotation)
                delta = motion['translation_previous_body_m']
                if delta is None: self.position = None
                elif self.position is not None: self.position += self.previous_rotation@np.asarray(delta)
            self.previous, self.previous_rotation = current, rotation.copy()
            return {'measured_ns': now_ns, 'local_surfaces': surface, 'relative_orientation': attitude,
                'motion': motion, 'surface_points': len(current[0]),
                'position_initial_body_m': self.position.tolist() if self.position is not None else None,
                'position_is_observed_anchor': motion is None,
                'arrival_verified': False, 'turn_clearance_qualified': False,
                'scope': 'development relative observation, not a calibrated motion or clearance certificate'}
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('relative depth observation failed; do not use stale state') from error
