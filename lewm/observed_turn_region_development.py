"""Sensor-derived candidate observation regions and nominal turn-volume evidence.

No scene geometry or true pose is accepted. Stored poses are the output of the
causal depth/gyro estimator. Sampling and modelling margins are development
assumptions, not hardware safety guarantees or future-gait certificates.
"""
from copy import deepcopy
import hashlib

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL, validate_depth
from lewm.causal_sensor_state import SensorContractError


def gravity_basis(up):
    up = np.asarray(up, dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1.) > 1e-6:
        raise SensorContractError('unit observed up direction required')
    forward = np.array([1., 0., 0.])-up[0]*up
    if np.linalg.norm(forward) < .8: raise SensorContractError('upright local control required')
    forward /= np.linalg.norm(forward)
    return np.stack((forward, np.cross(up, forward), up), axis=1)


def nominal_turn_volume(geometry, joints, up_body, *, spacing=.08, padding=.04):
    """Overapproximate yaw sweep of observed postures in gravity-aligned bands.

    Each primitive's transverse AABB gives a conservative radius for that
    posture. Future postures are NOT bounded by a finite measured history.
    Every retained sample is checked; resolution is explicitly reported.
    """
    if spacing != .08 or padding != .04: raise ValueError('fixed development envelope settings')
    basis = gravity_basis(up_body)
    q = np.asarray(joints)
    if q.ndim != 2 or q.shape[1] != 12 or not len(q) or not np.isfinite(q).all():
        raise SensorContractError('observed posture history required')
    shapes = []
    for posture in q:
        for shape in geometry.supports(posture, basis.T)['shapes']:
            lo, hi = np.asarray(shape['lower']), np.asarray(shape['upper'])
            radius = float(np.linalg.norm(np.maximum(np.abs(lo[:2]), np.abs(hi[:2])))+padding)
            shapes.append((lo[2]-padding, hi[2]+padding, radius,
                           'calf' in shape['link'].lower() or 'foot' in shape['link'].lower()))
    bottom, top = min(s[0] for s in shapes), max(s[1] for s in shapes)
    levels = np.linspace(bottom, top, int(np.ceil((top-bottom)/spacing))+1)
    points, support, bands = [], [], []
    for z in levels:
        active = [s for s in shapes if s[0]-spacing/2 <= z <= s[1]+spacing/2]
        if not active: continue
        radius = max(s[2] for s in active)
        allow_support = bool(z < -.15 and all(s[3] for s in active))
        axis = np.linspace(-radius, radius, int(np.ceil(2*radius/spacing))+1)
        x, y = np.meshgrid(axis, axis)
        selected = x*x+y*y <= radius*radius
        xy = np.stack((x[selected], y[selected]), axis=1)
        # Include the circular boundary as well as the interior grid.
        theta = np.linspace(0., 2*np.pi, int(np.ceil(2*np.pi*radius/spacing)), endpoint=False)
        xy = np.vstack((xy, radius*np.stack((np.cos(theta), np.sin(theta)), axis=1)))
        points.extend(np.column_stack((xy, np.full(len(xy), z)))@basis.T)
        support.extend([allow_support]*len(xy))
        bands.append({'height_m': float(z), 'radius_m': radius, 'ground_support_allowed': allow_support})
    return {'points_body_m': np.asarray(points), 'ground_support_allowed': np.asarray(support, bool),
            'bands': bands, 'spacing_m': spacing, 'padding_m': padding,
            'maximum_radius_m': max(b['radius_m'] for b in bands),
            'future_gait_qualified': False, 'continuous_volume_qualified': False}


def observed_corners(surface):
    """Intersect adjacent nonparallel observed supports, not arbitrary line ends."""
    segments = surface['surface_segments']; corners = []
    for left, right in zip(segments[:-1], segments[1:]):
        a = np.array([left['normal_body_xy'], right['normal_body_xy']])
        if abs(np.linalg.det(a)) < .8: continue
        columns = np.asarray(surface['sampled_columns'])
        between = (columns >= left['last_column']) & (columns <= right['first_column'])
        if (left['last_column'] not in columns or right['first_column'] not in columns
                or not np.asarray(surface['valid_columns'])[between].all()): continue
        points = np.asarray([p for p, keep in zip(surface['sampled_points_body_xy_m'], between, strict=True) if keep])
        distances = np.abs(points@a.T-np.array([left['offset_body_m'], right['offset_body_m']]))
        if np.any(distances.min(axis=1) > .02): continue
        point = np.linalg.solve(a, [left['offset_body_m'], right['offset_body_m']])
        dl = min(np.linalg.norm(point-p) for p in np.asarray(left['endpoints_body_xy_m']))
        dr = min(np.linalg.norm(point-p) for p in np.asarray(right['endpoints_body_xy_m']))
        if max(dl, dr) > .12: continue
        forward = left if abs(a[0, 0]) >= .8 else right
        side = right if forward is left else left
        side_sign = np.sign(point[1])
        beyond_x = np.asarray(side['endpoints_body_xy_m'])[:, 0]-point[0]
        beyond_y = side_sign*(np.asarray(forward['endpoints_body_xy_m'])[:, 1]-point[1])
        far = bool(beyond_x.max() > .05 and beyond_x.min() >= -.12)
        near = bool(beyond_x.min() < -.05 and beyond_x.max() <= .12)
        exterior = bool(abs(forward['normal_body_xy'][0]) >= .8 and abs(side['normal_body_xy'][1]) >= .8
                        and (far or near) and beyond_y.max() > .05 and beyond_y.min() >= -.12)
        corners.append({'point_body_xy_m': point.tolist(),
            'columns': [left['last_column'], right['first_column']],
            'support_gaps_m': [float(dl), float(dr)],
            'observed_ns': surface['measured_ns'],
            'exterior_side_corner': exterior, 'side': 'left' if side_sign > 0 else 'right',
            'longitudinal_role': 'far_boundary' if far else 'near_boundary' if near else 'unresolved',
            'opening_extent_verified': False})
    return corners


class RayEvidenceMemory:
    """Bounded local static-scene ray evidence; invalid localization fails closed."""
    def __init__(self):
        self.frames = []
        self.last_ns = self.identity = self.up_initial = None
        self.position = self.rotation = self.latest_surface = None
        self.latest_frame = None
        self.failed = False

    def observe(self, policy, depth, relative, *, now_ns):
        if self.failed: raise SensorContractError('ray evidence fault latched')
        try:
            validate_depth(depth, policy, now_ns=now_ns)
            expected = {'measured_ns', 'local_surfaces', 'relative_orientation', 'motion', 'surface_points',
                'position_initial_body_m', 'position_is_observed_anchor', 'arrival_verified',
                'turn_clearance_qualified', 'scope'}
            if set(relative) != expected or relative['measured_ns'] != now_ns:
                raise SensorContractError('exact causal relative-estimator output required')
            surface = relative['local_surfaces']
            if (surface['rgb_sha256'] != depth['rgb_sha256'] or surface['measured_ns'] != now_ns
                    or tuple(surface['identity']) != tuple(depth['identity'])
                    or surface['depth_sha256'] != hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()):
                raise SensorContractError('relative state must bind current sensor observations')
            rotation = np.asarray(relative['relative_orientation']['rotation_initial_body_from_current_body'])
            position = np.asarray(relative['position_initial_body_m'], dtype=float)
            if (position.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(position).all()
                    or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7, rtol=0)
                    or abs(np.linalg.det(rotation)-1) > 1e-7):
                raise SensorContractError('unbroken observed relative pose required; no command fallback')
            if self.last_ns is not None and (now_ns-self.last_ns != 100_000_000 or tuple(depth['identity']) != self.identity):
                raise SensorContractError('ray evidence episode or clock discontinuity')
            if relative['relative_orientation']['decision_ns'] != now_ns:
                raise SensorContractError('current relative orientation required')
            if self.last_ns is not None:
                motion = relative['motion']
                delta = np.asarray(motion['translation_previous_body_m'], dtype=float)
                if (motion['status'] != 'OBSERVED_TRANSLATION' or motion['rank'] != 3 or delta.shape != (3,)
                        or not np.isfinite(delta).all() or np.linalg.norm(delta) > .15
                        or not np.allclose(position, self.position+self.rotation@delta, atol=1e-8, rtol=0)):
                    raise SensorContractError('position must compose observed full translations')
            if self.last_ns is None:
                if relative['motion'] is not None or not np.array_equal(position, np.zeros(3)) or not np.array_equal(rotation, np.eye(3)):
                    raise SensorContractError('initial observation defines the relative origin')
                force = policy['sensor_state']['sensed']['specific_force']
                if not force['valid'].all(): raise SensorContractError('initial gravity observation unavailable')
                mean = force['values'].mean(axis=0)
                if abs(np.linalg.norm(mean)-9.81) > .75: raise SensorContractError('initial gravity hypothesis not quiet')
                self.up_initial = mean/np.linalg.norm(mean)
            self.position, self.rotation, self.latest_surface = position.copy(), rotation.copy(), deepcopy(surface)
            self.last_ns, self.identity = now_ns, tuple(depth['identity'])
            # Keep the first useful view through stationary holds; add views
            # when pose changes, not repeated copies of the same observation.
            add = not self.frames
            if self.frames:
                previous = self.frames[-1]
                add = (np.linalg.norm(position-previous['position']) >= .04
                       or np.linalg.norm(rotation-previous['rotation']) >= .08)
            self.latest_frame = {'position': position.copy(), 'rotation': rotation.copy(),
                'depth': depth['depth_m'].copy(), 'valid': depth['valid'].copy(), 'measured_ns': now_ns}
            if add:
                self.frames = [*self.frames[-63:], self.latest_frame]
            return {'retained_views': len(self.frames), 'oldest_ns': self.frames[0]['measured_ns'],
                    'new_view_added': bool(add), 'corners': observed_corners(surface),
                    'static_scene_assumed': True, 'calibrated_uncertainty': False}
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.failed = True
            raise SensorContractError('ray evidence unavailable; stop geometric control') from error

    def query(self, points_body, ground_support_allowed, *, now_ns):
        if self.failed or self.last_ns is None or self.last_ns != now_ns: raise SensorContractError('current valid ray evidence required')
        points = np.asarray(points_body, dtype=float); support = np.asarray(ground_support_allowed)
        if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
                or support.dtype != bool or support.shape != (len(points),)):
            raise SensorContractError('finite body-frame query and explicit support roles required')
        world = points@self.rotation.T+self.position  # Arbitrary initial body frame, not simulator/world pose.
        free, supported, conflict = (np.zeros(len(points), bool) for _ in range(3))
        transform = np.asarray(BODY_FROM_OPTICAL)
        frames = self.frames if self.frames[-1]['measured_ns'] == now_ns else [*self.frames, self.latest_frame]
        for frame in frames:
            # Inspect all retained views, including already supported points:
            # a contradictory observed surface must not be hidden by old free evidence.
            unresolved = np.arange(len(points))
            body = (world[unresolved]-frame['position'])@frame['rotation']
            optical = (body-transform[:3, 3])@transform[:3, :3]
            z = optical[:, 2]
            u = np.divide(FOCAL*optical[:, 0], z, out=np.full(len(z), -100.), where=z > .2)+320-.5
            v = np.divide(FOCAL*optical[:, 1], z, out=np.full(len(z), -100.), where=z > .2)+240-.5
            x, y = np.floor(u).astype(int), np.floor(v).astype(int)
            inside = (z >= .2) & (z <= 5.) & (x >= 1) & (x < 638) & (y >= 1) & (y < 478)
            ids = np.flatnonzero(inside)
            if not len(ids): continue
            x, y = x[ids], y[ids]
            dx = np.array([0, 1, 0, 1]); dy = np.array([0, 0, 1, 1])
            values = frame['depth'][y[:, None]+dy, x[:, None]+dx]
            valid = frame['valid'][y[:, None]+dy, x[:, None]+dx].all(axis=1)
            good_free = valid & (values.min(axis=1)-z[ids] >= .04)
            free[unresolved[ids[good_free]]] = True
            # Support is distinct from free space: only low calf/foot bands,
            # an actually observed locally horizontal surface below the body,
            # and a small signed distance to it can use this development rule.
            px = x[:, None]+dx+.5; py = y[:, None]+dy+.5
            optical_hits = np.stack((values*(px-320)/FOCAL, values*(py-240)/FOCAL, values), axis=-1)
            hits = optical_hits@transform[:3, :3].T+transform[:3, 3]
            normal = np.cross(hits[:, 1]-hits[:, 0], hits[:, 2]-hits[:, 0])
            length = np.linalg.norm(normal, axis=1)
            normal = np.divide(normal, length[:, None], out=np.zeros_like(normal), where=length[:, None] > 1e-8)
            up = frame['rotation'].T@self.up_initial
            horizontal = np.abs(normal@up) >= .97
            plane_error = np.abs(np.sum((hits[:, 3]-hits[:, 0])*normal, axis=1))
            height = np.sum((body[ids]-hits[:, 0])*up, axis=1)
            good_support = (valid & support[unresolved[ids]] & horizontal & (plane_error <= .003)
                            & (hits[:, 0]@up < -.15) & (np.abs(height) <= .06))
            supported[unresolved[ids[good_support]]] = True
            near_surface = valid & (np.min(np.abs(values-z[ids, None]), axis=1) <= .04) & ~good_support
            conflict[unresolved[ids[near_surface]]] = True
        free &= ~conflict; supported &= ~conflict
        return {'free': free, 'observed_ground_support': supported, 'contradictory_or_near_surface': conflict,
                'unknown_or_blocked': ~(free | supported),
                'all_samples_supported': bool(len(points) and (free | supported).all()),
                'retained_views': len(self.frames), 'continuous_volume_qualified': False,
                'future_gait_qualified': False, 'hardware_qualified': False}
