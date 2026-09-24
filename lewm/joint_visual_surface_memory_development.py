"""Causal measured-surface voxels and visited-pose history, without free-space claims.

Voxels enclose sampled returns. Their intersection with a nominal robot AABB is
only a possible conflict. Absence is UNKNOWN, never clearance. Static scene and
uncalibrated visual poses are explicit assumptions; no ray carving or pose reset.
"""
import hashlib
from copy import deepcopy
import numpy as np

from lewm.causal_depth_observation_development import body_points, validate_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_sensor_anchored_goal_development import current_joint_pose

VOXEL_M = .025
MAX_VOXELS = 250_000
MAX_FRAMES = 4096
MAX_COORDINATE_M = 50.
STRIDE = 4


class SurfaceIndex:
    """Monotone bounded observation index; each voxel retains its first witness."""
    def __init__(self):
        self.cells = {}

    def insert(self, points, witness):
        p = np.asarray(points, float)
        if (p.ndim != 2 or p.shape[1:] != (3,) or len(p) > 19200
                or not np.isfinite(p).all() or np.any(np.abs(p) > MAX_COORDINATE_M)):
            raise SensorContractError('bounded finite sampled surfaces required')
        keys = np.unique(np.floor(p/VOXEL_M).astype(np.int64), axis=0)
        new = [tuple(map(int, k)) for k in keys if tuple(k) not in self.cells]
        if len(self.cells)+len(new) > MAX_VOXELS:
            raise SensorContractError('surface memory capacity exhausted; no evidence eviction')
        for key in new:
            self.cells[key] = deepcopy(witness)

    def intersect(self, lower, upper):
        low, high = np.asarray(lower, float), np.asarray(upper, float)
        if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
                or np.any(low > high) or np.any(np.abs([low, high]) > MAX_COORDINATE_M)):
            raise SensorContractError('finite bounded ordered query box required')
        # Closed voxel enclosures: a shared boundary is a possible intersection.
        a = np.ceil(low/VOXEL_M).astype(int)-1
        b = np.floor(high/VOXEL_M).astype(int)
        volume = int(np.prod(b-a+1))
        if volume <= len(self.cells):
            keys = ((x, y, z) for x in range(a[0], b[0]+1)
                for y in range(a[1], b[1]+1) for z in range(a[2], b[2]+1))
            hits = [k for k in keys if k in self.cells]
        else:
            hits = [k for k in self.cells if all(a[i] <= k[i] <= b[i] for i in range(3))]
        key = min(hits) if hits else None
        return dict(status='POSSIBLE_SURFACE_INTERSECTION' if hits else 'UNKNOWN',
            intersecting_voxels=len(hits), first_cell=list(key) if key is not None else None,
            witness=deepcopy(self.cells[key]) if key is not None else None,
            free_space_established=False, motion_permitted=False)


class JointVisualSurfaceMemory:
    def __init__(self, *, identity):
        self.identity = tuple(identity)
        self.index = SurfaceIndex()
        self.latest = SurfaceIndex()
        self.route = []
        self.last_ns = None
        self.failed = False
        self.position = self.rotation = self.joints = None

    def observe(self, policy, depth, evidence, *, now_ns):
        if self.failed:
            raise SensorContractError('visual surface memory failure latched')
        try:
            validate_depth(depth, policy, now_ns=now_ns)
            p, R, pose = current_joint_pose(evidence, identity=self.identity, now_ns=now_ns)
            h = hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()
            if (pose['frame'] != len(self.route) or depth['measured_ns'] != now_ns
                    or pose['rgb_sha256'] != depth['rgb_sha256'] or pose['depth_sha256'] != h
                    or (self.last_ns is not None and now_ns-self.last_ns != 100_000_000)
                    or len(self.route) >= MAX_FRAMES):
                raise SensorContractError('uninterrupted same-image pose/depth history required')
            joints = policy['sensor_state']['sensed']['joints']
            if joints['measured_ns'][-1] != now_ns or not joints['valid'][-1, :12].all():
                raise SensorContractError('current measured joint posture required')
            witness = dict(frame=pose['frame'], measured_ns=now_ns,
                rgb_sha256=pose['rgb_sha256'], depth_sha256=h)
            cloud = body_points(depth, policy, now_ns=now_ns, stride=STRIDE)
            points = cloud['points_body_m'][cloud['valid']] @ R.T + p
            latest = SurfaceIndex(); latest.insert(points, witness)
            self.index.insert(points, witness)
            self.latest = latest
            self.route.append(witness | dict(position_initial_body_m=p.tolist(),
                rotation_initial_body_from_body=R.tolist()))
            self.position, self.rotation = p.copy(), R.copy()
            self.joints = joints['values'][-1, :12].copy()
            self.last_ns = now_ns
            return dict(**witness, sampled_returns=len(points), retained_voxels=len(self.index.cells),
                current_voxels=len(latest.cells), visited_poses=len(self.route),
                static_scene_assumed=True, uncertainty_calibrated=False,
                free_space_established=False, navigation_qualified=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('visual surface memory unavailable; stop') from error

    def _current(self, now_ns):
        if self.failed or self.last_ns is None or now_ns != self.last_ns:
            raise SensorContractError('current uninterrupted surface memory required')

    def footprint(self, geometry, displacement_body_xy, yaw_rad, *, now_ns, persistent=True):
        """Discrete predicted base pose, with CURRENT measured joints; no future gait."""
        self._current(now_ns)
        d = np.asarray(displacement_body_xy, float)
        if d.shape != (2,) or not np.isfinite(d).all() or np.linalg.norm(d) > 5.:
            raise SensorContractError('bounded finite predicted displacement required')
        if not np.isfinite(yaw_rad) or abs(yaw_rad) > np.pi or type(persistent) is not bool:
            raise SensorContractError('wrapped finite yaw and explicit memory variant required')
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        R = self.rotation @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        p = self.position + self.rotation @ np.r_[d, 0.]
        shapes = geometry.supports(self.joints, R)['shapes']
        index = self.index if persistent else self.latest
        rows = [dict(shape_id=shape['shape_id'], **index.intersect(
            np.asarray(shape['lower'])+p, np.asarray(shape['upper'])+p)) for shape in shapes]
        return dict(shapes=rows, memory='persistent' if persistent else 'current_frame',
            measured_ns=now_ns, possible_intersection=any(r['intersecting_voxels'] for r in rows),
            current_joint_posture_only=True, discrete_prediction_only=True,
            free_space_established=False, motion_permitted=False, navigation_qualified=False)

    def backtrack(self, target_frame, *, now_ns):
        """Reverse actually visited poses, without line-of-sight shortcuts or teleportation."""
        self._current(now_ns)
        if type(target_frame) is not int or not 0 <= target_frame < len(self.route):
            raise SensorContractError('previously observed route target required')
        return dict(targets=deepcopy(self.route[target_frame:-1][::-1]),
            target_frame=target_frame, current_frame=len(self.route)-1,
            proposal_only=True, physical_execution_required=True,
            clearance_must_be_reobserved=True, motion_permitted=False)
