"""Measured flat-floor coverage in a gravity-aligned visual map; unknown stays unknown."""
import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL, body_points
from lewm.causal_sensor_state import SensorContractError
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.joint_visual_surface_memory_development import JointVisualSurfaceMemory
from lewm.observed_floor_waypoint_development import CELL_M, propose

T = np.asarray(BODY_FROM_OPTICAL)
GRID = np.array([(x, y) for x in range(-100, 100) for y in range(-100, 100)], np.int64)


def floor_coverage(depth, valid, map_from_body, translation_map, floor_height, cells=GRID):
    """Require every image cell in each projected floor-square rectangle.

    The fixed measured plane proposes where to query; it cannot provide coverage.
    Every projected cell must independently pass the original ground mesh tests
    and have all four measured heights within 10 mm of that hypothesis.
    """
    R, p = np.asarray(map_from_body, float), np.asarray(translation_map, float)
    cells = np.asarray(cells)
    if (R.shape != (3, 3) or p.shape != (3,) or not np.isfinite([floor_height]).all()
            or not np.isfinite(R).all() or not np.isfinite(p).all()
            or not np.allclose(R.T@R, np.eye(3), atol=1e-8, rtol=0) or abs(np.linalg.det(R)-1) > 1e-8
            or cells.ndim != 2 or cells.shape[1:] != (2,) or cells.dtype.kind not in 'iu'
            or len(cells) > 40000 or np.any(cells < -100) or np.any(cells >= 100)):
        raise SensorContractError('bounded floor grid and proper observed map transform required')
    up_body = R[2]
    index = observed_floor_cell_index(depth, valid, up_body)
    yy, xx = np.indices((480, 640))
    optical = np.stack((depth*(xx+.5-320)/FOCAL, depth*(yy+.5-240)/FOCAL, depth), axis=-1)
    body = optical@T[:3, :3].T+T[:3, 3]
    heights = body@up_body+p[2]
    near = np.abs(heights-floor_height) <= .01
    good = index['ground_cells'] & near[:-1, :-1] & near[:-1, 1:] & near[1:, :-1] & near[1:, 1:]
    prefix = np.zeros((480, 640), np.int64); prefix[1:, 1:] = (~good).cumsum(0).cumsum(1)
    xy = (cells[:, None, :]+np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))*CELL_M
    world = np.concatenate((xy, np.full((*xy.shape[:-1], 1), floor_height)), axis=-1)
    points = (world-p)@R
    camera = (points-T[:3, 3])@T[:3, :3]
    z = camera[..., 2]
    uv = camera[..., :2]/np.maximum(z[..., None], 1e-12)*FOCAL+[319.5, 239.5]
    lo, hi = uv.min(1)-1e-9, uv.max(1)+1e-9
    visible = ((z >= .2)&(z <= 5.)).all(1) & (lo >= 0).all(1) & (hi < [639, 479]).all(1)
    a, b = np.floor(np.clip(lo, -1, 640)).astype(int), np.floor(np.clip(hi, -1, 640)).astype(int)
    covered = np.zeros(len(cells), bool)
    ids = np.flatnonzero(visible)
    if len(ids):
        x0, y0 = a[ids].T; x1, y1 = (b[ids]+1).T
        covered[ids] = prefix[y1, x1]-prefix[y0, x1]-prefix[y1, x0]+prefix[y0, x0] == 0
    return dict(covered=covered, floor_candidate_pixels=int(good.sum()),
        projected_lower_xy=a, projected_upper_xy=b, ground_support_approved=False)


class JointVisualFloorMap:
    def __init__(self, *, identity=(0, 0, 0)):
        self.surface = JointVisualSurfaceMemory(identity=identity)
        self.map_from_initial = None
        self.floor_height = None
        self.floor = {}; self.occupied = {}
        self.failed = False

    def observe(self, policy, depth, evidence, *, now_ns):
        if self.failed: raise SensorContractError('floor map failure latched')
        try:
            receipt = self.surface.observe(policy, depth, evidence, now_ns=now_ns)
            if self.map_from_initial is None:
                force = policy['sensor_state']['sensed']['specific_force']
                commands = policy['sensor_state']['control']['applied_command']
                if not force['valid'].all() or not commands['valid'].all() or np.any(np.abs(commands['values']) > 1e-8):
                    raise SensorContractError('quiet initial public force history required')
                up = force['values'].mean(0); magnitude = np.linalg.norm(up)
                if not 8 <= magnitude <= 12: raise SensorContractError('initial gravity magnitude inconsistent')
                up = up/magnitude; forward = np.array([1., 0., 0.])-up*up[0]
                if np.linalg.norm(forward) < .8: raise SensorContractError('initial gravity/forward frame degenerate')
                forward /= np.linalg.norm(forward)
                self.map_from_initial = np.stack((forward, np.cross(up, forward), up))
            R = self.map_from_initial@self.surface.rotation
            p = self.map_from_initial@self.surface.position
            cloud = body_points(depth, policy, now_ns=now_ns, stride=4)
            points = cloud['points_body_m'][cloud['valid']]@R.T+p
            if self.floor_height is None:
                up = R[2]; index = observed_floor_cell_index(depth['depth_m'], depth['valid'], up)
                rr, cc = np.nonzero(index['ground_cells'])
                if len(rr) < 100: raise SensorContractError('initial observed floor hypothesis unavailable')
                z = depth['depth_m'][rr, cc]
                optical = np.column_stack((z*(cc+.5-320)/FOCAL, z*(rr+.5-240)/FOCAL, z))
                xyz = (optical@T[:3, :3].T+T[:3, 3])@R.T+p
                self.floor_height = float(np.median(xyz[:, 2]))
            coverage = floor_coverage(depth['depth_m'], depth['valid'], R, p, self.floor_height)
            for cell in GRID[coverage['covered']]: self.floor.setdefault(tuple(map(int, cell)), receipt['frame'])
            above = points[(points[:, 2] > self.floor_height+.03)&(points[:, 2] < self.floor_height+.65)]
            keys = np.floor(above[:, :2]/CELL_M).astype(int)
            for cell in np.unique(keys, axis=0):
                if np.all(cell >= -100) and np.all(cell < 100): self.occupied.setdefault(tuple(map(int, cell)), receipt['frame'])
            return dict(frame=receipt['frame'], measured_ns=now_ns,
                rgb_sha256=receipt['rgb_sha256'], depth_sha256=receipt['depth_sha256'],
                current_observed_floor_cells=int(coverage['covered'].sum()),
                retained_observed_floor_cells=len(self.floor), retained_occupied_cells=len(self.occupied),
                floor_height_map_m=self.floor_height, map_from_initial=self.map_from_initial.tolist(),
                static_flat_floor_hypothesis=True, uncertainty_calibrated=False,
                continuous_floor_or_volume_coverage=False, navigation_qualified=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('observed floor map unavailable') from error

    def waypoint(self, goal_initial_xy, *, now_ns):
        if self.failed: raise SensorContractError('floor map failure latched')
        self.surface._current(now_ns)
        goal = np.asarray(goal_initial_xy, float)
        if goal.shape != (2,) or not np.isfinite(goal).all(): raise SensorContractError('finite mission XY required')
        position = self.map_from_initial@self.surface.position
        target = self.map_from_initial@np.r_[goal, 0.]
        return propose(self.floor, self.occupied, position[:2], target[:2])
