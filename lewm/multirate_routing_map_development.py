"""Measured routing cells at their actual observation time, independently of planning.

This map proposes routes only. It supplies no robot-volume/contact check and
does not authorize a command. Updates may skip observations; pose estimation
and the actuation loop must keep their own clocks and freshness requirements.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL, body_points, validate_depth
from lewm.auxiliary_downward45_depth_observation_development import (
    body_points as auxiliary_points, validate_depth as validate_auxiliary)
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.extended_return_budget_transport_development import current_measured_floor_pose
from lewm.body_projected_floor_geometry_development import BodyProjectedFloorGeometry
from lewm.tiled_density_progressive_floor_controller_development import TiledDensityRecordingFloorGeometry
from lewm.joint_visual_floor_map_development import GRID, CELL_M
from lewm.observed_floor_waypoint_development import propose


class Geometry(BodyProjectedFloorGeometry):
    index = TiledDensityRecordingFloorGeometry.index


@dataclass(frozen=True)
class RoutingSnapshot:
    frame: int
    measured_ns: int
    floor: frozenset
    occupied: frozenset
    position_map: tuple
    map_from_initial: tuple
    floor_height: float
    primary_current_floor_cells: int
    auxiliary_current_floor_cells: int
    fine_occupied: frozenset = frozenset()

    def age_ns(self, *, now_ns):
        if type(now_ns) is not int or now_ns < self.measured_ns:
            raise SensorContractError('routing snapshot cannot be used before its observation')
        return now_ns-self.measured_ns

    def route(self, goal_initial_xy):
        goal = np.asarray(goal_initial_xy, float)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise SensorContractError('finite routing goal required')
        mapped_goal = np.asarray(self.map_from_initial)@np.r_[goal, 0.]
        result = propose(self.floor, self.occupied, self.position_map[:2], mapped_goal[:2])
        return result | dict(map_frame=self.frame, map_measured_ns=self.measured_ns,
            command_authorized=False, current_robot_pose_assumed=False)


class MultirateRoutingMap:
    retain_fine_obstacles = False

    def __init__(self, *, identity=(0, 0, 0)):
        self.identity = identity
        self.floor = set(); self.occupied = set()
        self.fine_occupied = set()
        self.B = None; self.floor_height = None; self.latest = None
        self.failed = False

    def update(self, policy, depth, evidence, *, auxiliary_depth, measured_ns):
        if self.failed: raise SensorContractError('routing-map failure latched')
        geometry = Geometry()
        try:
            validate_depth(depth, policy, now_ns=measured_ns)
            validate_auxiliary(auxiliary_depth, policy, now_ns=measured_ns)
            p, R, pose = current_measured_floor_pose(evidence, identity=self.identity, now_ns=measured_ns)
            digest = hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()
            if (pose['depth_sha256'] != digest or pose['rgb_sha256'] != depth['rgb_sha256']
                    or depth['measured_ns'] != measured_ns or auxiliary_depth['measured_ns'] != measured_ns):
                raise SensorContractError('paired acquisition and measured pose identities required')
            if self.latest is not None and (pose['frame'] <= self.latest.frame
                    or measured_ns <= self.latest.measured_ns):
                raise SensorContractError('strictly newer measured map update required')
            if self.B is None:
                if pose['frame'] != 0: raise SensorContractError('initial measured map frame required')
                force = policy['sensor_state']['sensed']['specific_force']
                command = policy['sensor_state']['control']['applied_command']
                if not force['valid'].all() or not command['valid'].all() or np.any(np.abs(command['values']) > 1e-8):
                    raise SensorContractError('quiet measured gravity initialization required')
                up = force['values'].mean(0); magnitude = np.linalg.norm(up)
                if not 8 <= magnitude <= 12: raise SensorContractError('initial gravity magnitude inconsistent')
                up /= magnitude; forward = np.array([1., 0., 0.])-up*up[0]
                if np.linalg.norm(forward) < .8: raise SensorContractError('initial forward/gravity frame degenerate')
                forward /= np.linalg.norm(forward)
                self.B = np.stack((forward, np.cross(up, forward), up))
            Q, q = self.B@R, self.B@p
            if self.floor_height is None:
                index = geometry.index(depth['depth_m'], depth['valid'], Q[2])
                rr, cc = np.nonzero(index['ground_cells'])
                if len(rr) < 100: raise SensorContractError('initial measured floor unavailable')
                z = depth['depth_m'][rr, cc]; T = np.asarray(BODY_FROM_OPTICAL)
                optical = np.column_stack((z*(cc+.5-320)/FOCAL, z*(rr+.5-240)/FOCAL, z))
                xyz = (optical@T[:3, :3].T+T[:3, 3])@Q.T+q
                self.floor_height = float(np.median(xyz[:, 2]))
            counts = []
            new_floor = set(); new_occupied = set()
            for auxiliary, packet in ((False, depth), (True, auxiliary_depth)):
                camera_R, camera_p = reference_pose(Q, q) if auxiliary else (Q, q)
                coverage = geometry.floor_coverage(packet['depth_m'], packet['valid'],
                    camera_R, camera_p, self.floor_height)
                new_floor.update(tuple(map(int, cell)) for cell in GRID[coverage['covered']])
                counts.append(int(coverage['covered'].sum()))
                cloud = (auxiliary_points if auxiliary else body_points)(packet, policy, now_ns=measured_ns, stride=4)
                points = cloud['points_body_m'][cloud['valid']]
                # Match the original primary/auxiliary transform order.
                mapped = ((points@R.T+p)@self.B.T) if auxiliary else (points@Q.T+q)
                above = mapped[(mapped[:, 2] > self.floor_height+.03)&(mapped[:, 2] < self.floor_height+.65)]
                keys = np.unique(np.floor(above[:, :2]/CELL_M).astype(int), axis=0)
                new_occupied.update(tuple(map(int, cell)) for cell in keys
                    if np.all(cell >= -100) and np.all(cell < 100))
                if self.retain_fine_obstacles:
                    fine_keys = np.unique(np.floor(above[:, :2]/.01).astype(int), axis=0)
                    self.fine_occupied.update(tuple(map(int, cell)) for cell in fine_keys
                        if np.all(cell >= -500) and np.all(cell < 500))
            self.floor.update(new_floor); self.occupied.update(new_occupied)
            self.latest = RoutingSnapshot(pose['frame'], measured_ns,
                frozenset(self.floor), frozenset(self.occupied), tuple(map(float, q)),
                tuple(tuple(map(float, row)) for row in self.B), self.floor_height, *counts,
                frozenset(self.fine_occupied))
            return self.latest
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
            self.failed = True
            raise
        finally:
            geometry.close()
