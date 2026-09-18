"""Current paired-observation planning cells with persistent contact history intact."""
from copy import deepcopy
import hashlib
import json
from types import MappingProxyType
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import body_points as primary_points
from lewm.auxiliary_downward45_depth_observation_development import body_points as auxiliary_points
from lewm.joint_visual_floor_map_development import CELL_M
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMap
from lewm.later_floor_resolution_controller_development import RecordingFloorGeometry


def cell_digest(cells):
    return hashlib.sha256(json.dumps(sorted(cells), separators=(',', ':')).encode()).hexdigest()


def occupied_cells(points_map, floor_height):
    points = np.asarray(points_map)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise SensorContractError('finite current measured map points required')
    above = points[(points[:,2] > floor_height+.03) & (points[:,2] < floor_height+.65)]
    keys = np.floor(above[:,:2]/CELL_M).astype(int)
    return {tuple(map(int,c)) for c in np.unique(keys,axis=0) if np.all(c >= -100) and np.all(c < 100)}


class CurrentObservationPlanningView:
    """Explicit selector interface; no fallback to accumulated routing cells."""
    def __init__(self, owner, floor, occupied, witnesses, *, frame, now_ns):
        owner.surface._current(now_ns)
        if (owner.failed or owner.surface.failed or type(frame) is not int
                or frame != len(owner.surface.route)-1 or type(now_ns) is not int
                or len(witnesses) != 2
                or [w['camera'] for w in witnesses] != ['primary','auxiliary']
                or any(w['frame'] != frame or w['measured_ns'] != now_ns for w in witnesses)
                or not set(floor) <= set(owner.floor) or not set(occupied) <= set(owner.occupied)):
            raise SensorContractError('current paired measured cells within retained evidence required')
        self._owner = owner; self.surface = owner.surface
        self.map_from_initial = np.frombuffer(owner.map_from_initial.tobytes(), dtype=owner.map_from_initial.dtype).reshape(3,3)
        self.floor_height = owner.floor_height
        self.floor = MappingProxyType({c:frame for c in floor})
        self.occupied = MappingProxyType({c:frame for c in occupied})
        self.frame = frame; self.now_ns = now_ns
        self.receipt = dict(frame=frame, measured_ns=now_ns,
            planning_map_variant='current_paired_observation',
            current_floor_cells=len(self.floor), current_occupied_cells=len(self.occupied),
            current_floor_cell_sha256=cell_digest(self.floor), current_occupied_cell_sha256=cell_digest(self.occupied),
            retained_floor_cells=len(owner.floor), retained_occupied_cells=len(owner.occupied),
            camera_witnesses=deepcopy(witnesses), accumulated_planning_cells_queried=False,
            selector_scan_state_retained=True,
            persistent_contact_history_retained=True, tracking_and_floor_anchor_history_retained=True,
            learned_temporal_history_and_residual_retained=True, mission_and_settling_state_retained=True,
            current_cells_recomputed_from_first_witness_frames=False,
            native_state_used=False, memoryless_controller=False, navigation_qualified=False)

    @property
    def failed(self):
        return self._owner.failed or self.surface.failed

    def current(self, now_ns):
        self.surface._current(now_ns)
        if self.failed or now_ns != self.now_ns or self.frame != len(self.surface.route)-1:
            raise SensorContractError('planning view belongs to one current observation only')

    def waypoint(self, goal_initial_xy, *, now_ns):
        self.current(now_ns)
        return MeasuredFloorTransportMap.waypoint(self, goal_initial_xy, now_ns=now_ns)


class CurrentObservationPlanningMap(MeasuredFloorTransportMap):
    """Keep original accumulated evidence; capture a separate current planning view."""
    def __init__(self, *, identity=(0,0,0)):
        super().__init__(identity=identity)
        self.current_planning_view = None

    def _observe_both(self, policy, depth, evidence, *, auxiliary_depth, now_ns):
        self.current_planning_view = None
        receipt = super()._observe_both(policy, depth, evidence, auxiliary_depth=auxiliary_depth, now_ns=now_ns)
        context = self.frame_geometry
        if not isinstance(context, RecordingFloorGeometry) or len(context.observations) != 2:
            raise SensorContractError('original complete paired floor observations required')
        floor = {tuple(map(int,c)) for observation in context.observations for c in observation['cells']}
        B = self.map_from_initial; R = B@self.surface.rotation; p = B@self.surface.position
        # Preserve each predecessor's arithmetic order at cell boundaries.
        primary = primary_points(depth, policy, now_ns=now_ns, stride=4)
        primary_map = primary['points_body_m'][primary['valid']]@R.T+p
        auxiliary = auxiliary_points(auxiliary_depth, policy, now_ns=now_ns, stride=4)
        auxiliary_initial = auxiliary['points_body_m'][auxiliary['valid']]@self.surface.rotation.T+self.surface.position
        auxiliary_map = auxiliary_initial@B.T
        occupied = occupied_cells(primary_map,self.floor_height) | occupied_cells(auxiliary_map,self.floor_height)
        self.current_planning_view = CurrentObservationPlanningView(self, floor, occupied,
            [observation['witness'] for observation in context.observations],
            frame=receipt['frame'], now_ns=now_ns)
        return receipt

    def planning_view(self, *, now_ns):
        if self.current_planning_view is None:
            raise SensorContractError('complete current paired planning view unavailable')
        self.current_planning_view.current(now_ns)
        return self.current_planning_view
