"""Verified frozen-footprint policy with an observation-local mesh provider."""
from lewm.causal_sensor_state import SensorContractError
from lewm.frozen_footprint_anchored_controller_development import FrozenFootprintAnchoredController
from lewm.later_floor_resolution_controller_development import RecordingFloorGeometry
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMap
from lewm.reused_floor_mesh_development import ReusedFloorMeshCache

CONTROLLER = 'reused_floor_mesh_frozen_footprint_controller_v1'
FLAG = 'observation_local_floor_mesh_reuse_enabled'


class ReusedMeshRecordingFloorGeometry(RecordingFloorGeometry, ReusedFloorMeshCache):
    """Keep the original ordered floor witnesses and all geometry consumers."""


class ReusedMeshFloorMap(MeasuredFloorTransportMap):
    def observe(self, policy, depth, evidence, *, auxiliary_depth, now_ns):
        if self.frame_geometry is not None:
            raise SensorContractError('nested floor-map observation is unavailable')
        frame = len(self.surface.route)
        context = ReusedMeshRecordingFloorGeometry(depth, auxiliary_depth, frame, now_ns)
        self.frame_geometry = self.surface.frame_geometry = context
        try:
            receipt = self._observe_both(policy, depth, evidence, auxiliary_depth=auxiliary_depth, now_ns=now_ns)
            self.surface.later_floor_evidence.record_pair(frame, now_ns, self.map_from_initial,
                self.floor_height, context.observations)
            return receipt
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
            self.failed = self.surface.failed = True
            raise
        finally:
            self.last_cache_counts = context.counts()
            context.close()
            self.frame_geometry = self.surface.frame_geometry = None


class ReusedFloorMeshController(FrozenFootprintAnchoredController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (type(self.mapper) is not MeasuredFloorTransportMap
                or self.memory is not self.mapper.surface or self.memory.route
                or self.mapper.frame_geometry is not None or self.mapper.failed):
            raise ValueError('fresh original measured floor map and memory alias required')
        self.mapper = ReusedMeshFloorMap(identity=(0, 0, 0))
        self.memory = self.mapper.surface

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}
