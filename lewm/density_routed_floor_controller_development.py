"""Route floor kernels inside original registration and per-observation mapping."""
from lewm.density_routed_floor_cell_index_development import observed_floor_cell_index
from lewm.density_routed_floor_registration_development import DensityRoutedFloorRegistration
from lewm.eligible_floor_registration_development import bind
from lewm.frame_floor_index_cache_development import FrameFloorIndexCache
from lewm.later_floor_resolution_controller_development import RecordingFloorGeometry, LaterResolvedFloorMap
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMap
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.visibility_batched_footprint_controller_development import (
    VisibilityBatchedFootprintController, CONTROLLER as BASELINE)

CONTROLLER='density_routed_visibility_batched_floor_controller_v1'
FLAG='density_routed_exact_floor_index_enabled'


class DensityRoutedRecordingFloorGeometry(RecordingFloorGeometry):
    index=bind(FrameFloorIndexCache.index,observed_floor_cell_index=observed_floor_cell_index)


class DensityRoutedFloorMap(MeasuredFloorTransportMap):
    observe=bind(LaterResolvedFloorMap.observe,RecordingFloorGeometry=DensityRoutedRecordingFloorGeometry)


def install_fresh_routed_consumers(controller):
    old_map=controller.mapper;old_registration=controller.registration
    if (type(old_map) is not MeasuredFloorTransportMap or controller.memory is not old_map.surface
            or old_map.frame_geometry is not None or old_map.failed or controller.memory.route
            or controller.memory.frame_geometry is not None
            or type(old_registration) is not MeasuredFloorTransportRegistration):
        raise ValueError('original fresh mapper, memory alias and registration required')
    registration=DensityRoutedFloorRegistration(identity=old_registration.identity)
    if vars(old_registration)!=vars(registration):
        raise ValueError('untouched original initial registration required')
    # Transfer the fresh mapper fields without replacing packed memory or patch
    # stores installed by the baseline controller. No live mapper is converted.
    mapper=object.__new__(DensityRoutedFloorMap)
    mapper.__dict__=vars(old_map).copy()
    controller.mapper=mapper;controller.registration=registration


class DensityRoutedFloorController(VisibilityBatchedFootprintController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        install_fresh_routed_consumers(self)

    def _result(self,*args,**kwargs):
        return super()._result(*args,**kwargs)|{'controller':CONTROLLER,FLAG:True}


def normalize_to_visibility_batched(decision):
    if decision.get('controller')!=CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit density-routed controller identity required')
    result=decision.copy();result.pop(FLAG);result['controller']=BASELINE
    return result
