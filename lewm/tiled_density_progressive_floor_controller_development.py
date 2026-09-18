"""Compose tiled dense geometry with progressive retained-patch batching."""
from lewm.tiled_density_routed_floor_cell_index_development import observed_floor_cell_index
from lewm.tiled_density_floor_registration_development import TiledDensityFloorRegistration
from lewm.eligible_floor_registration_development import bind
from lewm.frame_floor_index_cache_development import FrameFloorIndexCache
from lewm.later_floor_resolution_controller_development import RecordingFloorGeometry, LaterResolvedFloorMap
from lewm.density_routed_floor_controller_development import DensityRoutedFloorMap
from lewm.density_routed_floor_registration_development import DensityRoutedFloorRegistration
from lewm.progressive_batched_floor_controller_development import ProgressiveBatchedFloorController, CONTROLLER as BASELINE

CONTROLLER='tiled_density_progressive_floor_controller_v1'
FLAG='tiled_dense_floor_geometry_enabled'


class TiledDensityRecordingFloorGeometry(RecordingFloorGeometry):
    index=bind(FrameFloorIndexCache.index,observed_floor_cell_index=observed_floor_cell_index)


class TiledDensityFloorMap(DensityRoutedFloorMap):
    observe=bind(LaterResolvedFloorMap.observe,RecordingFloorGeometry=TiledDensityRecordingFloorGeometry)


def install_fresh_tiled_consumers(controller):
    old_map=controller.mapper;old_registration=controller.registration
    if (type(old_map) is not DensityRoutedFloorMap or controller.memory is not old_map.surface
            or old_map.frame_geometry is not None or old_map.failed or controller.memory.route
            or controller.memory.frame_geometry is not None
            or type(old_registration) is not DensityRoutedFloorRegistration):
        raise ValueError('fresh density-routed mapper, memory alias and registration required')
    registration=TiledDensityFloorRegistration(identity=old_registration.identity)
    if vars(old_registration)!=vars(registration):
        raise ValueError('untouched original initial registration required')
    mapper=object.__new__(TiledDensityFloorMap)
    mapper.__dict__=vars(old_map).copy()
    controller.mapper=mapper;controller.registration=registration


class TiledDensityProgressiveFloorController(ProgressiveBatchedFloorController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        install_fresh_tiled_consumers(self)

    def _result(self,*args,**kwargs):
        return super()._result(*args,**kwargs)|{'controller':CONTROLLER,FLAG:True}


def normalize_to_progressive(decision):
    if decision.get('controller')!=CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit tiled-density controller identity required')
    result=decision.copy();result.pop(FLAG);result['controller']=BASELINE
    return result
