"""Reuse body projection only within the original tiled map observation scope."""
from lewm.body_projected_floor_geometry_development import BodyProjectedFloorGeometry
from lewm.eligible_floor_registration_development import bind
from lewm.later_floor_resolution_controller_development import LaterResolvedFloorMap
from lewm.tiled_density_progressive_floor_controller_development import (
    TiledDensityRecordingFloorGeometry, TiledDensityFloorMap,
    TiledDensityProgressiveFloorController, CONTROLLER as BASELINE)

CONTROLLER = 'body_projected_tiled_controller_v1'
FLAG = 'observation_local_body_projection_reuse_enabled'


class BodyProjectedRecordingFloorGeometry(TiledDensityRecordingFloorGeometry, BodyProjectedFloorGeometry):
    """Cooperative initialization retains recording and tiled index behavior."""


class BodyProjectedTiledFloorMap(TiledDensityFloorMap):
    observe = bind(LaterResolvedFloorMap.observe,
        RecordingFloorGeometry=BodyProjectedRecordingFloorGeometry)


def install_fresh_body_projected_map(controller):
    original = controller.mapper
    if (type(original) is not TiledDensityFloorMap
            or controller.memory is not original.surface
            or original.frame_geometry is not None or original.failed
            or controller.memory.route or controller.memory.frame_geometry is not None):
        raise ValueError('fresh tiled mapper and original memory alias required')
    replacement = object.__new__(BodyProjectedTiledFloorMap)
    replacement.__dict__ = vars(original).copy()
    controller.mapper = replacement


class BodyProjectedTiledController(TiledDensityProgressiveFloorController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_fresh_body_projected_map(self)

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}


def normalize_to_tiled(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit body-projected tiled controller identity required')
    result = decision.copy()
    result.pop(FLAG)
    result['controller'] = BASELINE
    return result
