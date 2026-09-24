"""Compose the existing single-pass bounds queries with body-projected mapping.

Only eight distinct empty indices are replaced. Their insertion method and
stored evidence schema are unchanged; model, selector and observation methods
remain those of the completed body-projected controller.
"""
from lewm.body_projected_tiled_controller_development import (
    BodyProjectedTiledController, BodyProjectedTiledFloorMap, CONTROLLER as BASELINE)
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm.packed_fused_scoped_controller_development import INDEX_FIELDS, index_owners
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex

CONTROLLER = 'single_pass_body_projected_controller_v1'
FLAG = 'single_pass_measured_bound_queries_enabled'


def install_empty_single_pass_indices(controller):
    memory = controller.memory
    if (type(controller.mapper) is not BodyProjectedTiledFloorMap
            or type(memory) is not MeasuredFloorTransportMemory
            or memory is not controller.mapper.surface or memory.route
            or memory.last_ns is not None or memory.failed or controller.mapper.failed
            or memory.frame_geometry is not None or controller.mapper.frame_geometry is not None):
        raise ValueError('fresh body-projected mapper and original memory alias required')
    owners = index_owners(memory)
    indices = [getattr(owner, name) for owner, name in owners]
    if (len({id(index) for index in indices}) != 8
            or any(type(index) is not PackedOwnedMeasuredSampleBoundsIndex
                or set(vars(index)) != INDEX_FIELDS
                or any(type(value) is not dict or value for value in vars(index).values())
                for index in indices)):
        raise ValueError('eight distinct empty packed-owned indices required')
    replacements = [SinglePassMeasuredSampleBoundsIndex() for _ in indices]
    for (owner, name), replacement in zip(owners, replacements, strict=True):
        setattr(owner, name, replacement)


class SinglePassBodyProjectedController(BodyProjectedTiledController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_empty_single_pass_indices(self)

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}


def normalize_to_body_projected(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit single-pass body-projected controller identity required')
    result = decision.copy()
    result.pop(FLAG)
    result['controller'] = BASELINE
    return result
