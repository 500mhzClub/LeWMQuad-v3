"""Compose the existing packed-owned insertion with the current receipt path."""
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex
from lewm.fused_scoped_batched_controller_development import (
    FusedScopedBatchedController, CONTROLLER as BASELINE)

CONTROLLER = 'packed_owned_fused_scoped_batched_anchored_controller_v1'
FLAG = 'packed_owned_measured_bound_insertion_enabled'
INDEX_PATHS = (('index',), ('auxiliary_index',), ('partition', 'floor'), ('partition', 'other'),
               ('auxiliary_partition', 'floor'), ('auxiliary_partition', 'other'),
               ('confirmed_auxiliary_partition', 'floor'), ('confirmed_auxiliary_partition', 'other'))
INDEX_FIELDS = {'cells', 'bounds', 'sample_counts', 'latest_frames'}


def index_owners(memory):
    return [(memory if len(path) == 1 else getattr(memory, path[0]), path[-1]) for path in INDEX_PATHS]


def install_empty_packed_indices(memory):
    if type(memory) is not MeasuredFloorTransportMemory:
        raise ValueError('exact current measured floor memory required')
    owners = index_owners(memory)
    indices = [getattr(owner, name) for owner, name in owners]
    if (len({id(index) for index in indices}) != 8
            or any(type(index) is not MeasuredSampleBoundsIndex
                   or set(vars(index)) != INDEX_FIELDS
                   or any(type(value) is not dict or value for value in vars(index).values())
                   for index in indices)):
        raise ValueError('eight distinct completely empty original bound indices required')
    # Validate the entire population before replacing any index.
    for owner, name in owners:
        setattr(owner, name, PackedOwnedMeasuredSampleBoundsIndex())


class PackedFusedScopedController(FusedScopedBatchedController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.memory is not self.mapper.surface:
            raise ValueError('original mapper/memory alias required')
        install_empty_packed_indices(self.memory)

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}


def normalize_to_fused(decision):
    if decision.get('controller') != CONTROLLER or decision.get(FLAG) is not True:
        raise ValueError('explicit packed-owned receipt composition required')
    result = decision.copy()
    result.pop(FLAG)
    result['controller'] = BASELINE
    return result
