"""Explicit empty-state index substitution for an exact-performance candidate."""
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex
from lewm.later_floor_resolution_controller_development import (
    LaterResolvedFloorMemory, LaterResolvedFloorMap, LaterFloorResolutionRoundTripController)


class PackedOwnedLaterFloorMemory(LaterResolvedFloorMemory):
    def __init__(self, *, identity):
        super().__init__(identity=identity)
        self.index = PackedOwnedMeasuredSampleBoundsIndex()
        self.auxiliary_index = PackedOwnedMeasuredSampleBoundsIndex()
        for partition in (self.partition, self.auxiliary_partition, self.confirmed_auxiliary_partition):
            partition.floor = PackedOwnedMeasuredSampleBoundsIndex()
            partition.other = PackedOwnedMeasuredSampleBoundsIndex()


class PackedOwnedLaterFloorMap(LaterResolvedFloorMap):
    def __init__(self, *, identity=(0,0,0)):
        super().__init__(identity=identity)
        self.surface = PackedOwnedLaterFloorMemory(identity=identity)


class PackedOwnedLaterFloorController(LaterFloorResolutionRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = PackedOwnedLaterFloorMap(identity=(0,0,0))
        self.memory = self.mapper.surface
