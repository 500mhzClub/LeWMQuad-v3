"""Explicit empty-state query optimization; original controller labels/semantics."""
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex
from lewm.later_floor_resolution_controller_development import (
    LaterResolvedFloorMemory, LaterResolvedFloorMap, LaterFloorResolutionRoundTripController)


class SinglePassLaterFloorMemory(LaterResolvedFloorMemory):
    def __init__(self, *, identity):
        super().__init__(identity=identity)
        self.index = SinglePassMeasuredSampleBoundsIndex()
        self.auxiliary_index = SinglePassMeasuredSampleBoundsIndex()
        for partition in (self.partition, self.auxiliary_partition, self.confirmed_auxiliary_partition):
            partition.floor = SinglePassMeasuredSampleBoundsIndex()
            partition.other = SinglePassMeasuredSampleBoundsIndex()


class SinglePassLaterFloorMap(LaterResolvedFloorMap):
    def __init__(self, *, identity=(0,0,0)):
        super().__init__(identity=identity)
        self.surface = SinglePassLaterFloorMemory(identity=identity)


class SinglePassLaterFloorController(LaterFloorResolutionRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = SinglePassLaterFloorMap(identity=(0,0,0))
        self.memory = self.mapper.surface
