"""Explicit empty-state wiring of the previously tested batched bound insertion.

No collision, support, target, planning or execution semantics change. Native
adoption requires exact full-controller replay and measured whole-loop timing.
"""
from lewm.batched_sample_bounds_development import BatchedMeasuredSampleBoundsIndex
from lewm.observed_floor_contact_development import ObservedFloorContactMemory,ObservedFloorContactMap
from lewm.exact_mission_target_goal_probe_development import ExactMissionTargetGoalProbe


class BatchedObservedFloorContactMemory(ObservedFloorContactMemory):
    def __init__(self,*,identity):
        super().__init__(identity=identity)
        self.index=BatchedMeasuredSampleBoundsIndex()
        self.partition.floor=BatchedMeasuredSampleBoundsIndex()
        self.partition.other=BatchedMeasuredSampleBoundsIndex()
        self.auxiliary_index=BatchedMeasuredSampleBoundsIndex()
        self.auxiliary_partition.floor=BatchedMeasuredSampleBoundsIndex()
        self.auxiliary_partition.other=BatchedMeasuredSampleBoundsIndex()


class BatchedObservedFloorContactMap(ObservedFloorContactMap):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity)
        self.surface=BatchedObservedFloorContactMemory(identity=identity)


class BatchedExactMissionTargetGoalProbe(ExactMissionTargetGoalProbe):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mapper=BatchedObservedFloorContactMap(identity=(0,0,0));self.memory=self.mapper.surface
