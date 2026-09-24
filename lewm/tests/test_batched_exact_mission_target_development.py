from lewm.batched_sample_bounds_development import BatchedMeasuredSampleBoundsIndex
from lewm.batched_exact_mission_target_development import BatchedObservedFloorContactMemory,BatchedObservedFloorContactMap,BatchedExactMissionTargetGoalProbe
from lewm.observed_floor_contact_development import ObservedFloorContactMemory,ObservedFloorContactMap
from lewm.exact_mission_target_goal_probe_development import ExactMissionTargetGoalProbe
from lewm.tests.test_observed_floor_contact_development import memory,insert,Geometry


def test_exact_floor_other_and_nonfoot_queries_after_explicit_empty_state_wiring():
    old,now,w=memory(ObservedFloorContactMemory);new,_,_=memory(BatchedObservedFloorContactMemory)
    for stream,floor in (('primary',True),('auxiliary',True),('primary',False),('auxiliary',False)):
        for m in (old,new):insert(m,stream,floor,w)
        for geometry in (Geometry(),Geometry('base:0')):
            assert old.footprint(geometry,[0.,0.],0.,now_ns=now)==new.footprint(geometry,[0.,0.],0.,now_ns=now)
    indices=[new.index,new.partition.floor,new.partition.other,new.auxiliary_index,
        new.auxiliary_partition.floor,new.auxiliary_partition.other]
    assert all(type(i) is BatchedMeasuredSampleBoundsIndex for i in indices)
    assert len({id(i) for i in indices})==6


def test_sensor_mapping_targets_constraints_and_execution_remain_inherited():
    for name in ('observe','classify_current','observe_auxiliary','footprint'):
        assert getattr(BatchedObservedFloorContactMemory,name) is getattr(ObservedFloorContactMemory,name)
    assert BatchedObservedFloorContactMap.observe is ObservedFloorContactMap.observe
    for name in ('observe','advance','_result'):
        assert getattr(BatchedExactMissionTargetGoalProbe,name) is getattr(ExactMissionTargetGoalProbe,name)
    a=BatchedExactMissionTargetGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    b=ExactMissionTargetGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    assert type(a.selector) is type(b.selector) and type(a.motion) is type(b.motion)
    assert a.memory is a.mapper.surface and not a.memory.index.cells
