import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.observed_floor_contact_development import ObservedFloorContactMemory,ObservedFloorContactMap,ObservedFloorContactGoalProbe
from lewm.auxiliary_downward45_floor_map_development import AuxiliaryDownward45SurfaceMemory,AuxiliaryDownward45FloorMap
from lewm.auxiliary_downward45_goal_probe_development import AuxiliaryDownward45GoalProbe


class Geometry:
    def __init__(self,shape_id='FL_foot:0'):
        self.shape_id=shape_id
        self._shapes=[dict(shape_id=shape_id,kind='sphere',dimensions=[.022])]
        if shape_id=='base:0':self._shapes.append(dict(shape_id='FL_foot:0',kind='sphere',dimensions=[.022]))
    def supports(self,q,R):
        return dict(shapes=[dict(shape_id=s['shape_id'],center_body_m=[.8,0.,-.32],
            lower=[.778,-.022,-.342],upper=[.822,.022,-.298]) for s in self._shapes])


def memory(cls):
    m=cls(identity=(0,0,0));now=1500000000;w=dict(frame=0,measured_ns=now)
    m.position=np.zeros(3);m.rotation=np.eye(3);m.joints=np.zeros(12)
    m.last_ns=m.classified_ns=m.auxiliary_ns=now;m.auxiliary_receipt=dict(frame=0)
    m.map_from_initial=np.eye(3);m.floor_cells={};m.route=[w]
    for patches in (m.patches,m.auxiliary_patches):
        patches.append(np.zeros((480,640),np.float32),np.zeros((480,640),bool),
            np.eye(3),np.zeros(3),-.32,w)
    return m,now,w


def insert(m,stream,floor,w,point=(.8,0.,-.32)):
    index,partition=(m.index,m.partition) if stream=='primary' else (m.auxiliary_index,m.auxiliary_partition)
    index.insert([point],w);partition.insert([point],np.array([floor]),w)


@pytest.mark.parametrize('stream',['primary','auxiliary'])
def test_explicit_policy_change_preserves_unknown_support_and_every_return(stream):
    old,now,w=memory(AuxiliaryDownward45SurfaceMemory)
    new,_,_=memory(ObservedFloorContactMemory)
    assert not old.footprint(Geometry(),[0,0],0,now_ns=now)['possible_intersection']
    assert not new.footprint(Geometry(),[0,0],0,now_ns=now)['possible_intersection']
    for m in (old,new):insert(m,stream,True,w)
    a=old.footprint(Geometry(),[0,0],0,now_ns=now);b=new.footprint(Geometry(),[0,0],0,now_ns=now)
    assert a['possible_intersection'] and b['coverage_required_possible_intersection']
    assert not b['possible_intersection'] and not b['ground_support_approved'] and not b['unobserved_space_certified']
    assert b['observed_ground_contacts'][0]['support_status']=='UNKNOWN'
    assert not b['observed_ground_contacts'][0]['complete_projection_observed']
    assert not new.floor_cells
    for stream_name in ('primary','auxiliary'):
        a_index=old.index if stream_name=='primary' else old.auxiliary_index
        b_index=new.index if stream_name=='primary' else new.auxiliary_index
        assert a_index.sample_counts==b_index.sample_counts
        assert a_index.cells==b_index.cells


@pytest.mark.parametrize('stream',['primary','auxiliary'])
def test_any_unknown_return_still_blocks_even_in_same_floor_voxel(stream):
    m,now,w=memory(ObservedFloorContactMemory)
    for source in ('primary','auxiliary'):insert(m,source,True,w)
    assert not m.footprint(Geometry(),[0,0],0,now_ns=now)['possible_intersection']
    insert(m,stream,False,w)
    r=m.footprint(Geometry(),[0,0],0,now_ns=now)
    assert r['possible_intersection'] and r[stream+'_possible_intersection']
    assert not r['non_floor_or_unknown_contacts_exempted']


@pytest.mark.parametrize('stream',['primary','auxiliary'])
def test_nonfoot_floor_contact_remains_a_collision(stream):
    m,now,w=memory(ObservedFloorContactMemory);insert(m,stream,True,w)
    r=m.footprint(Geometry('base:0'),[0,0],0,now_ns=now)
    assert r['possible_intersection'] and r[stream+'_possible_intersection']
    assert not r['non_foot_contacts_exempted']
    assert all(c['shape_id']!='base:0' for c in r['observed_ground_contacts'])


def test_mapping_and_execution_inherited_and_stale_evidence_still_rejected():
    assert ObservedFloorContactMemory.observe_auxiliary is AuxiliaryDownward45SurfaceMemory.observe_auxiliary
    assert ObservedFloorContactMap.observe is AuxiliaryDownward45FloorMap.observe
    assert ObservedFloorContactGoalProbe.observe is AuxiliaryDownward45GoalProbe.observe
    assert ObservedFloorContactGoalProbe.advance is AuxiliaryDownward45GoalProbe.advance
    m,now,w=memory(ObservedFloorContactMemory)
    with pytest.raises(SensorContractError):m.footprint(Geometry(),[0,0],0,now_ns=now+100000000)
