import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.auxiliary_downward45_floor_map_development import AuxiliaryDownward45SurfaceMemory,AuxiliaryDownward45FloorMap
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical,reference_pose
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth,FOCAL
from lewm.tests.test_causal_depth_observation_development import frame


def floor_depth():
    yy,xx=np.indices((480,640));ray=np.stack(((xx+.5-320)/FOCAL,(yy+.5-240)/FOCAL,np.ones_like(xx)),axis=-1)
    E=body_from_optical();direction=ray@E[:3,:3].T
    native=(-.32-E[2,3])/direction[...,2]
    valid=np.isfinite(native)&(native>=.2)&(native<=5.)
    return np.where(valid,native,0.).astype(np.float32)


class Geometry:
    _shapes=[dict(shape_id='FL_foot:0',kind='sphere',dimensions=[.022])]
    def supports(self,q,R):
        return dict(shapes=[dict(shape_id='FL_foot:0',center_body_m=[.8,0.,-.32],
            lower=[.778,-.022,-.342],upper=[.822,.022,-.298])])


def memory():
    policy,_,now=frame();m=AuxiliaryDownward45SurfaceMemory(identity=(0,0,0));w=dict(frame=0,measured_ns=now)
    m.position=np.zeros(3);m.rotation=np.eye(3);m.joints=np.zeros(12);m.last_ns=m.classified_ns=now
    m.map_from_initial=np.eye(3);m.floor_cells={};m.route=[w]
    point=[[.8,0.,-.32]];m.index.insert(point,w);m.partition.insert(point,np.array([True]),w)
    m.patches.append(np.zeros((480,640),np.float32),np.zeros((480,640),bool),np.eye(3),np.zeros(3),-.32,w)
    depth=from_native_depth(floor_depth(),policy,measured_ns=now,available_ns=now,now_ns=now)
    return m,policy,depth,now,w


def test_all_auxiliary_returns_are_classified_and_whole_floor_contact_is_supported():
    m,policy,depth,now,w=memory();occupied={}
    receipt=m.observe_auxiliary(policy,depth,np.eye(3),-.32,m.floor_cells,occupied,now_ns=now)
    assert receipt['current_returns']==receipt['current_floor_returns']+receipt['current_other_returns']
    assert receipt['total_returns']==sum(m.auxiliary_index.sample_counts.values())
    assert m.floor_cells and not occupied
    result=m.footprint(Geometry(),[0.,0.],0.,now_ns=now)
    assert not result['possible_intersection'] and result['all_auxiliary_sampled_returns_retained']
    assert result['auxiliary_foot_floor_contacts'][0]['auxiliary_patch']['complete_nominal_foot_patch']
    assert not result['ground_support_approved']
    with pytest.raises(SensorContractError):m.observe_auxiliary(policy,depth,np.eye(3),-.32,m.floor_cells,occupied,now_ns=now)


def test_primary_or_auxiliary_unknown_return_cannot_be_waived_by_complete_floor():
    for source in ('primary','auxiliary'):
        m,p,d,now,w=memory();m.observe_auxiliary(p,d,np.eye(3),-.32,m.floor_cells,{},now_ns=now)
        index,partition=(m.index,m.partition) if source=='primary' else (m.auxiliary_index,m.auxiliary_partition)
        point=[[.8,0.,-.32]];index.insert(point,w);partition.insert(point,np.array([False]),w)
        result=m.footprint(Geometry(),[0.,0.],0.,now_ns=now)
        assert result['possible_intersection']
        assert result[source+'_possible_intersection']


def test_auxiliary_obstacle_is_retained_in_surface_and_route_map():
    m,p,d,now,w=memory()
    native=np.full((480,640),.5,np.float32)
    d=from_native_depth(native,p,measured_ns=now,available_ns=now,now_ns=now);occupied={}
    r=m.observe_auxiliary(p,d,np.eye(3),-.32,m.floor_cells,occupied,now_ns=now)
    assert occupied and r['other_returns']>0 and r['total_returns']==19200
    assert not r['unknown_rays_inferred_free']


def test_missing_auxiliary_or_invalid_input_latches_full_map_failure():
    m,_,_,now,_=memory()
    with pytest.raises(SensorContractError,match='current auxiliary'):m.footprint(Geometry(),[0.,0.],0.,now_ns=now)
    mapper=AuxiliaryDownward45FloorMap()
    with pytest.raises(SensorContractError):mapper.observe({}, {}, {},auxiliary_depth={},now_ns=1)
    assert mapper.failed and mapper.surface.failed

