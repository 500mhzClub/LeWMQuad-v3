"""Fallback scope, unchanged continuity gates, and sensor-failure latching."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose
from lewm.multi_reference_rgbd_pose_development import Reference
from lewm.tests.test_direct_corner_flow_association_development import frames
from lewm import joint_temporal_anchor_continuity_development as primary
from lewm import dual_camera_anchor_pose_development as dual


def empty(*args):
    return np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),np.empty((0,2))


def observer(monkeypatch, *, bridge_limit=False):
    ref,cur=frames()
    views=dict(primary=ref,auxiliary=ref); current=dict(primary=cur,auxiliary=cur)
    model=DirectFlowDualCameraAnchorPose()
    model.frame=2 if bridge_limit else 1
    anchor=Reference(0,1_500_000_000,views,np.eye(3),np.eye(3),np.zeros(3))
    previous=Reference(model.frame-1,1_500_000_000+(model.frame-1)*100_000_000,
        views,np.eye(3),np.eye(3),np.zeros(3))
    model.references=[anchor]; model.previous=previous
    if bridge_limit: model.bridge_frames=10
    monkeypatch.setattr(primary,'matched_points',empty)
    monkeypatch.setattr(dual,'matched_points',empty)
    return model,current,1_500_000_000+model.frame*100_000_000


def test_original_two_camera_success_is_returned_without_flow(monkeypatch):
    result=object(); model=DirectFlowDualCameraAnchorPose()
    monkeypatch.setattr(DualCameraAnchorPose,'_measure',lambda *a:result)
    assert model._measure({},np.eye(3),1_600_000_000) is result
    assert model.last_direct_flow_fallback is None and not model.direct_flow_mode


def test_missing_original_pairs_can_pass_full_existing_anchor_continuity(monkeypatch):
    model,current,now=observer(monkeypatch)
    refs=list(model.references); previous=model.previous
    candidate,alternative,bridge=model._measure(current,np.eye(3),now)
    assert candidate['registration']['inliers']>=40 and not alternative and not bridge
    assert model.last_continuity['status']=='ANCHOR_MEASUREMENT'
    assert model.last_continuity['same_reference_measurement_reused']
    assert model.last_continuity['incremental_available'] and model.last_continuity['anchor_available']
    assert len(model.last_continuity['rotation_measurement_witnesses'])==1
    assert model.last_direct_flow_fallback['accepted'] and model.last_direct_flow_fallback['selected_camera']=='primary'
    assert model.last_direct_flow_fallback['original_camera_selection']['primary_continuity']['status']=='NO_CURRENT_MEASURED_TRANSLATION'
    assert model.last_direct_flow_fallback['original_auxiliary_continuity']['status']=='NO_CURRENT_MEASURED_TRANSLATION'
    assert model.references==refs and model.previous is previous and model.bridge_frames==0
    assert not model.direct_flow_mode and model.direct_flow_now is None


def test_no_recent_retained_anchor_does_not_bypass_bridge_budget(monkeypatch):
    model,current,now=observer(monkeypatch,bridge_limit=True)
    with pytest.raises(SensorContractError,match='bounded measured bridge exhausted'):
        model._measure(current,np.eye(3),now)
    assert model.bridge_frames==10 and model.total_bridge_frames==0
    assert model.last_continuity['status']=='MEASURED_BRIDGE_BUDGET_EXHAUSTED'
    assert not model.last_direct_flow_fallback['accepted'] and not model.direct_flow_mode


@pytest.mark.parametrize('status',['ANCHOR_CONFLICT_OR_INVALID','ANCHOR_INCREMENT_CONFLICT',
    'CROSS_CAMERA_MEASUREMENT_CONFLICT','VALIDATING_CURRENT_INPUT'])
def test_original_conflicts_and_invalid_states_never_enter_fallback(monkeypatch,status):
    model=DirectFlowDualCameraAnchorPose(); calls=[]
    def original(*args):
        calls.append(True)
        model.last_camera_selection=dict(auxiliary_attempted=True,primary_continuity={'status':'NO_CURRENT_MEASURED_TRANSLATION'})
        model.last_continuity={'status':status}
        raise SensorContractError('original conflict')
    monkeypatch.setattr(DualCameraAnchorPose,'_measure',original)
    with pytest.raises(SensorContractError,match='original conflict'): model._measure({},np.eye(3),1_600_000_000)
    assert len(calls)==1 and model.last_direct_flow_fallback is None and not model.direct_flow_mode


@pytest.mark.parametrize('conflict',['translation','rotation',None])
def test_qualified_original_measurements_remain_conflict_vetoes(monkeypatch,conflict):
    model=DirectFlowDualCameraAnchorPose(); model.frame=1; calls=[]
    witness=dict(position_initial_body_m=[.03 if conflict=='translation' else 0.,0.,0.],
        composed_rotation_initial_body_from_current_body=np.eye(3).tolist())
    if conflict=='rotation':
        c,s=np.cos(.11),np.sin(.11)
        witness['composed_rotation_initial_body_from_current_body']=[[c,-s,0.],[s,c,0.],[0.,0.,1.]]
    candidate=dict(p=np.zeros(3),R=np.eye(3),reference=SimpleNamespace(frame=0))
    def measure(*args):
        calls.append(model.direct_flow_mode)
        if not model.direct_flow_mode:
            model.last_camera_selection=dict(auxiliary_attempted=True,primary_continuity=dict(
                status='MEASURED_BRIDGE_BUDGET_EXHAUSTED',rotation_measurement_witnesses=[deepcopy(witness)]))
            model.last_continuity=dict(status='NO_CURRENT_MEASURED_TRANSLATION',rotation_measurement_witnesses=[])
            model.last_selection={}
            raise SensorContractError('missing usable pose')
        model.last_camera_selection={}; model.last_continuity={'status':'ANCHOR_MEASUREMENT'}
        return candidate,False,False
    monkeypatch.setattr(DualCameraAnchorPose,'_measure',measure)
    if conflict is None:
        assert model._measure({},np.eye(3),1_600_000_000)[0] is candidate
        assert model.last_direct_flow_fallback['accepted']
    else:
        with pytest.raises(SensorContractError,match='qualified original measurement'):
            model._measure({},np.eye(3),1_600_000_000)
        assert model.last_continuity['status']=='DIRECT_FLOW_ORIGINAL_MEASUREMENT_CONFLICT'
        assert not model.last_direct_flow_fallback['accepted']
    assert calls==[False,True] and model.last_direct_flow_fallback['original_qualified_measurements_checked']==1
    assert not model.direct_flow_mode


def test_new_controller_keeps_mapping_mission_and_failure_latching():
    from lewm.direct_flow_floor_transport_controller_development import DirectFlowFloorTransportController
    from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
    assert DirectFlowFloorTransportController.observe is MeasuredFloorTransportController.observe
    assert DirectFlowFloorTransportController.advance is MeasuredFloorTransportController.advance
    c=DirectFlowFloorTransportController(object(),object(),public_mission=dict(
        goal_initial_body_xy_m=[.2,0.],return_initial_body_xy_m=[0.,0.],require_return_after_goal=True),
        navigation_ticks=40,condition='jepa',variant='full',persistent=True)
    r=c.observe({}, {}, {},now_ns=1)
    assert r['terminal']=='SENSOR_OR_MODEL_FAILURE' and r['requested_command']==[0.,0.,0.]
    assert c.advance({}, {},now_ns=2)['terminal']==r['terminal']
