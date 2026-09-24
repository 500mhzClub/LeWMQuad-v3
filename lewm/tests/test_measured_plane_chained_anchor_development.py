"""Combined actual image fits, plane constraints and failure accounting."""
from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm import measured_plane_chained_anchor_development as new
from lewm import measured_plane_dual_camera_pose_development as plane
from lewm import joint_temporal_anchor_continuity_development as primary
from lewm import dual_camera_anchor_pose_development as dual
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence, run


def empty(*args):
    return np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),np.empty((0,2))


def test_original_anchor_packets_and_public_evidence_remain_exact():
    original=new.MeasuredPlaneVisualMotion(); candidate=new.MeasuredPlaneChainedAnchorVisualMotion()
    for item in sequence():
        a=run(original,item); b=run(candidate,item)
        assert a==b
        assert candidate.model.last_chained_anchor_fallback is None
        assert candidate.model.last_direct_flow_fallback is None
        assert set(candidate.model._planes)=={r.frame for r in candidate.model.references}|{candidate.model.previous.frame}
    assert len(candidate.model._image_history)==4


@pytest.mark.parametrize('camera',['primary','auxiliary'])
def test_actual_chained_fit_receives_measured_plane_refinement(monkeypatch,camera):
    model=new.MeasuredPlaneChainedAnchorVisualMotion()
    items=list(sequence(3,blank_primary_at=(0,1,2) if camera=='auxiliary' else ()))
    initial=run(model,items[0]); assert initial['current_pose'] is not None
    reference=model.model.references[0]
    # Remove descriptor associations only. Flow, endpoint geometry, plane fit,
    # temporal admission and public pose witness checking are actual code.
    monkeypatch.setattr(primary,'matched_points',empty)
    monkeypatch.setattr(dual,'matched_points',empty)
    first=run(model,items[1])
    assert first['terminal_failure'] is None, first['terminal_failure']
    # At frame one the retained anchor is also the immediately previous frame,
    # so the existing one-interval flow fallback can measure that anchor.
    assert first['continuity_evidence']['status']=='ANCHOR_MEASUREMENT'
    row=run(model,items[2])
    assert row['terminal_failure'] is None, row['terminal_failure']
    assert row['continuity_evidence']['status']=='ANCHOR_MEASUREMENT'
    assert row['camera_selection']['selected_camera']==camera
    receipt=row['chained_anchor_fallback']
    assert receipt['accepted'] and receipt['original_qualified_measurements_checked']>0
    assert receipt['bridge_budget_unchanged'] and receipt['pose_increments_composed'] is False
    refinement=row['measured_plane_selected_pair']
    assert refinement['applied'] and refinement['original_inliers_preserved']
    assert refinement['reference_floor']['frame']==reference.frame
    assert all(w['measured_plane_constrained'] for w in row['continuity_evidence']['rotation_measurement_witnesses'])
    p,R,_=current_joint_pose(row,identity=(0,0,0),now_ns=items[2][3]['now_ns'])
    np.testing.assert_allclose(p,0,atol=1e-9)
    np.testing.assert_allclose(R,np.eye(3),atol=1e-9)
    assert model.model.bridge_frames==0 and model.model.total_bridge_frames==0
    assert model.model.references[0] is reference
    # Returned fallback details must not alias the observer's retained receipt.
    frozen=deepcopy(model.model.last_chained_anchor_fallback)
    row['chained_anchor_fallback']['accepted']=False
    assert model.model.last_chained_anchor_fallback==frozen


def test_missing_image_chain_preserves_all_ten_bridges_then_terminal(monkeypatch):
    motion=new.MeasuredPlaneChainedAnchorVisualMotion()
    items=list(sequence(13));run(motion,items[0]);model=motion.model
    reference=model.references[0]
    monkeypatch.setattr(primary,'matched_points',empty)
    monkeypatch.setattr(dual,'matched_points',empty)
    assert run(motion,items[1])['continuity_evidence']['status']=='ANCHOR_MEASUREMENT'
    for frame in range(2,12):
        model._image_history.clear()
        row=run(motion,items[frame])
        assert row['terminal_failure'] is None, row['terminal_failure']
        assert row['continuity_evidence']['status']=='MEASURED_INCREMENT_BRIDGE'
        assert model.bridge_frames==model.total_bridge_frames==frame-1
        assert model.references==[reference] and model.references[0] is reference
        assert set(model._planes)=={0,frame}
    model._image_history.clear()
    row=run(motion,items[12])
    assert row['status']=='VISUAL_TERMINAL_FAILURE'
    assert model.bridge_frames==10 and model.total_bridge_frames==10
    assert model.last_continuity['status']=='MEASURED_BRIDGE_BUDGET_EXHAUSTED'
    assert model.previous.frame==11 and model.failed


@pytest.mark.parametrize('when',['original','chained'])
def test_plane_image_conflict_cannot_be_reclassified_by_chain_fallback(monkeypatch,when):
    motion=new.MeasuredPlaneChainedAnchorVisualMotion();items=list(sequence(3))
    run(motion,items[0]);model=motion.model; original_refine=plane.refine;calls=[]
    if when=='chained':
        monkeypatch.setattr(primary,'matched_points',empty)
        monkeypatch.setattr(dual,'matched_points',empty)
        assert run(motion,items[1])['terminal_failure'] is None
    def conflict(candidate,*args,**kwargs):
        chained='chained_corner_flow_association' in candidate['registration']
        if (when=='chained')==chained:
            calls.append(chained)
            raise plane.PlaneImageConflict('synthetic qualified plane conflict')
        return original_refine(candidate,*args,**kwargs)
    monkeypatch.setattr(plane,'refine',conflict)
    prior=model.previous; references=list(model.references)
    row=run(motion,items[2 if when=='chained' else 1])
    assert row['status']=='VISUAL_TERMINAL_FAILURE' and model.failed
    assert model.last_continuity['status']=='MEASURED_PLANE_IMAGE_CONFLICT'
    assert calls==[when=='chained']
    assert model.previous is prior and model.references==references
    assert not model._chain_mode


def test_rejected_chained_plane_never_publishes_unconstrained_anchor(monkeypatch):
    motion=new.MeasuredPlaneChainedAnchorVisualMotion();items=list(sequence(3))
    run(motion,items[0])
    monkeypatch.setattr(primary,'matched_points',empty)
    monkeypatch.setattr(dual,'matched_points',empty)
    assert run(motion,items[1])['terminal_failure'] is None
    original=plane.refine;rejected=[]
    def refine(candidate,*args,**kwargs):
        if 'chained_corner_flow_association' in candidate['registration']:
            rejected.append(candidate['reference'].frame)
            raise SensorContractError('synthetic plane refinement loses an inlier')
        return original(candidate,*args,**kwargs)
    monkeypatch.setattr(plane,'refine',refine)
    row=run(motion,items[2]);model=motion.model
    assert rejected and row['terminal_failure'] is None
    assert row['continuity_evidence']['status']=='MEASURED_INCREMENT_BRIDGE'
    assert not row['chained_anchor_fallback']['accepted']
    assert row['chained_anchor_fallback']['original_result_restored']
    assert model.bridge_frames==model.total_bridge_frames==1
    assert all(w['reference_frame']==1 and w['measured_plane_constrained']
        for w in row['continuity_evidence']['rotation_measurement_witnesses'])


@pytest.mark.parametrize('fault',['duplicate','blank','clock'])
def test_invalid_current_input_latches_without_reset(fault):
    motion=new.MeasuredPlaneChainedAnchorVisualMotion()
    items=list(sequence(3,blank_both_at=(1,) if fault=='blank' else ()))
    run(motion,items[0]);item=items[0 if fault=='duplicate' else 1]
    if fault=='clock':item[3]['auxiliary_depth']['measured_ns']+=1
    row=run(motion,item)
    assert row['status']=='VISUAL_TERMINAL_FAILURE' and row['current_pose'] is None
    state=motion.model.frame,tuple(motion.model._planes),tuple(motion.model._image_history)
    repeated=run(motion,items[2])
    assert repeated['terminal_failure']==row['terminal_failure']
    assert state==(motion.model.frame,tuple(motion.model._planes),tuple(motion.model._image_history))


def test_cooperative_mro_wraps_every_candidate_and_preserves_original_code():
    cls=new.MeasuredPlaneChainedAnchorPose
    assert cls._candidate is new.MeasuredPlaneDualCameraPose._candidate
    assert cls._measure is new.MeasuredPlaneDualCameraPose._measure
    assert cls.observe is new.MeasuredPlaneDualCameraPose.observe
    assert cls.__mro__.index(new.MeasuredPlaneDualCameraPose)<cls.__mro__.index(new.ChainedAnchorDualCameraPose)
    model=cls()
    assert model._image_history=={} and model._planes=={}
    assert model.bridge_frames==0 and model._chain_mode is False


@pytest.mark.parametrize('condition,variant',[('direct','no_rgb'),('jepa','full')])
def test_complete_controller_first_forecast_unchanged_when_anchor_available(condition,variant):
    import torch
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
    from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    models=[FixedHeadModel(condition),FixedHeadModel(condition)]
    states=[{k:v.clone() for k,v in model.state_dict().items()} for model in models]
    options=dict(public_mission=deepcopy(MISSION),navigation_ticks=40,
        condition=condition,variant=variant,persistent=True)
    controllers=[cls(model,ArticulatedCollisionGeometry(URDF),**options) for cls,model in zip(
        (new.MeasuredPlaneResidualController,new.MeasuredPlaneChainedAnchorController),models,strict=True)]
    for source in sequence():
        rows=[]
        for controller in controllers:
            p,d,f,kwargs=[move_test_origin(deepcopy(v)) for v in source]
            rows.append(controller.observe(p,d,f,**kwargs))
        original,candidate=rows
        assert original['terminal'] is None and candidate['terminal'] is None
        assert candidate['chained_anchor_reacquisition_enabled'] is True
        assert candidate['direct_corner_flow_missingness_fallback_enabled'] is True
        stripped={k:v for k,v in candidate.items() if k not in (
            'chained_anchor_reacquisition_enabled','direct_corner_flow_missingness_fallback_enabled')}
        stripped['controller']=original['controller']
        assert stripped==original
        assert controllers[0].mapper.floor==controllers[1].mapper.floor
        assert controllers[0].mapper.occupied==controllers[1].mapper.occupied
        assert controllers[0].memory.route==controllers[1].memory.route
    assert [len(model.calls) for model in models]==[1,1]
    for model,state in zip(models,states,strict=True):
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
        assert all(parameter.grad is None for parameter in model.parameters())
