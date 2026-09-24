"""Causal retention, original conflict vetoes and unchanged raw pair gates."""
from copy import deepcopy
import ast
from pathlib import Path
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose,Reference
from lewm.recent_qualified_anchor_development import RecentQualifiedAnchorPose
from lewm.tests.test_dual_camera_anchor_pose_development import observe,paired
from lewm.tests.test_rgbd_correspondence_motion_development import packets,texture


def setup():
    model=RecentQualifiedAnchorPose()
    model.frame=2;model.recent_query_ns=1_700_000_000
    model.references=[Reference(0,1_500_000_000,{},np.eye(3),np.eye(3),np.zeros(3))]
    model.previous=Reference(1,1_600_000_000,{},np.eye(3),np.eye(3),np.zeros(3))
    model.recent_qualified_reference=model.previous
    model.last_recent_qualified_anchor={'attempts':[]}
    return model


def test_original_success_has_priority_and_reference_population_is_untouched(monkeypatch):
    model=setup(); original=object(); refs=list(model.references)
    monkeypatch.setattr(MultiReferenceRGBDPose,'_choose',lambda *a:original)
    monkeypatch.setattr(model,'_candidate',lambda *a:pytest.fail('extra fit after original success'))
    assert model._choose({},np.eye(3)) is original
    assert model.references==refs and not model.last_recent_qualified_anchor['attempts']


@pytest.mark.parametrize('fault',[None,'conflict','no_reference','different_object','stale_frame',
    'stale_time','missing_query','already_retained','pair_rejected'])
def test_only_causally_retained_qualified_previous_view_can_extend_failed_search(monkeypatch,fault):
    model=setup();calls=[];candidate={'fixture':True};refs=list(model.references)
    if fault=='no_reference':model.recent_qualified_reference=None
    elif fault=='different_object':model.recent_qualified_reference=deepcopy(model.previous)
    elif fault=='stale_frame':model.frame+=1
    elif fault=='stale_time':model.recent_query_ns+=1
    elif fault=='missing_query':model.recent_query_ns=None
    elif fault=='already_retained':model.references.append(model.previous);refs=list(model.references)
    def original(*a):
        calls.append('original');model.last_selection=dict(status='CONFLICTING_ALTERNATIVES' if fault=='conflict' else 'NO_QUALIFIED_REFERENCE',attempts=[])
        raise SensorContractError('original unavailable')
    def fit(ref,current,G):
        assert ref is model.previous;calls.append('fit')
        if fault=='pair_rejected':raise SensorContractError('unchanged rigid gate rejected')
        return candidate
    monkeypatch.setattr(MultiReferenceRGBDPose,'_choose',original)
    monkeypatch.setattr(model,'_candidate',fit)
    if fault is None:
        assert model._choose({},np.eye(3))==(candidate,True)
        assert model.last_selection['selected_reference']==1
        assert model.last_recent_qualified_anchor['attempts'][0]['qualified']
    else:
        with pytest.raises(SensorContractError):model._choose({},np.eye(3))
        if fault=='pair_rejected':
            assert calls==['original','fit']
            assert not model.last_recent_qualified_anchor['attempts'][0]['qualified']
            assert model.last_selection['status']=='NO_QUALIFIED_REFERENCE'
        else:assert calls==['original']
    assert model.references==refs


def test_normal_raw_pose_and_all_original_measurement_receipts_match():
    old=DirectFlowDualCameraAnchorPose();new=RecentQualifiedAnchorPose()
    for i,item in enumerate(packets([texture()]*4)):
        expected=observe(old,item);actual=observe(new,item)
        assert actual==expected and new.last_continuity==old.last_continuity
        assert new.last_selection==old.last_selection and new.last_camera_selection==old.last_camera_selection
        assert new.last_recent_qualified_anchor['retained_current_frame']==i
        assert not new.last_recent_qualified_anchor['attempts']
        assert new.recent_qualified_reference is new.previous


def test_recent_qualified_view_uses_identical_raw_fit_instead_of_first_bridge(monkeypatch):
    old=DirectFlowDualCameraAnchorPose();new=RecentQualifiedAnchorPose()
    items=list(packets([texture()]*3))
    for item in items[:2]:
        assert observe(old,item)==observe(new,item)
    assert [r.frame for r in new.references]==[0] and new.recent_qualified_reference.frame==1
    original=DirectFlowDualCameraAnchorPose._candidate
    def missing_old_anchor(self,ref,current,G):
        if ref.frame==0:raise SensorContractError('synthetic old reference missingness')
        return original(self,ref,current,G)
    monkeypatch.setattr(DirectFlowDualCameraAnchorPose,'_candidate',missing_old_anchor)
    expected=observe(old,items[2]);actual=observe(new,items[2])
    for key in ('position_initial_body_m','rotation_initial_body_from_current_body','registration','reference_frame'):
        assert actual[key]==expected[key]
    assert old.last_continuity['status']=='MEASURED_INCREMENT_BRIDGE' and old.bridge_frames==1
    assert new.last_continuity['status']=='ANCHOR_MEASUREMENT' and new.bridge_frames==0
    assert actual['promoted_keyframe'] and not expected['promoted_keyframe']
    assert new.last_continuity['same_reference_measurement_reused']
    assert len(new.last_continuity['rotation_measurement_witnesses'])==1
    assert new.last_recent_qualified_anchor['attempts'][0]['qualified']
    assert [r.frame for r in new.references]==[0,2]


def test_bridge_is_never_retained_and_ten_frame_limit_still_latches(monkeypatch):
    model=RecentQualifiedAnchorPose();items=list(packets([texture()]*14))
    observe(model,items[0]);observe(model,items[1])
    # Simulate absence of an eligible retained previous anchor at the first gap.
    model.recent_qualified_reference=None
    original=DirectFlowDualCameraAnchorPose._candidate
    def missing_old_anchor(self,ref,current,G):
        if ref.frame==0:raise SensorContractError('synthetic retained reference missingness')
        return original(self,ref,current,G)
    monkeypatch.setattr(DirectFlowDualCameraAnchorPose,'_candidate',missing_old_anchor)
    for i in range(2,12):
        row=observe(model,items[i])
        assert not row['promoted_keyframe'] and model.bridge_frames==i-1
        assert model.recent_qualified_reference is None
        assert model.last_recent_qualified_anchor['retained_current_frame'] is None
    with pytest.raises(SensorContractError):observe(model,items[12])
    assert model.failed and model.bridge_frames==10 and model.recent_qualified_reference is None
    assert model.last_continuity['status']=='MEASURED_BRIDGE_BUDGET_EXHAUSTED'
    with pytest.raises(SensorContractError):observe(model,items[13])


def test_controller_keeps_height_registration_map_residual_and_failure_latching():
    from lewm.recent_qualified_anchor_controller_development import RecentQualifiedAnchorController
    from lewm.partial_floor_height_controller_development import PartialHeightDirectFlowController
    assert RecentQualifiedAnchorController.observe is PartialHeightDirectFlowController.observe
    assert RecentQualifiedAnchorController.advance is PartialHeightDirectFlowController.advance
    kwargs=dict(public_mission=dict(goal_initial_body_xy_m=[.2,0.],return_initial_body_xy_m=[0.,0.],
        require_return_after_goal=True),navigation_ticks=40,condition='jepa',variant='full',persistent=True)
    old=PartialHeightDirectFlowController(object(),object(),**kwargs)
    new=RecentQualifiedAnchorController(object(),object(),**kwargs)
    for key in ('registration','mapper','memory','residual','selector'):assert type(getattr(old,key)) is type(getattr(new,key))
    result=new.observe({}, {}, {},now_ns=1)
    assert result['terminal']=='SENSOR_OR_MODEL_FAILURE' and result['requested_command']==[0.,0.,0.]
    assert new.advance({}, {},now_ns=2)['terminal']==result['terminal']


def test_continuity_calculations_and_original_witness_wrapper_preserved():
    from lewm.recent_qualified_anchor_development import _RecentReferenceSearch
    from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorRGBDPose
    from lewm.temporal_anchor_continuity_development import TemporalAnchorRGBDPose
    def method(path,cls):
        tree=ast.parse(Path(path).read_text())
        c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_measure')
    class Normalize(ast.NodeTransformer):
        def visit_Call(self,node):
            if ast.unparse(node.func)=='self._choose':
                node.func=ast.parse('super()._choose',mode='eval').body
            return self.generic_visit(node)
    old=method('lewm/temporal_anchor_continuity_development.py','TemporalAnchorRGBDPose')
    new=method('lewm/recent_qualified_anchor_development.py','_RecentReferenceSearch')
    assert ast.dump(old)==ast.dump(Normalize().visit(new))
    mro=RecentQualifiedAnchorPose.__mro__
    assert mro.index(JointTemporalAnchorRGBDPose)<mro.index(_RecentReferenceSearch)<mro.index(TemporalAnchorRGBDPose)


def test_additional_reference_passes_existing_public_pose_witness_admission(monkeypatch):
    from lewm.recent_qualified_anchor_controller_development import RecentQualifiedAnchorVisualMotion
    from lewm.dual_camera_visual_motion_development import current_dual_camera_pose
    motion=RecentQualifiedAnchorVisualMotion();items=list(packets([texture()]*3))
    def step(item):
        p,d,f,now,image,aux=paired(item)
        evidence=motion.observe(p,d,f,auxiliary_rgb=image,auxiliary_depth=aux,now_ns=now)
        assert evidence['status']=='CURRENT_VISUAL_POSE' and evidence['terminal_failure'] is None
        current_dual_camera_pose(evidence,p,image,aux,identity=(0,0,0),now_ns=now)
        return evidence
    step(items[0]);step(items[1])
    original=DirectFlowDualCameraAnchorPose._candidate
    def missing_old_anchor(self,ref,current,G):
        if ref.frame==0:raise SensorContractError('synthetic old reference missingness')
        return original(self,ref,current,G)
    monkeypatch.setattr(DirectFlowDualCameraAnchorPose,'_candidate',missing_old_anchor)
    result=step(items[2])
    assert result['continuity_evidence']['status']=='ANCHOR_MEASUREMENT'
    assert result['recent_qualified_anchor']['attempts'][0]['qualified']
    assert result['current_pose']['frame']==2 and result['current_pose']['reference_frame']==1
