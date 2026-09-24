from copy import deepcopy

import cv2
import numpy as np
import pytest

from lewm.causal_depth_observation_development import FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_correspondence_motion_development import RGBDCorrespondenceMotion
from lewm.rgbd_inertial_fusion_development import (
    PointFusionHypotheses,complementary_constraints,ComplementaryRGBDIntegrator,RGBDInertialState)
from lewm.rgbd_inertial_ray_memory_development import RGBDInertialRayMemory, _SingleUseFusion
from lewm.setup_velocity_prior_development import SetupVelocityPrior,SetupVelocityIntegrator
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_depth_inertial_fusion_development import state,frame,Stream
from lewm.tests.test_rgbd_correspondence_motion_development import texture,packets


def prior(epoch=1_600_000_000):
    return SetupVelocityPrior((0,0,0),epoch,(0.,0.,0.),.02,'a'*64)


def hypotheses():
    # Synthetic accounting example, not a runtime error-model selection.
    return PointFusionHypotheses(.002,.01)


def point_for(relative, previous_rgb, delta=None):
    surface=relative['local_surfaces']; surface['depth_sha256']='d'*64
    anchor=relative['motion'] is None
    motion=dict(status='INITIAL_RGBD_ANCHOR' if anchor else ('INSUFFICIENT_POINT_SUPPORT' if delta is None else 'CONDITIONAL_RGBD_POINT_TRANSLATION'),
        translation_previous_body_m=None if delta is None else list(delta),conditional_point_correspondence_rank=0 if delta is None else 3)
    return dict(measured_ns=relative['measured_ns'],identity=list(surface['identity']),rgb_sha256=surface['rgb_sha256'],
        depth_sha256=surface['depth_sha256'],previous_rgb_sha256=previous_rgb,relative_orientation=deepcopy(relative['relative_orientation']),
        motion=motion,static_point_correspondences_assumed=True,native_pose_input=False,plane_depth_rank_modified=False,navigation_qualified=False)


def initialized():
    stream=Stream(); model=ComplementaryRGBDIntegrator(prior(),hypotheses())
    p=frame(stream,0); s=state(p); point=point_for(s,None)
    row=model.observe(p,s,point)
    assert row['position_initial_body_m']==[0.,0.,0.]
    return stream,model


@pytest.mark.parametrize('weak', [[[1,0,0]],[[0,1,0]],[[0,0,1]],[[1,0,0],[0,1,0]],[]])
def test_only_missing_directions_are_filled_and_raw_rank_preserved(weak):
    stream,model=initialized(); delta=np.array([.01,.02,-.003])
    for tick in range(1,6):
        p=frame(stream,tick); s=state(p,delta,weak)
        point=point_for(s,model.previous_rgb,delta); before=deepcopy((s,point))
        row=model.observe(p,s,point)
        assert (s,point)==before
        assert row['depth_rank']==3-len(weak)
        np.testing.assert_allclose(row['position_initial_body_m'],tick*delta,atol=1e-12)
        assert row['constraints']['point_used_for_weak_directions']==bool(weak)
        assert not row['independent_measurements_assumed'] and not row['navigation_qualified']
        assert row['point_position_scale_m']==pytest.approx(tick*.002 if weak else 0.)
        assert row['initial_velocity_prior_transport']['velocity_radius_m_s']==0.


def test_observed_components_are_not_averaged_with_correlated_points():
    p=frame(Stream(),0); s=state(p,[.01,.02,.003],[[0,1,0]])
    point=point_for(s,None,[.014,.023,.003])
    row=complementary_constraints(s['motion'],point['motion'],hypotheses())
    np.testing.assert_allclose(row['projected_previous_body_m'],[.01,.023,.003])
    assert row['observed_subspace_disagreement_m']==pytest.approx(.004)


def test_point_velocity_error_propagates_through_dropout_and_past_error_never_resets():
    stream,model=initialized(); scales=[]; point_scales=[]; prior_scales=[]
    for tick in range(1,8):
        delta=[.01,0,0]; weak=[] if tick==7 else [[1,0,0]]
        p=frame(stream,tick); s=state(p,delta,weak)
        point=point_for(s,model.previous_rgb,delta if tick in (2,6) else None)
        row=model.observe(p,s,point)
        scales.append(row['position_error_scale_m']); point_scales.append(row['point_position_scale_m'])
        prior_scales.append(row['initial_velocity_prior_transport']['position_radius_m'])
        if tick==1: assert row['translation_previous_body_m']==[0.,0.,0.]  # supplied prior, not command/delta truth
        if tick in (3,4,5):assert row['kind']=='INERTIALLY_PREDICTED_WEAK_COMPONENT'
    np.testing.assert_allclose(point_scales,[0,.002,.004,.006,.008,.010,.010],atol=1e-12)
    assert np.all(np.diff(scales)>=0) and np.all(np.diff(prior_scales)>=-1e-12)
    assert prior_scales[-1]>=.002 and row['point_velocity_scale_m_s']==0.
    assert row['consecutive_weak_seconds']==0 and row['depth_rank']==3


def test_rejected_rgb_exactly_preserves_original_setup_integrator_outputs():
    stream=Stream(); old=SetupVelocityIntegrator(prior()); new=ComplementaryRGBDIntegrator(prior(),hypotheses())
    for tick in range(10):
        p=frame(stream,tick); s=state(p,None if not tick else [.01,0,0],[[1,0,0]] if tick>2 else [])
        point=point_for(s,new.previous_rgb); expected=old.observe(p,s); actual=new.observe(p,s,point)
        for key in ('position_initial_body_m','velocity_initial_body_m_s','position_error_scale_m',
                    'initial_velocity_prior_transport','depth_rank','consecutive_weak_seconds','weak_intervals'):
            assert actual[key]==expected[key], key


def test_previous_body_frame_and_rotating_weak_projector():
    stream,model=initialized(); previous=np.eye(3); velocity=np.array([.1,.03,-.01])
    for tick in range(1,7):
        R=rotation_increment([0,0,.2*tick]); delta=previous.T@(.1*velocity)
        p=frame(stream,tick); s=state(p,delta,[[1,0,0]],R)
        row=model.observe(p,s,point_for(s,model.previous_rgb,delta))
        np.testing.assert_allclose(row['position_initial_body_m'],.1*tick*velocity,atol=1e-12)
        previous=R


@pytest.mark.parametrize('fault',['conflict','point_clock','point_identity','previous_rgb','depth_hash','point_orientation',
    'native_flag','extra_privileged','point_status','point_rank','point_nan','plane_rank','plane_basis','plane_projection',
    'clock','force_rewrite','policy_privileged'])
def test_faults_latch(fault):
    stream,model=initialized(); p=frame(stream,1); s=state(p,[.01,.02,0],[[0,1,0]])
    point=point_for(s,model.previous_rgb,[.01,.02,0])
    if fault=='conflict':point['motion']['translation_previous_body_m'][0]+=.03
    elif fault=='point_clock':point['measured_ns']+=1
    elif fault=='point_identity':point['identity']=[0,0,9]
    elif fault=='previous_rgb':point['previous_rgb_sha256']='0'*64
    elif fault=='depth_hash':point['depth_sha256']='0'*64
    elif fault=='point_orientation':point['relative_orientation']['decision_ns']+=1
    elif fault=='native_flag':point['native_pose_input']=True
    elif fault=='extra_privileged':point['native_pose']=[0,0,0]
    elif fault=='point_status':point['motion']['status']='INITIAL_RGBD_ANCHOR'
    elif fault=='point_rank':point['motion']['conditional_point_correspondence_rank']=True
    elif fault=='point_nan':point['motion']['translation_previous_body_m'][0]=np.nan
    elif fault=='plane_rank':s['motion']['rank']=3
    elif fault=='plane_basis':s['motion']['weak_directions_previous_body']=[[0,2,0]]
    elif fault=='plane_projection':s['motion']['observable_projection_previous_body_m'][1]=.01
    elif fault=='clock':s['measured_ns']+=1
    elif fault=='force_rewrite':p['sensor_state']['sensed']['specific_force']['values'][0,0]+=.1
    elif fault=='policy_privileged':p['native_pose']=[0,0,0]
    with pytest.raises(SensorContractError):model.observe(p,s,point)
    assert model.failed
    with pytest.raises(SensorContractError):model.observe(p,s,point)


@pytest.mark.parametrize('values',[(0,.01),(-1,.01),(.01,float('nan')),(True,.01),(.01,float('inf'))])
def test_no_implicit_or_invalid_point_error_assumptions(values):
    with pytest.raises(SensorContractError):PointFusionHypotheses(*values)


def test_live_shared_gyro_exactly_matches_frozen_point_observer():
    image=texture(); images=[cv2.warpAffine(image,np.float32([[1,0,3*i],[0,1,0]]),(640,480)) for i in range(4)]
    data=list(packets(images)); model=RGBDInertialState(prior=prior(data[0][3]),hypotheses=hypotheses())
    reference=RGBDCorrespondenceMotion()
    for tick,(p,d,f,now) in enumerate(data):
        row=model.observe(p,d,f,now_ns=now); expected=reference.observe(p,d,f,now_ns=now)
        assert row['point_state']==expected
        if tick:
            assert row['depth_state']['motion']['rank']==1
            assert row['depth_state']['motion']['translation_previous_body_m'] is None
            np.testing.assert_allclose(row['fusion']['position_initial_body_m'],[0,tick*6/FOCAL,0],atol=.0015)
    assert model.depth.orientation.samples_integrated==150


def test_persistent_ray_memory_consumes_once_and_keeps_raw_rank():
    image=texture(); moved=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,0]]),(640,480))
    data=list(packets([image,moved])); model=RGBDInertialRayMemory(prior=prior(data[0][3]),hypotheses=hypotheses())
    first=model.observe(*data[0][:3],now_ns=data[0][3]); second=model.observe(*data[1][:3],now_ns=data[1][3])
    assert first['fusion']['position_initial_body_m']==[0,0,0]
    assert second['ray_memory']['depth_rank']==1
    assert second['ray_memory']['motion_kind']=='POINT_COMPLEMENTED_PLANE_TRANSLATION'
    assert model._reader.pending is None and model.state.depth.orientation.samples_integrated==50
    assert model.rays.frames[0]['measured_ns']==data[0][3]
    args=(np.array([[1.,0,0]]),np.array([False]))
    a=model.query(*args,now_ns=data[1][3],backend='compiled');b=model.query(*args,now_ns=data[1][3],backend='reference')
    for key in ('free','observed_ground_support','contradictory_or_near_surface','unknown_or_blocked'):
        np.testing.assert_array_equal(a[key],b[key])
    with pytest.raises(SensorContractError):model.observe(*data[1][:3],now_ns=data[1][3])
    assert model.failed
    with pytest.raises(SensorContractError):model.query(*args,now_ns=data[1][3])


def test_single_use_reader_rejects_replacement_wrong_object_and_reconsumption():
    reader=_SingleUseFusion(); p={}; s={}; value={'position':[1,2,3]}
    reader.stage(p,s,value,np.array([0,0,9.81])); value['position'][0]=9
    with pytest.raises(SensorContractError):reader.stage(p,s,{},np.zeros(3))
    assert reader.observe(p,s)=={'position':[1,2,3]}
    with pytest.raises(SensorContractError):reader.observe(p,s)
    reader.stage(p,s,value,np.array([0,0,9.81]))
    with pytest.raises(SensorContractError):reader.observe({},s)


def test_repeated_local_success_cannot_erase_accumulated_budget_exhaustion():
    stream,model=initialized(); scales=[]
    for tick in range(1,46):
        p=frame(stream,tick); s=state(p,[.001,0,0],[[1,0,0]])
        row=model.observe(p,s,point_for(s,model.previous_rgb,[.001,0,0]))
        scales.append(row['position_error_scale_m'])
    assert np.all(np.diff(scales)>0)
    assert row['point_position_scale_m']==pytest.approx(.09)
    assert not row['usable_under_declared_proxy_budget']
    assert row['constraints']['conditional_combined_rank']==3


def test_memory_budget_failure_latches_even_when_point_tracking_succeeds():
    image=texture(); moved=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,0]]),(640,480))
    data=list(packets([image,moved]))
    model=RGBDInertialRayMemory(prior=prior(data[0][3]),hypotheses=PointFusionHypotheses(.1,.01))
    model.observe(*data[0][:3],now_ns=data[0][3])
    with pytest.raises(SensorContractError):model.observe(*data[1][:3],now_ns=data[1][3])
    assert model.failed and model.rays.failed and model._reader.pending is None
    assert model.state.integrator.point_position_scale==.1
    assert model.rays.last_ns==data[0][3]  # unadmitted view was never stored


@pytest.mark.parametrize('fault',['depth_mutation','image_mutation','fast_gyro_mutation','repeat_packet'])
def test_live_sensor_faults_latch_both_state_and_memory(fault):
    data=list(packets([texture(),texture()]))
    model=RGBDInertialRayMemory(prior=prior(data[0][3]),hypotheses=hypotheses())
    model.observe(*data[0][:3],now_ns=data[0][3]); p,d,f,now=deepcopy(data[1])
    if fault=='depth_mutation':d['depth_m'][0,0]=6.
    elif fault=='image_mutation':p['image']['rgb'][0,0]=[0,0,0]
    elif fault=='fast_gyro_mutation':f['values'][0,0]+=.1
    elif fault=='repeat_packet':p,d,f,now=data[0]
    with pytest.raises(SensorContractError):model.observe(p,d,f,now_ns=now)
    assert model.failed and model.rays.failed and model.state.failed
    with pytest.raises(SensorContractError):model.observe(*data[1][:3],now_ns=data[1][3])
