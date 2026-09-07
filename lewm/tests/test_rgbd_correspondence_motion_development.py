from copy import deepcopy

import cv2
import numpy as np
import pytest

from lewm.causal_depth_observation_development import FOCAL,from_native_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_correspondence_motion_development import lift,project,solve_translation,track,RGBDCorrespondenceMotion
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_correlated_moment_sensitivity_development import stream


def texture():
    rng=np.random.default_rng(604)
    image=cv2.GaussianBlur(rng.integers(0,256,(480,640),dtype=np.uint8),(5,5),1.)
    return np.repeat(image[:,:,None],3,axis=2)


def packets(images):
    for (_,p,f,_),image in zip(stream(len(images)),images,strict=True):
        p['image']['rgb']=image.copy();now=p['sensor_state']['decision_ns']
        d=from_native_depth(np.full((480,640),2.,np.float32),p,measured_ns=now,available_ns=now,now_ns=now)
        yield p,d,f,now


def test_planar_texture_tracks_tangent_motion_even_with_zero_target():
    image=texture(); moved=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,0]]),(640,480))
    c=RGBDCorrespondenceMotion();a,b=list(packets([image,moved]))
    assert c.observe(*a[:3],now_ns=a[3])['motion']['status']=='INITIAL_RGBD_ANCHOR'
    row=c.observe(*b[:3],now_ns=b[3]);motion=row['motion']
    assert motion['conditional_point_correspondence_rank']==3 and motion['inliers']>=12
    np.testing.assert_allclose(motion['translation_previous_body_m'],[0.,6/FOCAL,0.],atol=.001)
    assert not row['plane_depth_rank_modified'] and not row['native_pose_input']
    assert np.all(b[0]['sensor_state']['control']['applied_command']['values']==0)


@pytest.mark.parametrize('kind',['blank','repeated','unrelated','occluded'])
def test_uninformative_or_inconsistent_images_do_not_invent_translation(kind):
    image=texture(); other=image.copy()
    if kind=='blank':image[:]=128;other[:]=128
    if kind=='repeated':
        checker=((np.indices((480,640))//20).sum(0)%2*255).astype(np.uint8)
        image=np.repeat(checker[:,:,None],3,2);other=np.roll(image,3,axis=1)
    if kind=='unrelated':other=np.flip(image,axis=0).copy()
    if kind=='occluded':other[:]=0
    a,b=list(packets([image,other]))
    row=track(image,other,a[1],b[1],np.eye(3))
    assert row['translation_previous_body_m'] is None and row['conditional_point_correspondence_rank']==0


def test_lifting_uses_pixel_centres_and_rejects_unknown_and_discontinuous_depth():
    p,d,_,_=next(packets([texture()]))
    uv=np.array([[320.,240.],[319.5,239.5],[40.5,20.5]])
    points,valid=lift(d,uv);assert valid.all()
    np.testing.assert_allclose(points[1],[2.326,0.,.043])
    projected,good=project(points);np.testing.assert_allclose(projected,uv,atol=1e-12);assert good.all()
    d['depth_m'][20,40]=3.
    assert not lift(d,uv)[1][-1]
    d['depth_m'][20,40]=0.;d['valid'][20,40]=False
    assert not lift(d,uv)[1][-1]


def point_fixture():
    _,d,_,_=next(packets([texture()]))
    uv=np.array([[x,y] for x in (80.,240.,400.,560.) for y in (60.,160.,260.,360.)])
    a,_=lift(d,uv);R=rotation_increment([.01,-.02,.03]);t=np.array([.01,.02,-.003])
    b=(a-t)@R;uc,_=project(b)
    return a,b,uv,uc,R,t


def test_known_rotation_translation_preserves_camera_lever_arm_conventions():
    a,b,up,uc,R,t=point_fixture();row=solve_translation(a,b,up,uc,R)
    np.testing.assert_allclose(row['translation_previous_body_m'],t,atol=1e-12)


@pytest.mark.parametrize('fault',['outliers','reprojection','coverage','rotation'])
def test_point_solver_rejects_wrong_or_insufficient_constraints(fault):
    a,b,up,uc,R,_=point_fixture()
    if fault=='outliers':b[::2]+=[.1,.2,.3]
    if fault=='reprojection':uc+=10
    if fault=='coverage':up[:]=[80,80];uc[:]=[80,80]
    if fault=='rotation':R*=2
    if fault=='rotation':
        with pytest.raises(SensorContractError):solve_translation(a,b,up,uc,R)
    else:assert solve_translation(a,b,up,uc,R)['translation_previous_body_m'] is None


@pytest.mark.parametrize('fault',['clock','depth_clock','camera','episode','image_binding','privileged'])
def test_packet_failures_latch_without_reinitializing(fault):
    c=RGBDCorrespondenceMotion();a,b=list(packets([texture(),texture()]))
    c.observe(*a[:3],now_ns=a[3]);p,d,f,now=deepcopy(b)
    if fault=='clock':now+=100_000_000
    if fault=='depth_clock':d['available_ns']=now+1
    if fault=='camera':d['calibration_id']='wrong'
    if fault=='episode':d['identity']=(0,0,99)
    if fault=='image_binding':p['image']['rgb'][0,0]=0
    if fault=='privileged':p['native_pose']=[0,0,0]
    with pytest.raises(SensorContractError):c.observe(p,d,f,now_ns=now)
    with pytest.raises(SensorContractError):c.observe(*b[:3],now_ns=b[3])
