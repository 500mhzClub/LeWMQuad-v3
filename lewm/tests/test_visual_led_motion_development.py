from copy import deepcopy
import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.visual_led_motion_development import ContactMotionSample, VisualLedMotion, contact_comparison
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture


def sample(t, v=(0.,0.,0.), **kwargs):
    return ContactMotionSample((0,0,0),'contact-a',t,t,v,tuple(map(tuple,np.eye(3))),**kwargs)


def window(start, missing=()):
    return [sample(start+i*20_000_000,None if i in missing else (0.,0.,0.)) for i in range(6)]


@pytest.mark.parametrize('missing', [(),(0,),(3,),(5,),tuple(range(6))])
def test_contact_dropout_never_changes_visual_pose(missing):
    fixture=list(packets([texture()]*2)); a=VisualLedMotion('gyro',identity=(0,0,0))
    b=VisualLedMotion('gyro',identity=(0,0,0),contact_acquisition_identity='contact-a')
    for p,d,f,now in fixture:
        x=a.observe(p,d,f,now_ns=now)
        y=b.observe(p,d,f,now_ns=now,contact_samples=window(now-100_000_000,missing))
        assert x['current_pose']==y['current_pose'] and x['current_pose'] is not None
        assert y['motion_permission']=='NOT_EVALUATED' and not y['pose_updated_from_contact']
    assert y['contact_diagnostic']['status']==('CONTACT_VELOCITY_UNAVAILABLE' if missing else 'CONTACT_VISUAL_COMPARISON_AVAILABLE')


@pytest.mark.parametrize('fault',['wrong_episode','late','stale','wrong_stream','gap'])
def test_bad_optional_contact_is_explicit_but_visual_survives(fault):
    fixture=list(packets([texture()]*2)); model=VisualLedMotion('joint',identity=(0,0,0),contact_acquisition_identity='contact-a')
    p,d,f,now=fixture[0]; model.observe(p,d,f,now_ns=now)
    p,d,f,now=fixture[1]; contact=window(now-100_000_000)
    if fault=='gap': contact.pop()
    elif fault=='stale': contact=window(now-200_000_000)
    else:
        old=contact[3]
        contact[3]=ContactMotionSample((1,0,0) if fault=='wrong_episode' else old.identity,
            'wrong' if fault=='wrong_stream' else old.acquisition_identity,
            old.measured_ns,now+1 if fault=='late' else old.available_ns,old.velocity_body_m_s,old.rotation_contact_anchor_from_body)
    row=model.observe(p,d,f,now_ns=now,contact_samples=contact)
    assert row['status']=='CURRENT_VISUAL_POSE'
    assert row['contact_diagnostic']['status']=='CONTACT_CONTRACT_REJECTED'


def test_future_queries_do_not_impute_stationary_pose_or_mutate_history():
    p,d,f,now=next(packets([texture()])); model=VisualLedMotion('gyro',identity=(0,0,0))
    row=model.observe(p,d,f,now_ns=now); row['current_pose']['position_initial_body_m'][0]=123
    stale=model.snapshot(now_ns=now+20_000_000)
    assert stale['current_pose'] is None and stale['visual_age_ns']==20_000_000
    assert stale['last_visual']['position_initial_body_m']==[0.,0.,0.]
    assert not stale['command_integration_used']
    with pytest.raises(SensorContractError): model.snapshot(now_ns=now)


@pytest.mark.parametrize('fault',['blank','clock','episode','privileged'])
def test_visual_fault_latches_and_does_not_reinvoke_model(fault):
    import hashlib
    fixture=list(packets([texture()]*3)); model=VisualLedMotion('joint',identity=(0,0,0))
    p,d,f,now=fixture[0]; model.observe(p,d,f,now_ns=now)
    p,d,f,now=deepcopy(fixture[1])
    if fault=='blank': p['image']['rgb'][:]=128; d['rgb_sha256']=hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    elif fault=='clock': now+=1
    elif fault=='episode': p['sensor_state']['identity']=(1,0,0)
    else: p['native_pose']=[0,0,0]
    bad=model.observe(p,d,f,now_ns=now); count=model.model.frame
    p,d,f,now=fixture[2]; after=model.observe(p,d,f,now_ns=now)
    assert bad['status']==after['status']=='VISUAL_TERMINAL_FAILURE'
    assert after['current_pose'] is None and model.model.frame==count
    assert after['terminal_failure']==bad['terminal_failure']


def test_rotating_contact_axes_are_integrated_in_fixed_frame():
    start=1_000_000_000; end=start+100_000_000; contacts=[]
    for i in range(6):
        angle=i*.1; R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        t=start+i*20_000_000
        contacts.append(ContactMotionSample((0,0,0),'contact-a',t,t,tuple(R.T@np.array([1.,0,0.])),tuple(map(tuple,R))))
    before=dict(measured_ns=start,position_initial_body_m=[0.,0.,0.])
    after=dict(measured_ns=end,position_initial_body_m=[.1,0.,0.],rotation_initial_body_from_current_body=R.tolist())
    row=contact_comparison(contacts,identity=(0,0,0),acquisition_identity='contact-a',before=before,after=after,now_ns=end)
    assert row['displacement_disagreement_m']<1e-14 and row['physical_disagreement_bound_m'] is None


@pytest.mark.parametrize('fault',['nan','reflection','backward','calibration'])
def test_typed_contact_preserves_units_clock_and_rotation_contract(fault):
    args=dict(identity=(0,0,0),acquisition_identity='contact-a',measured_ns=20,available_ns=20,
        velocity_body_m_s=(0.,0.,0.),rotation_contact_anchor_from_body=tuple(map(tuple,np.eye(3))))
    if fault=='nan': args['velocity_body_m_s']=(np.nan,0,0)
    elif fault=='reflection': args['rotation_contact_anchor_from_body']=tuple(map(tuple,-np.eye(3)))
    elif fault=='backward': args['available_ns']=19
    else: args['calibration_id']='vendor-counts-unqualified'
    with pytest.raises(SensorContractError): ContactMotionSample(**args)
