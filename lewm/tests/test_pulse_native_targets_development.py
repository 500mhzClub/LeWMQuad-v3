"""Target-only motion/contact semantics, censoring and SE(3) frame invariance."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch
from lewm.pulse_native_targets_development import PulseNativeTargets
from lewm.pulse_timed_observation_pairing_development import pulse_window
from lewm.tests.test_pulse_timed_observation_pairing_development import fixture


def data():
    frames,tape,args=fixture();window=pulse_window(frames,tape,**args)
    ns=np.arange(1,3251)*2_000_000
    pose=np.tile([0.,0.,.3,0.,0.,0.,1.],(len(ns),1));pose[:,0]=ns/1e9*.2
    command=np.zeros((len(ns),3));command[(ns>args['departure_ns'])&(ns<=args['departure_ns']+200_000_000),0]=.2
    raw=dict(timestamp_s=ns/1e9,base_pose_world=pose,requested_command=command,physics_contact=np.zeros(len(ns),bool))
    return raw,window


def test_actual_partial_time_and_not_command_kinematics():
    raw,w=data();out=PulseNativeTargets(raw).labels(w)
    assert out['motion_valid'].tolist()==[True]*5+[False]*3
    assert out['motion'][4,0].item()==pytest.approx(.44)
    assert out['contact'][out['contact_valid']].count_nonzero()==0
    # The fixture actually travels .44m despite requesting only a .2s pulse.
    assert out['motion'][4,0].item()!=pytest.approx(.04)


def test_missing_rgb_does_not_destroy_observed_native_outcome():
    raw,w=data();w['targets'][4].update(observation_available=False,future_valid=False,future_observation_index=None)
    assert PulseNativeTargets(raw).labels(w)['motion_valid'][4]


@pytest.mark.parametrize('at',[2_300_000_000,2_302_000_000])
def test_contact_at_or_between_targets_is_absorbing_after_physical_stop(at):
    raw,w=data();ns=np.rint(raw['timestamp_s']*1e9).astype(np.int64);raw['physics_contact'][ns==at]=True
    raw={k:v[ns<=at] for k,v in raw.items()}
    for t in w['targets']:
        if t['target_ns'] is not None and t['target_ns']>at:t['command_prefix_executed']=False
    out=PulseNativeTargets(raw).labels(w)
    assert out['contact_valid'].tolist()==[True]*5+[False]*3
    assert out['motion_valid'][0].item()==(at>2_300_000_000)
    assert out['contact'][4]==1 and not out['motion_valid'][4]


def test_noncontact_stop_is_unknown_not_safe():
    raw,w=data();raw={k:v[:1151] for k,v in raw.items()}
    out=PulseNativeTargets(raw).labels(w)
    assert out['motion_valid'].tolist()==[True]+[False]*7
    assert out['contact_valid'].tolist()==[True]+[False]*7
    assert torch.isnan(out['contact'][1:]).all()


def test_contact_after_diverged_command_is_not_attributed_to_original_plan():
    raw,w=data();raw['requested_command'][1000]=[0.,0.,.45];raw['physics_contact'][1100]=True
    out=PulseNativeTargets(raw).labels(w)
    assert not out['contact_valid'].any() and out['accounting']['first_matched_contact_ns'] is None


def test_incomplete_tape_cannot_be_overridden_by_native_availability():
    raw,w=data();w['targets'][4]['command_prefix_executed']=False
    out=PulseNativeTargets(raw).labels(w)
    assert not out['motion_valid'][4] and not out['contact_valid'][4]


def test_contact_before_departure_is_not_new_safe_training_window():
    raw,w=data();raw['physics_contact'][100]=True;out=PulseNativeTargets(raw).labels(w)
    assert not out['contact_valid'].any() and not out['motion_valid'].any()


def test_full_rigid_coordinate_change_leaves_current_body_targets_identical():
    raw,w=data();n=len(raw['timestamp_s']);yaw=np.linspace(0,.7,n)
    raw['base_pose_world'][:,3:]=Rotation.from_euler('xyz',np.c_[yaw*.2,yaw*.3,yaw]).as_quat()
    original=PulseNativeTargets(raw).labels(w);changed=deepcopy(raw)
    global_R=Rotation.from_euler('xyz',[.4,-.3,1.2]);p=changed['base_pose_world']
    p[:,:3]=global_R.apply(p[:,:3])+[4.,-2.,1.]
    p[:,3:]=(global_R*Rotation.from_quat(p[:,3:])).as_quat()
    result=PulseNativeTargets(changed).labels(w)
    torch.testing.assert_close(result['motion'][:5],original['motion'][:5],rtol=0,atol=1e-6)


@pytest.mark.parametrize('mode',['clock','quaternion','contact_type','nan_command','target_time'])
def test_malformed_native_or_target_contract_rejected(mode):
    raw,w=data()
    if mode=='clock':raw['timestamp_s'][100]+=.001
    elif mode=='quaternion':raw['base_pose_world'][100,3:]=0
    elif mode=='contact_type':raw['physics_contact']=raw['physics_contact'].astype(float)
    elif mode=='nan_command':raw['requested_command'][100,0]=np.nan
    else:w['targets'][4]['offset_ns']+=100_000_000
    with pytest.raises(ValueError):PulseNativeTargets(raw).labels(w)
