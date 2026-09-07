"""Prospective scene, reader, storage and unchanged native acceptance tests."""
import copy
import math
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.intent_room_return_scene_development import specification,pack,TRIALS
from lewm.coupled_room_return_scene_development import specification as predecessor
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay,validate_frame_population,MAX_FRAMES
from lewm.cached_rgbd_replay_development import CachedRGBDReplay
from scripts.audit_go2_intent_room_return_v1 import score_hold,score_winding
from scripts.audit_go2_coupled_room_return_v1 import score_hold as old_hold,score_winding as old_winding
from scripts.run_go2_intent_room_return_v1 import OUTPUT,ROOT,artifacts
from scripts.navigation_artifact_root_development import BASE,validate_root


@pytest.mark.parametrize('trial',TRIALS)
def test_actual_new_scene_identity_and_unchanged_geometry(trial):
    s=specification(trial);old=predecessor(trial);p=pack(s)
    assert s['geometry']['spawn_se2_world']!=old['geometry']['spawn_se2_world']
    assert s['procedural_seed']!=old['procedural_seed'] and s['appearance_seed']!=old['appearance_seed']
    assert s['geometry']['wall_boxes']==old['geometry']['wall_boxes']
    assert s['friction_mu']==old['friction_mu'] and s['turn_sign']==old['turn_sign']
    x,y,yaw=s['geometry']['spawn_se2_world']
    assert p.robot.spawn_xyz_m==(x,y,.375)
    assert p.robot.spawn_quat_wxyz==(math.cos(yaw/2),0.,0.,math.sin(yaw/2))
    altered=copy.deepcopy(s);altered['geometry']['spawn_se2_world'][0]+=.01
    with pytest.raises(ValueError):pack(altered)


def test_low_friction_pair_retained():
    a,b=specification('nominal_left'),specification('lower_friction_left')
    assert a['geometry']==b['geometry'] and a['procedural_seed']==b['procedural_seed']
    assert a['appearance_seed']==b['appearance_seed']
    assert (a['friction_mu'],b['friction_mu'])==(1.,.15)


def test_complete_terminal_drain_population_not_silently_truncated():
    assert MAX_FRAMES==3611
    validate_frame_population([None]*3611)
    for x in ([],[None]*3612,None,()):
        with pytest.raises(SensorContractError):validate_frame_population(x)
    names=artifacts({'rgbd_frames':3611})
    assert all(n in names for n in ('rgb_3610.png','depth_3610.npz','native_depth_3610.npz'))
    assert len(names)==len(set(names))


@pytest.mark.parametrize('frame',[0,366,376])
def test_successor_reader_preserves_existing_packet_bytes(frame):
    directory=ROOT/'.generated/go2_coupled_room_return_v1_attempt_001/lower_friction_left'
    old=CachedRGBDReplay(directory).packet(frame)
    new=IntentReturnRGBDReplay(directory).packet(frame)
    def same(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
        else:assert a==b
    for a,b in zip(old,new,strict=True):same(a,b)


def test_external_output_is_explicit_not_source_relocation():
    assert OUTPUT.parent==BASE and not OUTPUT.is_relative_to(ROOT)
    assert validate_root(OUTPUT,must_exist=False)==OUTPUT


@pytest.mark.parametrize('error,passed',[(.05999,True),(.06005187983247706,False)])
def test_native_hold_limit_unchanged_including_previous_failure(error,passed):
    p=np.zeros((1300,3));p[799:1300,0]=error;y=np.zeros(1300);v=np.zeros((1300,6))
    args=dict(end=1299,target_xy=[0.,0.],target_yaw=0.)
    result=score_hold(p,y,v,**args)
    assert result==old_hold(p,y,v,**args) and result['passed']==passed


def test_signed_winding_is_not_wrapped_success():
    goal=dict(anchor_ns=1_500_000_000,anchor_rotation=np.eye(3).tolist(),
              target_yaw_rad=math.pi,requested_yaw_delta_rad=-math.pi)
    y=np.zeros(2000);y[750:1499]=np.linspace(0,math.pi,749);y[1499:]=math.pi
    assert score_winding(y,goal,end=1999)==old_winding(y,goal,end=1999)
    assert not score_winding(y,goal,end=1999)['passed']
