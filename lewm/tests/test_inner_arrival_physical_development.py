"""Paired intervention boundaries and unchanged independent score."""
import copy
import numpy as np
import pytest
from lewm.inner_goal_room_return_development import InnerGoalRoomReturn
from lewm.inner_goal_pulse_feedback_development import InnerGoalPulseServo
from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion
from lewm.tests.test_coupled_pulse_rollout_development import table
from lewm.tests.test_continuous_pulse_execution_development import visual
from scripts import run_go2_inner_arrival_room_return_v1 as new
from scripts import run_go2_intent_room_return_v1 as old
from scripts import audit_go2_inner_arrival_room_return_v1 as audit
from scripts import audit_go2_intent_room_return_v1 as old_audit
from scripts.navigation_artifact_root_development import BASE,validate_root


def test_same_scene_session_gait_and_original_observer():
    assert new.specification is old.specification
    assert new.IntentRoomReturnSession is old.IntentRoomReturnSession
    assert new.TRIALS==old.TRIALS==('nominal_left','nominal_right','lower_friction_left')
    assert new.load_fixed_table is old.load_fixed_table
    model=InnerGoalRoomReturn(1,table())
    assert type(model.runtime.motion) is MultiReferenceVisualLedMotion


def test_exclusive_output_and_exact_external_baseline():
    assert new.PREVIOUS==old.OUTPUT and new.OUTPUT!=old.OUTPUT
    assert new.OUTPUT.parent==BASE and validate_root(new.OUTPUT,must_exist=False)==new.OUTPUT
    assert new.IDENTITIES['raw_return_audit.json']=='a350c74d4f5851a7486bf01a8276be7f9eb420a83cd8198ad684159b8385923d'


def test_external_score_functions_are_the_unchanged_baseline():
    assert audit.score_hold is old_audit.score_hold
    assert audit.score_winding is old_audit.score_winding
    p=np.zeros((1300,3));p[799:1300,0]=.06155710142307066
    assert not audit.score_hold(p,np.zeros(1300),np.zeros((1300,6)),
        end=1299,target_xy=[0.,0.],target_yaw=0.)['passed']


def test_inner_controller_uses_existing_mission_budgets():
    model=InnerGoalRoomReturn(1,table());e,t=visual(0)
    ex=model.runtime.executor;ex.observe(e,now_ns=t);ex.begin([.4,0],0.,now_ns=t)
    assert type(ex.active) is InnerGoalPulseServo
    assert (ex.maximum_ticks,ex.maximum_pulses,ex.maximum_legs)==(3600,140,36)


def paired_fixture():
    raw={k:np.zeros((751,n)) for k,n in (
        ('base_pose_world',7),('base_twist_world',6),('joint_position',12),
        ('joint_velocity',12),('requested_command',3),('applied_command',3))}
    camera=[{'rgb_sha256':'a'*64}]
    return raw,camera


def test_pairing_checks_setup_not_changed_control_trajectory():
    raw,cam=paired_fixture();baseline=copy.deepcopy(raw)
    for v in raw.values():v[750]=1.
    audit.verify_paired_setup(raw,cam,baseline,cam)


@pytest.mark.parametrize('key',['base_pose_world','base_twist_world','joint_position',
    'joint_velocity','requested_command','applied_command'])
def test_pairing_rejects_any_changed_initial_dynamics_field(key):
    raw,cam=paired_fixture();baseline=copy.deepcopy(raw);raw[key][749,0]=1e-12
    with pytest.raises(AssertionError):audit.verify_paired_setup(raw,cam,baseline,cam)


def test_pairing_rejects_appearance_change():
    raw,cam=paired_fixture()
    with pytest.raises(AssertionError):audit.verify_paired_setup(raw,cam,raw,[{'rgb_sha256':'b'*64}])


def test_pairing_rejects_incomplete_setup():
    raw,cam=paired_fixture();baseline={k:v[:749] for k,v in raw.items()}
    with pytest.raises(AssertionError):audit.verify_paired_setup(raw,cam,baseline,cam)
