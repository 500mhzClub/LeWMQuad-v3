import copy

import numpy as np
import pytest

from lewm.tests.test_successive_choice_metrics_development import fixture,reduce
from scripts.audit_go2_successive_choice_maze_development_v1 import body_delta,scalar_metrics,audit_commands,camera_reference


def example():
    raw,tape,choices=fixture()
    row={'prefix_terminal_sample_index':749,'intent_xy_body_start_m':[.8,0],'branchable':True,
        'sensor_fault':None,'stop_reason':None,'metrics':reduce(raw,tape,choices)}
    return raw,tape,choices,row


def test_independent_vector_reference_and_scalar_metrics():
    raw,tape,choices,row=example(); scalar_metrics(raw,row,tape,choices)
    assert body_delta(raw['base_pose_world'][749],raw['base_pose_world'][2749])==pytest.approx([.4,0,0],abs=1e-12)


@pytest.mark.parametrize('fault',['progress','brier','mask','motion'])
def test_independent_checks_reject_corrupt_metrics(fault):
    raw,tape,choices,row=example()
    if fault=='progress': row['metrics']['observed_signed_control_displacement_m']+=.1
    if fault=='brier': row['metrics']['executed_decision_errors'][0]['contact_brier']=.9
    if fault=='mask': row['metrics']['executed_decision_errors'][0]['label']['motion_valid']=False
    if fault=='motion': row['metrics']['executed_decision_errors'][0]['label']['delta_xy_yaw_start_body'][0]+=.1
    with pytest.raises(ValueError): scalar_metrics(raw,row,tape,choices)


def command_example():
    raw,tape,choices,row=example(); events=[]
    raw['edge_index']=np.zeros(3000,dtype=int); raw['edge_index'][750:]=1
    raw['phase']=np.zeros(3000,dtype=int); raw['phase'][750:2750]=1; raw['phase'][2750:]=2
    raw['requested_command']=np.zeros((3000,3)); raw['requested_command'][750:2750]=[.2,0,0]
    raw['applied_command']=raw['requested_command'].astype(np.float32)
    for index,choice in enumerate(choices):
        choice['requested_command_tape']=[[.2,0,0]]*5
        choice['expected_applied_command_tape']=[[.2,0,0]]*5
        events.append({'pre_sample_index':749+index*250,'selection':choice})
    for tick,entry in enumerate(tape):
        entry.update(tick=tick,timestamp_s=float(raw['timestamp_s'][entry['pre_sample_index']]),
            requested_command=[.2,0,0] if tick<40 else [0,0,0])
    return raw,row,tape,events


def test_all_commands_and_slew_independently_match():
    audit_commands(*command_example())


def test_camera_mount_rejects_proper_but_wrong_frame():
    pose=np.array([1.,2.,.3,0,0,0,1]); optical=np.eye(4)
    optical[:3,:3]=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    optical[:3,3]=[1.326,2.,.343]; camera_reference(pose,optical)
    optical[:3,:3]=np.eye(3)
    with pytest.raises(ValueError,match='camera rigid'): camera_reference(pose,optical)


@pytest.mark.parametrize('fault',['unexplained','changed_action','slew','phase','after_release'])
def test_command_audit_rejects_execution_corruption(fault):
    raw,row,tape,events=command_example()
    if fault=='unexplained': tape=tape[:-1]
    if fault=='changed_action': tape[1]['requested_command']=[-.2,0,0]
    if fault=='slew': raw['applied_command'][850,0]=-.2
    if fault=='phase': raw['phase'][850]=2
    if fault=='after_release': tape[41]['stage']='control'
    with pytest.raises(ValueError): audit_commands(raw,row,tape,events)
