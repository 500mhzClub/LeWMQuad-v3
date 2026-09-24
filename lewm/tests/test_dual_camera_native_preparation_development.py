from copy import deepcopy
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from lewm.tests.test_causal_auxiliary_rgb_observation_development import inputs
from lewm.auxiliary_downward45_depth_geometry_development import CALIBRATION_ID
from lewm.dual_camera_native_admission_development import admit_prefix, admit_predecessor
from scripts.dual_camera_native_prefix_comparison_development import prefix_shape
from scripts.dual_camera_novel_maze_session_development import DualCameraNovelMazeSession
from scripts.novel_maze_round_trip_session_development import NovelMazeRoundTripSession


def test_native_packet_extension_reads_captured_rgb_and_preserves_parent_packets(tmp_path,monkeypatch):
    policy,aux,image,now,native,rgb=inputs()
    np.savez(tmp_path/'auxiliary_depth_0000.npz',native_optical_depth_m=native,
        depth_m=aux['depth_m'],valid=aux['valid'],
        diagnostic_segmentation=np.array([{'privileged':True}],dtype=object))
    Image.fromarray(rgb).save(tmp_path/'auxiliary_rgb_0000.png')
    row=dict(frame=0,measured_ns=now,calibration_id=CALIBRATION_ID,rgb_sha256=image['rgb_sha256'],
        native_depth_sha256=hashlib.sha256(native.tobytes()).hexdigest(),world_from_optical='not public')
    session=object.__new__(DualCameraNovelMazeSession)
    session.output=tmp_path;session.model_manifest=[{}];session.auxiliary_audit=[row]
    primary=object();fast=object();calls=[]
    def parent(self):
        calls.append(self);return policy,primary,fast,aux,now
    monkeypatch.setattr(NovelMazeRoundTripSession,'sensor_packets',parent)
    p,d,f,a,new_image,t=session.sensor_packets()
    assert calls==[session] and p is policy and d is primary and f is fast and a is aux and t==now
    np.testing.assert_array_equal(new_image['rgb'],rgb)
    assert 'world_from_optical' not in new_image and not np.shares_memory(new_image['rgb'],rgb)


def prefix():
    return dict(status='DUAL_CAMERA_CONTROLLER_PREFIX_V2_COMPLETE',frames=1873,
        first_auxiliary_intervention_frame=1872,exact_primary_decision_frames=1872,
        final_terminal=None,final_failure=None,final_registered_pose_available=True,
        following_recorded_observations_consumed=False,native_audit_replaced=False,native_execution=False,
        all_preintervention_requested_commands_exact=True,
        complete_preintervention_decisions_exact_outside_added_modality_metadata=True,
        model_state_unchanged=True,failed_json_identity_attempt_preserved=True,controller_implementation_unchanged=True,
        completed_native_audit_and_matching_collection_bindings_required_before_next_native=True)


@pytest.mark.parametrize('field,value',[('status','DUAL_CAMERA_CONTROLLER_PREFIX_COMPLETE'),
    ('final_terminal','SENSOR_OR_MODEL_FAILURE'),('final_registered_pose_available',False),
    ('following_recorded_observations_consumed',True),('native_audit_replaced',True),
    ('model_state_unchanged',False),('all_preintervention_requested_commands_exact',False)])
def test_partial_failed_or_counterfactual_prefix_cannot_admit_native(field,value):
    p=prefix();admit_prefix(p)
    p[field]=value
    with pytest.raises(ValueError):admit_prefix(p)


@pytest.mark.parametrize('field,value',[('frames',1872),('frames',True),('first_auxiliary_intervention_frame',0),
    ('first_auxiliary_intervention_frame',1883),('exact_primary_decision_frames',1871)])
def test_physical_prefix_requires_exact_intervention_boundary(field,value):
    p=prefix();p[field]=value
    with pytest.raises(ValueError):prefix_shape(p)


def predecessor():
    case='full_jepa_novel_maze_00'
    bindings={'launch.json':'a'*64,case+'/result.json':'b'*64,case+'/rgb_0000.png':'c'*64}
    native=dict(status='SETTLED_BOUNDARY_MAZE_PILOT_V2_COMPLETE',conditions=[dict(case=case,
        status='SETTLED_BOUNDARY_MAZE_COLLECTED_AND_RAW_AUDITED',prefix_comparison=dict(physical_and_public_prefix_exact=True))],
        artifact_sha256=dict(bindings))
    audit=dict(raw_sensor_reconstruction_pass=True,raw_model_command_replay_pass=True,
        raw_command_audit_pass=True,model_state_unchanged=True,verified_round_trip=False,strict_physical_visibility_pass=False)
    return native,audit,dict(collected_input_sha256=bindings),case


@pytest.mark.parametrize('fault',['partial','replay','prefix','pixels','missing_binding','relabeled_visibility'])
def test_final_audit_must_validate_the_same_collection_not_just_status(fault):
    native,audit,launch,case=predecessor()
    assert admit_predecessor(native,audit,launch,case=case)['matched_artifact_count']==3
    if fault=='partial':native['conditions'][0]['status']='COLLECTED'
    if fault=='replay':audit['raw_model_command_replay_pass']=False
    if fault=='prefix':native['conditions'][0]['prefix_comparison']['physical_and_public_prefix_exact']=False
    if fault=='pixels':native['artifact_sha256'][case+'/rgb_0000.png']='d'*64
    if fault=='missing_binding':native['artifact_sha256'].pop(case+'/rgb_0000.png')
    if fault=='relabeled_visibility':audit['strict_physical_visibility_pass']=True
    with pytest.raises(ValueError):admit_predecessor(native,audit,launch,case=case)
