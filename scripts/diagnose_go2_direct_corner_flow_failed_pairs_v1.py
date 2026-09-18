"""Fixed development-pair association probe with unchanged rigid geometry gates."""
import time
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.direct_corner_flow_association_development import tracked_points, FLOW_RULES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.joint_rgbd_rigid_pose_development import register, angle, RIGID_RULES
from lewm.auxiliary_reference_pose_adapter_development import gyro_in_reference, pose_in_body
from lewm.physical_execution_development import rotation_xyzw
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check

INPUT=BASE/'go2_independent_floor_transport_mazes_v1_attempt_001'
OUTPUT=BASE/'go2_direct_corner_flow_failed_pairs_v1_attempt_001'
SOURCE='scripts/diagnose_go2_direct_corner_flow_failed_pairs_v1.py'
CASES=(('full_jepa_novel_maze_01',214,'97d2075639548de17e73f8857703975b57bebadef6497ea8e3fbe2f59496f1dc'),
       ('full_jepa_novel_maze_03',264,'ca6553e0441dfffddc91430444c04591d4c0c506c81a28377e4b9fa7f0005311'))
LAUNCH_SHA='3053ca602d8e45700550188a3da12e69c3b83314af5a74f32bc616c5425b91c9'


def fitted(values, gyro, camera, frame):
    try:
        Q,t,mask,receipt=register(*values,gyro_rotation=gyro if camera=='primary' else gyro_in_reference(gyro),mode='joint',frame=frame)
    except SensorContractError as error:
        return dict(lifted_matches=len(values[0]),rigid_registration_pass=False,failure=str(error))
    if camera=='auxiliary': Q,t=pose_in_body(Q,t)
    return dict(lifted_matches=len(values[0]),rigid_registration_pass=True,failure=None,registration=receipt,
        rotation_reference_body_from_current_body=Q.tolist(),translation_reference_body_m=t.tolist(),
        body_increment_envelope_pass=bool(np.linalg.norm(t)<=RIGID_RULES['maximum_increment_translation_m']
            and angle(Q)<=RIGID_RULES['maximum_increment_rotation_rad']))


def main():
    if not __debug__: raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fixed-pair probe required')
    cv2.setNumThreads(1); bindings={'launch.json':LAUNCH_SHA}
    verify_artifacts(INPUT,bindings)
    for case,frame,sha in CASES:
        name=case+'_worker_terminal.json'; verify_artifacts(INPUT,{name:sha})
        terminal=read_json(INPUT,name)
        if terminal['status']!='INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED' or terminal['case']!=case:
            raise ValueError('completed original case required')
        for k,v in terminal['artifact_sha256'].items():
            if k in bindings and bindings[k]!=v: raise ValueError('conflicting case artifact identities')
            bindings[k]=v
        bindings[name]=sha
    verify_artifacts(INPUT,bindings)
    sources=discover_sources((SOURCE,'lewm/tests/test_direct_corner_flow_association_development.py'),
        read_json(INPUT,'launch.json')['source_sha256'])
    source_check(sources); resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+64*1024**2:
        raise ValueError('bounded fixed-pair probe resource admission failed')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_artifact_sha256=bindings,hardware=resources,
        cases=[dict(case=c,reference_frame=f-1,current_frame=f) for c,f,_ in CASES],
        association_rules=FLOW_RULES,opencv_threads=1,native_execution=False,checkpoint_loaded=False,
        posthoc_diagnosis=True,native_pose_used_only_for_posthoc_error=True))
    started=time.perf_counter()
    try:
        rows=[]
        for case,frame,_ in CASES:
            directory=INPUT/case; reader=IntentReturnRGBDReplay(directory)
            acquisition=read_json(directory,'auxiliary_camera_audit.json'); features={}; orientation=FastRelativeOrientation()
            for i in (frame-1,frame):
                p,d,f,now=reader.packet(i)
                attitude=orientation.begin(p,f,now_ns=now) if i==frame-1 else orientation.step(p,f,now_ns=now)
                image,aux=packet(directory,i,p,public_acquisition(acquisition[i]),now_ns=now)
                features[i]=dict(primary=CornerSupportFeatureFrame(p['image']['rgb'],d),auxiliary=CornerSupportFeatureFrame(image['rgb'],aux))
            G=np.asarray(attitude['rotation_initial_body_from_current_body'])
            decision=next(r['decision'] for r in read_rows(directory) if r['tick']==frame)
            raw=decision['original_visual_evidence']
            assert decision['terminal']=='SENSOR_OR_MODEL_FAILURE' and raw['status']=='VISUAL_TERMINAL_FAILURE'
            for camera in ('primary','auxiliary'):
                reference=features[frame-1][camera]; current=features[frame][camera]
                original=fitted(matched_points(reference,current),G,camera,frame)
                selection=raw['camera_selection']['primary_reference_selection'] if camera=='primary' else raw['reference_selection']
                expected=next(r['reason'] for r in selection['attempts'] if r['reference_frame']==frame-1)
                assert not original['rigid_registration_pass'] and original['failure']==expected
                values,association=tracked_points(reference,current)
                alternative=fitted(values,G,camera,frame)
                # Native state is opened only after public-input association and
                # geometric fitting; it is never supplied to either algorithm.
                with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
                    pose=archive['base_pose_world'][[749+50*(frame-1),749+50*frame]]
                R0=rotation_xyzw(pose[0,3:]); truth_R=R0.T@rotation_xyzw(pose[1,3:]); truth_t=R0.T@(pose[1,:3]-pose[0,:3])
                error=None
                if alternative['rigid_registration_pass']:
                    error=dict(translation_error_m=float(np.linalg.norm(np.asarray(alternative['translation_reference_body_m'])-truth_t)),
                        rotation_error_rad=angle(truth_R.T@np.asarray(alternative['rotation_reference_body_from_current_body'])),
                        evaluator_only=True)
                rows.append(dict(case=case,reference_frame=frame-1,current_frame=frame,camera=camera,
                    original=original,alternative=alternative,association=association,native_error=error,
                    temporal_continuity_evaluated=False,pose_admitted=False,command_selected=False))
        source_check(sources); verify_artifacts(INPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='DIRECT_CORNER_FLOW_FAILED_PAIRS_V1_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'),rows=rows,wall_s=time.perf_counter()-started,
            source_and_input_bindings_verified_before_after=True,original_four_rejections_reproduced=True,
            rigid_geometry_thresholds_changed=False,association_rule_changed=True,
            native_execution=False,checkpoint_loaded=False,pose_admitted=False,controller_integrated=False,
            original_outcomes_unchanged=True,navigation_qualified=False,goal_achieved=False))
        print('DIRECT_CORNER_FLOW_FAILED_PAIRS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_DIRECT_CORNER_FLOW_FAILED_PAIRS_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__': main()
