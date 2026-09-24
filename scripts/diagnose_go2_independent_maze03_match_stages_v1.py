"""Fixed failed-frame tracking diagnosis; no new pose, command or threshold."""
import time
import cv2
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.rgbd_correspondence_motion_development import RULES
from lewm.rgbd_match_stage_diagnostic_development import diagnose_matches
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check

INPUT = BASE/'go2_independent_floor_transport_mazes_v1_attempt_001'
OUTPUT = BASE/'go2_independent_maze03_match_stage_diagnosis_v1_attempt_001'
CASE = 'full_jepa_novel_maze_03'
FRAME = 264
REFERENCES = tuple(range(256, 264))
TERMINAL_SHA = 'ca6553e0441dfffddc91430444c04591d4c0c506c81a28377e4b9fa7f0005311'
LAUNCH_SHA = '3053ca602d8e45700550188a3da12e69c3b83314af5a74f32bc616c5425b91c9'
SOURCE = 'scripts/diagnose_go2_independent_maze03_match_stages_v1.py'


def main():
    if not __debug__: raise ValueError('audit assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive maze3 diagnosis required')
    cv2.setNumThreads(1)
    name = CASE+'_worker_terminal.json'
    verify_artifacts(INPUT, {name:TERMINAL_SHA, 'launch.json':LAUNCH_SHA})
    terminal = read_json(INPUT, name)
    if (terminal['status'] != 'INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
            or terminal['case'] != CASE or terminal['layout_index'] != 3):
        raise ValueError('completed exact maze3 worker required')
    bindings = dict(terminal['artifact_sha256']) | {name:TERMINAL_SHA, 'launch.json':LAUNCH_SHA}
    verify_artifacts(INPUT, bindings)
    audit = read_json(INPUT, CASE+'_audit.json')
    if not all(audit[k] is True for k in ('raw_sensor_reconstruction_pass', 'raw_model_command_replay_pass',
            'raw_command_audit_pass', 'model_state_unchanged', 'strict_physical_visibility_pass')):
        raise ValueError('completed raw audit and visibility evidence required')
    sources = discover_sources((SOURCE,), read_json(INPUT, 'launch.json')['source_sha256'])
    source_check(sources); resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+64*1024**2:
        raise ValueError('bounded CPU diagnosis resource admission failed')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_artifact_sha256=bindings,
        hardware=resources, current_frame=FRAME, reference_frames=list(REFERENCES),
        opencv_threads=1, native_execution=False, checkpoint_loaded=False, posthoc_diagnosis=True,
        concurrency_reason='one brief CPU feature diagnostic beside existing replay and reactive audit'))
    started = time.perf_counter()
    try:
        decision = next(r['decision'] for r in read_rows(INPUT/CASE) if r['tick'] == FRAME)
        raw = decision['original_visual_evidence']
        assert decision['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and raw['status'] == 'VISUAL_TERMINAL_FAILURE'
        expected = {'primary':raw['camera_selection']['primary_reference_selection'],
                    'auxiliary':raw['reference_selection']}
        for selection in expected.values():
            assert sorted(x['reference_frame'] for x in selection['attempts']) == list(REFERENCES)
            assert all(x['reason'] == 'insufficient rigid-pose matches' for x in selection['attempts'])
        reader = IntentReturnRGBDReplay(INPUT/CASE)
        acquisitions = read_json(INPUT/CASE, 'auxiliary_camera_audit.json'); features = {}
        for frame in (*REFERENCES, FRAME):
            policy, depth, _, now = reader.packet(frame)
            image, auxiliary = packet(INPUT/CASE, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
            features[frame] = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'], depth),
                auxiliary=CornerSupportFeatureFrame(image['rgb'], auxiliary))
        rows = []
        for camera in ('primary', 'auxiliary'):
            for frame in REFERENCES:
                row = diagnose_matches(features[frame][camera], features[FRAME][camera])
                assert row['counts']['valid_depth_pair'] < RULES['minimum_matches']
                rows.append(dict(camera=camera, reference_frame=frame, current_frame=FRAME, **row))
        source_check(sources); verify_artifacts(INPUT, bindings)
        report = dict(status='INDEPENDENT_MAZE03_MATCH_STAGE_DIAGNOSIS_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'), rows=rows,
            feature_witnesses={str(f):{c:x.witness() for c,x in views.items()} for f,views in features.items()},
            all_sixteen_recorded_reference_rejections_reproduced=True,
            original_correspondence_arrays_verified=64, reference_pairs_verified=16,
            source_and_input_bindings_verified_before_after=True, wall_s=time.perf_counter()-started,
            native_execution=False, model_training=False, thresholds_changed=False,
            pose_admitted=False, command_selected=False, original_outcome_unchanged=True,
            later_pose_gates_evaluated=False, navigation_qualified=False, goal_achieved=False)
        write_json(OUTPUT/'result.json', report)
        print(report['status'], digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MAZE03_MATCH_STAGE_DIAGNOSIS_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
