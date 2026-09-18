"""Post-hoc fixed-frame correspondence diagnosis; no pose or command is admitted."""
from copy import deepcopy
import json
import time
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.joint_rgbd_rigid_pose_development import fit, inliers, RIGID_RULES
from lewm.rgbd_correspondence_motion_development import RULES
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check

INPUT = BASE/'go2_independent_floor_transport_mazes_v1_attempt_001'
OUTPUT = BASE/'go2_independent_maze01_correspondence_diagnosis_v1_attempt_001'
CASE = 'full_jepa_novel_maze_01'
FRAME = 214
REFERENCES = tuple(range(206, 214))
TERMINAL_SHA = '97d2075639548de17e73f8857703975b57bebadef6497ea8e3fbe2f59496f1dc'
LAUNCH_SHA = '3053ca602d8e45700550188a3da12e69c3b83314af5a74f32bc616c5425b91c9'
SOURCE = 'scripts/diagnose_go2_independent_maze01_correspondences_v1.py'


def consensus_support(a, b, ua, ub):
    """Exact joint proposal/pruning arithmetic, stopping before later pose gates.

No gyro value is fabricated: joint fitting does not use gyro until the later
consistency gate. This diagnostic never reaches or substitutes that admission.
"""
    if len(a) < RULES['minimum_matches']:
        return dict(lifted_matches=len(a), failure='insufficient rigid-pose matches')
    rng = np.random.default_rng(np.random.SeedSequence([RIGID_RULES['seed'], FRAME]))
    subsets = [np.arange(len(a))]+[rng.choice(len(a), 3, replace=False) for _ in range(RIGID_RULES['proposals'])]
    best = rank = None; valid = 0
    for indices in subsets:
        try: R, t, _ = fit(a[indices], b[indices], gyro_rotation=None)
        except SensorContractError: continue
        valid += 1; mask, residual = inliers(a, b, ua, ub, R, t)
        candidate = (int(mask.sum()), -float(np.sum(residual[mask]**2)))
        if best is None or candidate > rank: best = mask; rank = candidate
    if best is None:
        return dict(lifted_matches=len(a), valid_proposals=valid, failure='no conditioned rigid-pose proposal')
    mask = best.copy(); counts = [int(mask.sum())]
    while int(mask.sum()) >= RULES['minimum_matches']:
        R, t, _ = fit(a[mask], b[mask], gyro_rotation=None)
        good, _ = inliers(a, b, ua, ub, R, t); use = mask & good
        if np.array_equal(use, mask):
            return dict(lifted_matches=len(a), valid_proposals=valid, pruning_counts=counts,
                failure=None, later_fraction_grid_displacement_and_gyro_gates_evaluated=False)
        mask = use; counts.append(int(mask.sum()))
    return dict(lifted_matches=len(a), valid_proposals=valid, pruning_counts=counts,
        failure='insufficient rigid consensus after pruning')


def main():
    if not __debug__: raise ValueError('audit assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive diagnosis output required')
    cv2.setNumThreads(1)
    terminal_name = CASE+'_worker_terminal.json'
    verify_artifacts(INPUT, {terminal_name:TERMINAL_SHA, 'launch.json':LAUNCH_SHA})
    terminal = read_json(INPUT, terminal_name)
    if (terminal['status'] != 'INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
            or terminal['case'] != CASE or terminal['layout_index'] != 1):
        raise ValueError('completed exact maze1 worker required')
    bindings = dict(terminal['artifact_sha256']) | {terminal_name:TERMINAL_SHA, 'launch.json':LAUNCH_SHA}
    verify_artifacts(INPUT, bindings)
    audit = read_json(INPUT, CASE+'_audit.json')
    if not all(audit[k] is True for k in ('raw_sensor_reconstruction_pass', 'raw_model_command_replay_pass',
            'raw_command_audit_pass', 'model_state_unchanged', 'strict_physical_visibility_pass')):
        raise ValueError('exact completed raw audit and visibility evidence required')
    sources = discover_sources((SOURCE,), read_json(INPUT, 'launch.json')['source_sha256'])
    source_check(sources); resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+64*1024**2:
        raise ValueError('bounded CPU diagnosis resource admission failed')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_artifact_sha256=bindings,
        hardware=resources, current_frame=FRAME, reference_frames=list(REFERENCES),
        opencv_threads=1, native_execution=False, checkpoint_loaded=False, posthoc_diagnosis=True))
    started = time.perf_counter()
    try:
        decision = next(r['decision'] for r in read_rows(INPUT/CASE) if r['tick'] == FRAME)
        raw = decision['original_visual_evidence']
        assert decision['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and raw['status'] == 'VISUAL_TERMINAL_FAILURE'
        expected = {'primary':raw['camera_selection']['primary_reference_selection'],
                    'auxiliary':raw['reference_selection']}
        for selection in expected.values():
            assert sorted(x['reference_frame'] for x in selection['attempts']) == list(REFERENCES)
        reader = IntentReturnRGBDReplay(INPUT/CASE)
        acquisitions = read_json(INPUT/CASE, 'auxiliary_camera_audit.json')
        features = {}
        for frame in (*REFERENCES, FRAME):
            policy, depth, _, now = reader.packet(frame)
            image, auxiliary = packet(INPUT/CASE, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
            features[frame] = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'], depth),
                auxiliary=CornerSupportFeatureFrame(image['rgb'], auxiliary))
        rows = []
        for camera in ('primary', 'auxiliary'):
            for frame in REFERENCES:
                values = matched_points(features[frame][camera], features[FRAME][camera])
                support = consensus_support(*values)
                expected_failure = next(r['reason'] for r in expected[camera]['attempts'] if r['reference_frame'] == frame)
                assert support['failure'] == expected_failure, (camera, frame, support, expected_failure)
                rows.append(dict(camera=camera, reference_frame=frame, current_frame=FRAME, **support))
        source_check(sources); verify_artifacts(INPUT, bindings)
        report = dict(status='INDEPENDENT_MAZE01_CORRESPONDENCE_DIAGNOSIS_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'), rows=rows,
            feature_witnesses={str(f):{c:x.witness() for c,x in views.items()} for f,views in features.items()},
            original_terminal_failure=deepcopy(raw['terminal_failure']), matching_rules=deepcopy(RULES),
            rigid_rules=deepcopy(RIGID_RULES), all_sixteen_recorded_reference_rejections_reproduced=True,
            source_and_input_bindings_verified_before_after=True, wall_s=time.perf_counter()-started,
            native_execution=False, model_training=False, thresholds_changed=False,
            pose_admitted=False, command_selected=False, original_outcome_unchanged=True,
            later_pose_gates_evaluated=False, navigation_qualified=False, goal_achieved=False)
        write_json(OUTPUT/'result.json', report)
        print(report['status'], digest(OUTPUT/'result.json'), json.dumps(rows), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_CORRESPONDENCE_DIAGNOSIS_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
