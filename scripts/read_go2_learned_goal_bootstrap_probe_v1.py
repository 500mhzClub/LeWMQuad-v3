"""Receipt-bound terminal summary and unchanged-consensus failure decomposition."""
from collections import Counter
import json
import cv2
import numpy as np

from lewm.rigid_consensus_diagnostic_development import diagnose
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.keyframe_rgbd_pose_development import FeatureFrame, matched_points
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.run_go2_learned_goal_bootstrap_probe_v1 import OUTPUT as INPUT, TRIALS
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify

OUTPUT = BASE/'go2_learned_goal_bootstrap_readout_v1_attempt_001'
PROTOCOL = 'docs/go2_learned_goal_bootstrap_readout_v1_2026-09-08.md'
IDS = dict(launch_json='962f9d461648d5aaa353c2a7a20e0a90245724cbf66cac223b60a89466422254',
    result_json='71601c660272b579db520ba631af661d76f869267cf1c4a26fb60b17cdfbb7f1')


def timing(values):
    a = np.asarray(values, float)
    return dict(count=len(a), median_ms=float(np.median(a)), maximum_ms=float(a.max()),
        over_100ms=int((a > 100.).sum())) if len(a) else None


def failure_pairs(trial, rows):
    failed = next((r for r in rows if r['decision']['terminal'] == 'SENSOR_OR_MODEL_FAILURE'), None)
    if failed is None:
        return None
    tick = failed['tick']
    evidence = failed['decision']['evidence']
    reader = IntentReturnRGBDReplay(INPUT/trial)
    gyro = FastRelativeOrientation()
    rotations, frames = {}, {}
    refs = [a['reference_frame'] for a in evidence['reference_selection']['attempts']]
    for i in range(tick+1):
        p, d, f, now = reader.packet(i)
        attitude = gyro.begin(p, f, now_ns=now) if i == 0 else gyro.step(p, f, now_ns=now)
        if i in refs or i == tick:
            rotations[i] = np.asarray(attitude['rotation_initial_body_from_current_body'])
            frames[i] = FeatureFrame(p['image']['rgb'], d)
    pairs = []
    for ref in refs:
        a, b, ua, ub = matched_points(frames[ref], frames[tick])
        row = diagnose(a, b, ua, ub, gyro_rotation=rotations[ref].T@rotations[tick], frame=tick)
        assert not row['original_accepted']
        expected = next(r['reason'] for r in evidence['reference_selection']['attempts'] if r['reference_frame'] == ref)
        assert row['original_rejection'] == expected
        pairs.append(dict(reference_frame=ref, current_frame=tick, **row))
    return dict(first_failure_tick=tick, failure=failed['decision']['failure'],
        observer_failure=evidence['terminal_failure'], pairs=pairs)


def main():
    if not __debug__:
        raise ValueError('audit assertions required')
    cv2.setNumThreads(1)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive diagnostic readout')
    ids = {k.replace('_json', '.json'): v for k, v in IDS.items()}
    verify_artifacts(INPUT, ids)
    result, original = read_json(INPUT, 'result.json'), read_json(INPUT, 'launch.json')
    assert result['status'] == 'LEARNED_GOAL_BOOTSTRAP_PROBE_COMPLETE' and result['trials'] == list(TRIALS)
    bindings = ids | result['artifact_sha256']
    verify_artifacts(INPUT, bindings)
    sources = discover_sources((PROTOCOL, 'scripts/read_go2_learned_goal_bootstrap_probe_v1.py',
        'lewm/tests/test_rigid_consensus_diagnostic_development.py'), original['source_sha256'])
    launch = original | dict(source_sha256=sources, output_root=str(OUTPUT), protocol=PROTOCOL,
        input_probe_root=str(INPUT), probe_artifact_sha256=bindings, native_execution=False)
    verify(launch)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', launch)
    try:
        reports = []
        for trial in TRIALS:
            audit = read_json(INPUT, trial+'_audit.json')
            rows = read_json(INPUT/trial, 'context_decisions.json')
            collection = read_json(INPUT/trial, 'result.json')
            selected = [r for r in rows if r['decision']['new_selection'] is not None]
            reports.append(dict(trial=trial, collection=collection, goal=audit['goal'],
                online_selection_count=len(selected),
                selected_actions=dict(Counter(r['decision']['selected_action'] for r in selected)),
                selection_sequence=[dict(tick=r['tick'], action=r['decision']['selected_action']) for r in selected],
                failure_diagnosis=failure_pairs(trial, rows),
                raw_model_command_replay_pass=audit['raw_model_command_replay_pass'],
                raw_sensor_reconstruction_pass=audit['raw_sensor_reconstruction_pass'],
                strict_physical_visibility_pass=audit['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=audit['hard_measurement_failed_frames'],
                maximum_observed_pose_xy_error_m=max(audit['observed_pose_xy_errors_m']),
                observation_control_timing=timing(audit['observation_and_control_wall_ms']),
                complete_iteration_timing=timing(audit['iteration_with_command_wall_ms'])))
        verify(launch)
        verify_artifacts(INPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='LEARNED_GOAL_PROBE_DIAGNOSIS_COMPLETE',
            conditions=reports, probe_sha256=ids, source_sha256=sources,
            launch_sha256=digest(OUTPUT/'launch.json'), native_execution=False,
            original_outcomes_changed=False, navigation_qualified=False, goal_achieved=False))
        print(json.dumps(dict(result_sha256=digest(OUTPUT/'result.json'), conditions=reports), indent=2), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_PROBE_DIAGNOSIS_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
