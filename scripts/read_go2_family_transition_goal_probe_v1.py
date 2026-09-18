"""Receipt-bound failure diagnosis; no controller, model or renderer changes."""
from collections import Counter
import json
import cv2
import numpy as np

from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.rigid_consensus_diagnostic_development import diagnose
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing
from scripts.run_go2_family_transition_goal_probe_v1 import OUTPUT as INPUT, TRIALS
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify

OUTPUT = BASE/'go2_family_transition_goal_readout_v1_attempt_001'
PROTOCOL = 'docs/go2_family_transition_goal_readout_v1_2026-09-08.md'
IDS = {'launch.json': '5d708568e8e1fb8d518d741d471172d0f718d7d158d130e98fe1fd6af0a55608',
    'result.json': 'eb70bb999f78f063031575b2773e2871651f095cb583e0bd6fb4b94ec84bfa4e'}


def failure_pairs(trial, rows):
    failed = next((r for r in rows if r['decision']['terminal'] == 'SENSOR_OR_MODEL_FAILURE'), None)
    if failed is None:
        return None
    tick = failed['tick']
    evidence = failed['decision']['evidence']
    attempts = evidence['reference_selection']['attempts']
    refs = [r['reference_frame'] for r in attempts]
    reader = IntentReturnRGBDReplay(INPUT/trial)
    gyro = FastRelativeOrientation()
    rotations, frames = {}, {}
    for i in range(tick+1):
        p, d, f, now = reader.packet(i)
        attitude = gyro.begin(p, f, now_ns=now) if i == 0 else gyro.step(p, f, now_ns=now)
        if i in refs or i == tick:
            rotations[i] = np.asarray(attitude['rotation_initial_body_from_current_body'])
            frames[i] = CornerSupportFeatureFrame(p['image']['rgb'], d)
    pairs = []
    for attempt in attempts:
        ref = attempt['reference_frame']
        a, b, ua, ub = matched_points(frames[ref], frames[tick])
        row = diagnose(a, b, ua, ub, gyro_rotation=rotations[ref].T@rotations[tick], frame=tick)
        assert not row['original_accepted'] and row['original_rejection'] == attempt['reason']
        pairs.append(dict(reference_frame=ref, current_frame=tick,
            reference_features=frames[ref].witness(), current_features=frames[tick].witness(), **row))
    return dict(first_failure_tick=tick, observer_failure=evidence['terminal_failure'], pairs=pairs)


def contact_witness(trial, rows):
    events = [e for e in read_json(INPUT/trial, 'contact_events.json') if e['disallowed_contacts']]
    if not events:
        return None
    first = events[0]
    selected = [r for r in rows if r['decision']['new_selection'] is not None
        and r['pre_sample_index'] < first['sample_index']]
    row = selected[-1]
    choice = row['decision']['new_selection']
    score = next(c for c in choice['candidates'] if c['action'] == choice['action'])
    prediction = np.asarray(choice['prediction'])[choice['action_index']]
    return dict(first_contact=first, last_selection_tick=row['tick'], selected_action=choice['action'],
        physics_steps_after_selection=first['sample_index']-row['pre_sample_index'],
        selected_candidate=score, predicted_first_half_second_contact_score=float(
            1./(1.+np.exp(-float(prediction[0, 4])))), calibrated_probability=False,
        counterfactual_action_outcomes_observed=False)


def main():
    if not __debug__:
        raise ValueError('audit assertions required')
    cv2.setNumThreads(1)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive diagnosis attempt')
    verify_artifacts(INPUT, IDS)
    result, original = read_json(INPUT, 'result.json'), read_json(INPUT, 'launch.json')
    assert result['status'] == 'FAMILY_TRANSITION_GOAL_PROBE_COMPLETE' and result['trials'] == list(TRIALS)
    bindings = IDS | result['artifact_sha256']
    verify_artifacts(INPUT, bindings)
    sources = discover_sources((PROTOCOL, 'scripts/read_go2_family_transition_goal_probe_v1.py'),
        original['source_sha256'])
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
            selected = [r for r in rows if r['decision']['new_selection'] is not None]
            reports.append(dict(trial=trial, collection=read_json(INPUT/trial, 'result.json'),
                goal=audit['goal'], online_selection_count=len(selected),
                selected_actions=dict(Counter(r['decision']['selected_action'] for r in selected)),
                failure_diagnosis=failure_pairs(trial, rows), contact_witness=contact_witness(trial, rows),
                raw_model_command_replay_pass=audit['raw_model_command_replay_pass'],
                raw_sensor_reconstruction_pass=audit['raw_sensor_reconstruction_pass'],
                strict_physical_visibility_pass=audit['strict_physical_visibility_pass'],
                hard_measurement_failures=[dict(frame=i, footprint=audit['footprint_checks'][i])
                    for i in audit['hard_measurement_failed_frames']],
                maximum_observed_pose_xy_error_m=max(audit['observed_pose_xy_errors_m']),
                timing={key: timing([r[key] for r in rows if key in r]) for key in
                    ('acquisition_wall_ms', 'controller_wall_ms', 'observation_and_control_wall_ms',
                     'iteration_with_command_wall_ms')}))
        verify(launch)
        verify_artifacts(INPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='FAMILY_TRANSITION_GOAL_DIAGNOSIS_COMPLETE',
            conditions=reports, probe_sha256=IDS, source_sha256=sources,
            launch_sha256=digest(OUTPUT/'launch.json'), native_execution=False,
            original_outcomes_changed=False, navigation_qualified=False, goal_achieved=False))
        print('FAMILY_TRANSITION_GOAL_DIAGNOSIS_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_DIAGNOSIS_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
