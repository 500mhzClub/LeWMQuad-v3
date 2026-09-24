"""Replay one fresh joint-floor candidate against a fully audited predecessor, stopping at intervention."""
import argparse
import json
import time
import cv2
import torch
from lewm.joint_floor_registered_controller_development import JointFloorRegisteredRoundTripController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows, writer, NAME as DECISIONS
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_confirmed_floor_maze_pilot_v1 import OUTPUT as INPUT, CASE, CORRECTION, FITS
from scripts.read_go2_confirmed_floor_maze_pilot_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE/'go2_joint_floor_registered_maze_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_joint_floor_registered_maze_prefix_v1_2026-09-08.md'
FAILED = BASE/'go2_floor_registered_maze_prefix_v1_attempt_001'
FAILED_BINDINGS = {
    'launch.json': 'bfcf3260e39a15ec7d94858b492099e85fde2349fbb11eeddc2aafd348a5b92e',
    'failure.json': '562acbc1463ef16864be4e6cf2128e47289d2441258585b4e521799b6fe53c81',
    DECISIONS: '542e85af02532711c6777ffb8ef7fd0f20360274459de35d793cc07b6f006889',
}


def compare_current(old, candidate, *, tick):
    """No equality claim for intentionally corrected maps, scores or mission distances."""
    if candidate['original_visual_evidence'] != old['evidence']:
        raise ValueError(f'original visual fit changed at {tick}')
    if candidate['failure'] is not None: raise ValueError(f'candidate admission failure at {tick}: {candidate["failure"]}')
    e = candidate['evidence']
    if e is not None and e['original_visual_evidence'] != old['evidence']:
        raise ValueError('registration must retain the complete original visual witness')
    a, b = old['new_selection'], candidate['new_selection']
    if a and b and 'prediction' in a and 'prediction' in b:
        if a['prediction'] != b['prediction']:
            raise ValueError('same model and public history must yield identical raw forecasts')
    return dict(command_changed=old['requested_command'] != candidate['requested_command'],
        terminal_changed=old['terminal'] != candidate['terminal'])


def replay(admission):
    name, index, variant, condition, model_name = CASE; directory = INPUT/name
    model, c, v = load_assigned(admission, model_name)
    if (c, v) != (condition, variant): raise ValueError('same assigned learned model required')
    before = state_digest(model.state_dict()); geometry = ArticulatedCollisionGeometry(URDF)
    kwargs = dict(condition=c, variant=v, persistent=True, navigation_ticks=NAVIGATION_TICKS,
        public_mission=public_mission(index))
    candidate = JointFloorRegisteredRoundTripController(model, geometry, **kwargs)
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    if len(reader.frames) != len(acquisitions) or len(reader.frames) != len(tape)+1:
        raise ValueError('complete paired observations and executed intervals required')
    first_command = first_terminal = None; frames = 0; last = None; prior = None
    with writer(OUTPUT) as append:
        for i, recorded in enumerate(read_rows(directory)):
            if recorded['tick'] != i: raise ValueError('consecutive original frames required')
            p, d, f, now = reader.packet(i)
            auxiliary = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
            old = recorded['decision']
            prior = old
            last = json.loads(json.dumps(candidate.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary)))
            try:
                change = compare_current(old, last, tick=i)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                write_json(OUTPUT/'terminal_decision_diagnostic.json', dict(tick=i,
                    candidate_decision=last, reason=repr(error), command_executed=False,
                    unexecuted_outcome_inferred=False))
                raise
            if i < len(tape) and old['requested_command'] != tape[i]['requested_command']:
                raise ValueError('original request differs from executed command tape')
            append(dict(tick=i, decision=last, predecessor_completed_raw_audit_verified=True, comparison=change))
            frames += 1
            if change['command_changed']: first_command = i
            if change['terminal_changed']: first_terminal = i
            if i % 100 == 0: print('JOINT_FLOOR_REGISTERED_PREFIX_FRAME', i, flush=True)
            # Never feed outcomes acquired after a request the candidate did not issue.
            if first_command is not None or first_terminal is not None: break
    if (state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters())
            or last is None): raise ValueError('unchanged learned model and nonempty replay required')
    return dict(case=name, model=model_name, frames=frames, model_state_sha256=before, model_state_unchanged=True,
        first_requested_command_difference=first_command, first_terminal_policy_difference=first_terminal,
        final_requested_command=last['requested_command'], prior_requested_command=prior['requested_command'],
        final_terminal=last['terminal'], changed_selection=last['new_selection'],
        final_floor_registration=None if last['evidence'] is None else last['evidence']['floor_registration'],
        completed_predecessor_raw_audit_reused=True, predecessor_controller_rerun=False,
        original_visual_evidence_exact=True, identical_raw_forecasts_when_both_controllers_plan=True,
        map_contact_residual_mission_pose_correction_is_the_declared_change=True,
        original_visibility_outcome_unchanged=True, prospective_observation_and_constraint_scope_only=True,
        stopped_before_unexecuted_outcome=first_command is not None or first_terminal is not None,
        unexecuted_outcomes_inferred=False, new_native_navigation_verified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-result-sha256', required=True)
    parser.add_argument('--readout-result-sha256', required=True)
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive floor-registration prefix required')
    bound = []
    for root, sha, status in ((INPUT, args.native_result_sha256, 'CONFIRMED_FLOOR_MAZE_PILOT_COMPLETE'),
            (READOUT, args.readout_result_sha256, 'CONFIRMED_FLOOR_MAZE_READOUT_COMPLETE')):
        verify_artifacts(root, {'result.json': sha}); result = read_json(root, 'result.json')
        if result['status'] != status: raise ValueError('completed predecessor and readout required')
        ids = {'result.json': sha, **result.get('artifact_sha256', {})}
        if 'launch_sha256' in result: ids['launch.json'] = result['launch_sha256']
        verify_artifacts(root, ids); bound.append((root, ids, result))
    if bound[1][2]['native_result_sha256'] != args.native_result_sha256:
        raise ValueError('readout must bind the same completed native result')
    audit = read_json(INPUT, CASE[0]+'_audit.json')
    if not all(audit[k] is True for k in ('raw_sensor_reconstruction_pass',
            'raw_model_command_replay_pass', 'raw_command_audit_pass', 'model_state_unchanged')):
        raise ValueError('completed predecessor full raw audit required before single-candidate replay')
    verify_artifacts(FAILED, FAILED_BINDINGS)
    failed_launch = read_json(FAILED, 'launch.json'); verify(failed_launch)
    old = read_json(INPUT, 'launch.json'); verify(old)
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_joint_floor_registered_maze_prefix_v1.py',
        'lewm/tests/test_joint_measured_floor_plane_development.py',
        'lewm/tests/test_joint_floor_registered_controller_development.py',
        'lewm/tests/test_joint_floor_registered_prefix_development.py',
        'docs/go2_joint_floor_registered_controller_candidate_2026-09-08.md',
        'docs/go2_floor_registered_prefix_failure_2026-09-08.md'), failed_launch['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 16*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+512*1024**2:
        raise ValueError('bounded single-candidate CPU replay resources unavailable')
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), hardware=resources,
        replay_input_bindings={str(p): ids for p, ids, _ in bound}, native_execution=False, model_training=False,
        shadow_replay_only=True, replay_workers=1, fresh_replays=1,
        completed_predecessor_raw_audit_reused=True, failed_prefix_bindings=FAILED_BINDINGS, failed_prefix_root=str(FAILED))
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); start = time.perf_counter()
    print('JOINT_FLOOR_REGISTERED_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = replay(launch['correction_admission'])
        verify(launch); verify_artifacts(FAILED, FAILED_BINDINGS)
        for root, ids, _ in bound: verify_artifacts(root, ids)
        verify_artifacts(FITS, launch['correction_admission']['base_admission']['fit_artifact_sha256'])
        verify_artifacts(CORRECTION, launch['correction_admission']['correction_artifact_sha256'])
        bindings = {n: digest(OUTPUT/n) for n in ('launch.json', DECISIONS)}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='JOINT_FLOOR_REGISTERED_MAZE_PREFIX_COMPLETE', source_sha256=sources,
            artifact_sha256=bindings, report=report, wall_s=time.perf_counter()-start, hardware_after=hardware(),
            native_execution=False, model_training=False, shadow_replay_only=True, navigation_qualified=False, goal_achieved=False))
        print('JOINT_FLOOR_REGISTERED_PREFIX_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_JOINT_FLOOR_REGISTERED_PREFIX_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
