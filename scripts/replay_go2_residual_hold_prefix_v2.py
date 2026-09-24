"""Prospective hold reconsideration on a completed residual native trajectory."""
import argparse
from itertools import islice
import json
import shutil
import time
import cv2
import torch
from lewm.residual_hold_feasibility_controller_development import ResidualHoldFeasibilityController
from lewm.residual_hold_prefix_development import compare_step, MAX_FRAMES
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.run_go2_residual_first_interval_maze_pilot_v1 import OUTPUT as INPUT, CASE, verify_inputs as verify_native
from scripts.read_go2_residual_first_interval_maze_pilot_v1 import admit_results, LEARNED, LEARNED_CASE, LEARNED_SHA
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows, writer, NAME as DECISIONS
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT = BASE/'go2_residual_hold_prefix_v2_attempt_001'
PROTOCOL = 'docs/go2_residual_hold_prefix_v2_2026-09-09.md'
MAX_OUTPUT_BYTES = 2*1024**3


FAILED = BASE/'go2_residual_hold_prefix_v1_attempt_001'
FAILED_BINDINGS = {
    'launch.json':'b344a274f4649943c24d0e6350c5d8cf00a2820a5cb6283ad9ba3e6dd222a725',
    DECISIONS:'b7d386890457e4c29e91660ddf2b19c72f4ac151daf3c7713673786bd753b33d',
    'failure.json':'548bad1eef69df9c2afd51accd3df9d2db17ef7aaf90939c3a45b990cd55e085',
}


def verify_failed_predecessor():
    from scripts.replay_go2_residual_hold_prefix_v1 import verify_inputs as verify_failed
    verify_artifacts(FAILED, FAILED_BINDINGS)
    launch = read_json(FAILED, 'launch.json'); verify_failed(launch)
    failure = read_json(FAILED, 'failure.json')
    if failure != dict(status='TERMINAL_RESIDUAL_HOLD_PREFIX_FAILURE',
            reason="ValueError('compressed replay output headroom exceeded')"):
        raise ValueError('exact preserved predecessor output-headroom failure required')
    count = 0
    for i, row in enumerate(read_rows(FAILED)):
        c = row['comparison']
        if (i >= 1185 or row['tick'] != i or row['decision']['tick'] != i
                or c['requested_command_changed'] is not False
                or c['hold_reconsideration_changed_action'] is not False
                or c['complete_original_selection_preserved'] is not True
                or c['unchanged_observed_mission_and_residual_state_exact'] is not True):
            raise ValueError('all preserved unchanged V1 decisions required')
        count += 1
    if count != 1185: raise ValueError('complete 1185-row failed output required')
    return launch['source_sha256']


def verify_inputs(launch):
    verify(launch)
    verify_artifacts(INPUT, launch['replay_input_bindings'])
    verify_native(read_json(INPUT, 'launch.json'))
    verify_failed_predecessor()


def replay(launch):
    model, condition, variant = load_assigned(launch['correction_admission'], CASE[4])
    before = state_digest(model.state_dict())
    if (condition, variant) != (CASE[3], CASE[2]) or before != MODEL_STATE:
        raise ValueError('exact original assigned model required')
    controller = ResidualHoldFeasibilityController(model, ArticulatedCollisionGeometry(URDF),
        navigation_ticks=NAVIGATION_TICKS, public_mission=public_mission(2),
        condition=condition, variant=variant, persistent=True)
    directory = INPUT/CASE[0]; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    if not len(reader.frames) == len(acquisitions) == len(tape)+1 or len(tape) < MAX_FRAMES:
        raise ValueError('complete paired observations and bounded actual command population required')
    first_command = None
    frames = forecasts = 0; last = old = None
    with writer(OUTPUT) as append:
        for i, original in enumerate(islice(read_rows(directory), MAX_FRAMES)):
            if original['tick'] != i or original['pre_sample_index'] != 749+50*i or original['observation_index'] != i:
                raise ValueError('ordered actual observation endpoints required')
            if shutil.disk_usage(BASE).free < RESERVE_BYTES+MAX_OUTPUT_BYTES:
                raise ValueError('replay storage reserve unavailable')
            policy, depth, fast, now = reader.packet(i)
            image, auxiliary = packet(directory, i, policy, public_acquisition(acquisitions[i]), now_ns=now)
            inputs = fingerprint((policy, depth, fast, auxiliary, image, now))
            result = json.loads(json.dumps(controller.observe(policy, depth, fast, now_ns=now,
                auxiliary_depth=auxiliary, auxiliary_rgb=image), allow_nan=False))
            if fingerprint((policy, depth, fast, auxiliary, image, now)) != inputs:
                raise ValueError('controller mutated public input arrays')
            if not tape[i]['completed']: raise ValueError('original command was not completely dispatched')
            old = original['decision']
            check = compare_step(old, result, tape[i]['requested_command'], frame=i)
            if check['requested_command_changed']: first_command = i
            forecasts += int(check['raw_model_forecasts_compared'])
            append(dict(tick=i, decision=result, comparison=check,
                original_requested_command=tape[i]['requested_command'], public_input_arrays_unchanged=True))
            frames += 1; last = result
            if (OUTPUT/DECISIONS).stat().st_size > MAX_OUTPUT_BYTES//2:
                raise ValueError('compressed replay output headroom exceeded')
            if i%32 == 0: print('RESIDUAL_HOLD_PREFIX_FRAME', i, flush=True)
            if first_command is not None or old['terminal'] is not None: break
    if last is None or last['failure'] is not None:
        raise ValueError('nonempty replay without an internal comparator failure required')
    if first_command is None and old['terminal'] is None and frames != MAX_FRAMES:
        raise ValueError('complete unchanged replay prefix required; truncated decisions rejected')
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('unchanged weights and absent gradients required')
    return dict(case=CASE[0], frames=frames, maximum_frames=MAX_FRAMES,
        first_requested_command_difference=first_command,
        hold_reconsideration_interventions=int(first_command is not None),
        final_requested_command=last['requested_command'], prior_requested_command=old['requested_command'],
        final_terminal=last['terminal'], changed_selection=last['new_selection'],
        raw_model_forecast_comparisons=forecasts, complete_original_selection_preserved=True,
        unchanged_observed_mission_and_residual_state_exact=True,
        original_actual_commands_before_intervention_exact=True,
        stopped_at_first_command_or_terminal_difference=first_command is not None,
        following_recorded_observations_consumed=False, public_input_arrays_unchanged=True,
        model_state_sha256=before, model_state_unchanged=True,
        unexecuted_outcomes_inferred=False, native_execution=False, navigation_verified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive hold reconsideration prefix required')
    verify_artifacts(INPUT, {'result.json':args.native_result_sha256}); result = read_json(INPUT, 'result.json')
    ids = dict(result['artifact_sha256']); ids['result.json'] = args.native_result_sha256
    verify_artifacts(INPUT, ids); old = read_json(INPUT, 'launch.json'); verify_native(old)
    verify_artifacts(LEARNED, {'result.json':LEARNED_SHA}); learned = read_json(LEARNED, 'result.json')
    admission = admit_results(result, old, learned, read_json(LEARNED, 'launch.json'),
        read_json(INPUT, CASE[0]+'_audit.json'), read_json(LEARNED, LEARNED_CASE[0]+'_audit.json'))
    inherited = verify_failed_predecessor()
    for name, sha in result['source_sha256'].items():
        if inherited.get(name) != sha: raise ValueError('conflicting native/predecessor source: '+name)
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_residual_hold_prefix_v2.py',
        'lewm/tests/test_residual_hold_feasibility_development.py',
        'lewm/tests/test_residual_hold_prefix_v2_development.py'), inherited)
    launch = old|dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), replay_input_bindings=ids,
        native_result_sha256=args.native_result_sha256, completed_native_admission=admission,
        model_state_sha256=MODEL_STATE, implementation_class='ResidualHoldFeasibilityController',
        residual_hold_feasibility_enabled=True, failed_predecessor_artifact_sha256=dict(FAILED_BINDINGS),
        scientific_policy_and_replay_unchanged=True, fresh_replay_from_observation_zero=True,
        native_execution=False, native_scene_workers=0, model_loaded=True, model_training=False,
        shadow_replay_only=True, replay_workers=1, maximum_frames=MAX_FRAMES,
        opencv_threads=1, blas_threads=1, output_allowance_bytes=MAX_OUTPUT_BYTES,
        memory_admission_bytes=8*1024**3, minimum_free_bytes=RESERVE_BYTES,
        concurrency_reason='one CPU replay may overlap one independently owned native scene with measured headroom',
        input_scope='completed raw-audited residual maze2; stop at first changed command')
    verify_inputs(launch); resources = hardware(); launch['hardware'] = resources
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+MAX_OUTPUT_BYTES:
        raise ValueError('bounded hold replay resources unavailable')
    if args.preflight_only:
        print('RESIDUAL_HOLD_PREFIX_PREFLIGHT', json.dumps(dict(source_count=len(sources),
            hardware=resources, input_and_source_bindings_verified=True, output_created=False, native_execution=False)), flush=True)
        return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('RESIDUAL_HOLD_PREFIX_V2_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = replay(launch); verify_inputs(launch)
        bindings = {n:digest(OUTPUT/n) for n in ('launch.json', DECISIONS)}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='RESIDUAL_HOLD_PREFIX_V2_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, report=report, hardware_after=hardware(),
            native_result_sha256=args.native_result_sha256, wall_s=time.perf_counter()-started,
            model_loaded=True, model_training=False, native_execution=False,
            shadow_replay_only=True, navigation_qualified=False, goal_achieved=False))
        print('RESIDUAL_HOLD_PREFIX_V2_COMPLETE', digest(OUTPUT/'result.json'),
            {k:v for k,v in report.items() if k != 'changed_selection'}, flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_RESIDUAL_HOLD_PREFIX_V2_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
