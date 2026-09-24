"""Actual sensor replay for the shared-state reactive comparator; no new scene."""
import argparse
from copy import deepcopy
from itertools import islice
import json
import time
import cv2
import torch
from lewm.reactive_floor_transport_controller_development import ReactiveFloorTransportController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows, writer, NAME as DECISIONS
from scripts.run_go2_dual_camera_settled_maze_pilot_v1 import OUTPUT as INPUT, CASE
from scripts.run_go2_measured_floor_transport_maze_pilot_v1 import OUTPUT as CANDIDATE_SOURCE
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE/'go2_reactive_floor_transport_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_reactive_floor_transport_prefix_v1_2026-09-09.md'
INPUT_SHA = '44710966178a57b31f7da3bec10ad4f21710bcff3701b0a1038750f1ef6d747c'
CANDIDATE_LAUNCH = '8dbd36ce4c300bef2b42b31cb6c04d5633624163684f1a5e88de5fcbb8002369'
MAX_FRAMES = 64


def compare_shared(candidate, original):
    for key in ('evidence', 'original_visual_evidence', 'memory_receipt',
            'observed_goal_distance_m', 'auxiliary_floor_partition_receipt'):
        if candidate[key] != original[key]: raise ValueError('shared observed state differs: '+key)
    mission = deepcopy(candidate['mission_receipt'])
    if mission is not None and mission.get('observed_settling') is not None:
        receipt = mission['observed_settling']
        if receipt['motion_source'] != 'consecutive_admitted_visual_positions_in_floor_reference':
            raise ValueError('exact current mission pose-source label required')
        receipt['motion_source'] = 'consecutive_admitted_floor_registered_visual_positions'
    if mission != original['mission_receipt']:
        raise ValueError('shared physical settling mission differs')


def verify_inputs(launch):
    verify(launch)
    for root, bindings in launch['replay_input_bindings'].items():
        verify_artifacts(root, bindings)


def replay():
    directory = INPUT/CASE[0]
    controller = ReactiveFloorTransportController(ArticulatedCollisionGeometry(URDF),
        navigation_ticks=NAVIGATION_TICKS, public_mission=public_mission(0))
    if hasattr(controller, 'model') or hasattr(controller, 'residual'):
        raise ValueError('reactive comparator must not instantiate a learned model or forecast residual')
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    tape = read_json(directory, 'command_tape.json')
    if not len(reader.frames) == len(acquisitions) == len(tape)+1:
        raise ValueError('complete paired acquisition and actual command population required')
    first_command = first_terminal = None; frames = 0; last = old = None
    with writer(OUTPUT) as append:
        for i, original in enumerate(islice(read_rows(directory), MAX_FRAMES)):
            if original['tick'] != i or original['pre_sample_index'] != 749+50*i:
                raise ValueError('actual ordered observation endpoints required')
            policy, depth, fast, now = reader.packet(i)
            image, auxiliary = packet(directory, i, policy, public_acquisition(acquisitions[i]), now_ns=now)
            result = json.loads(json.dumps(controller.observe(policy, depth, fast, now_ns=now,
                auxiliary_depth=auxiliary, auxiliary_rgb=image)))
            old = original['decision']; compare_shared(result, old)
            if old['requested_command'] != tape[i]['requested_command'] or not tape[i]['completed']:
                raise ValueError('original command must have been physically dispatched')
            if result['learned_model_used'] or result['candidate_future_outcomes_evaluated']:
                raise ValueError('reactive action rule must not use a model or candidate future outcomes')
            selected = result['new_selection']
            if selected is not None:
                if ('prediction' in selected or 'nominal_path_checks' in selected
                        or selected['candidate_future_outcomes_evaluated']
                        or selected['predictive_surface_or_path_gates_applied']
                        or not selected['current_geometry_checked']):
                    raise ValueError('explicit current-geometry reactive action rule required')
            if result['requested_command'] != tape[i]['requested_command']: first_command = i
            if result['terminal'] != old['terminal']: first_terminal = i
            append(dict(tick=i, decision=result, shared_observed_state_exact=True,
                original_requested_command=tape[i]['requested_command'],
                current_requested_command_matches_original=first_command is None))
            frames += 1; last = result
            if i % 16 == 0: print('REACTIVE_FLOOR_TRANSPORT_PREFIX_FRAME', i, flush=True)
            # The observation following a changed command was generated by the
            # old policy and cannot be used as this comparator's outcome.
            if first_command is not None or first_terminal is not None: break
    if last is None or last['failure'] is not None:
        raise ValueError('nonempty replay without an internal comparator failure required')
    return dict(case=CASE[0], frames=frames, maximum_frames=MAX_FRAMES,
        first_requested_command_difference=first_command, first_terminal_policy_difference=first_terminal,
        final_requested_command=last['requested_command'], prior_requested_command=old['requested_command'],
        final_terminal=last['terminal'], changed_selection=last['new_selection'],
        causal_observations_maps_and_settling_mission_exact=True,
        mission_comparison_normalization='validated_pose_source_wording_only',
        original_actual_commands_before_intervention_exact=True,
        stopped_at_first_command_or_terminal_difference=first_command is not None or first_terminal is not None,
        following_recorded_observations_consumed=False, learned_model_used=False,
        candidate_future_outcomes_evaluated=False, learned_residual_used=False,
        isolated_prediction_ranking_ablation=False, future_constraint_gates_matched=False,
        unexecuted_outcomes_inferred=False, native_execution=False, navigation_verified=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive new reactive prefix required')
    verify_artifacts(INPUT, {'result.json':INPUT_SHA}); result = read_json(INPUT, 'result.json')
    if result['status'] != 'DUAL_CAMERA_SETTLED_MAZE_PILOT_V1_COMPLETE' or len(result['conditions']) != 1:
        raise ValueError('completed eleventh native episode required')
    ids = {'result.json':INPUT_SHA, **result['artifact_sha256']}; verify_artifacts(INPUT, ids)
    audit = read_json(INPUT, CASE[0]+'_audit.json')
    for key in ('raw_sensor_reconstruction_pass', 'raw_model_command_replay_pass', 'raw_command_audit_pass', 'model_state_unchanged'):
        if audit[key] is not True: raise ValueError('completed original raw audit required: '+key)
    verify_artifacts(CANDIDATE_SOURCE, {'launch.json':CANDIDATE_LAUNCH})
    current = read_json(CANDIDATE_SOURCE, 'launch.json'); old = read_json(INPUT, 'launch.json')
    for name, sha in old['source_sha256'].items():
        if current['source_sha256'].get(name) != sha: raise ValueError('incompatible frozen source: '+name)
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_reactive_floor_transport_prefix_v1.py',
        'lewm/tests/test_reactive_floor_transport_controller_development.py',
        'lewm/tests/test_reactive_floor_transport_prefix_development.py'), current['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+256*1024**2:
        raise ValueError('bounded reactive prefix resources unavailable')
    keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules')
    launch = {k:current[k] for k in keys}
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), hardware=resources,
        replay_input_bindings={str(INPUT):ids, str(CANDIDATE_SOURCE):{'launch.json':CANDIDATE_LAUNCH}},
        implementation_class='ReactiveFloorTransportController', native_execution=False,
        model_loaded=False, model_training=False, learned_model_used=False, shadow_replay_only=True,
        replay_workers=1, maximum_frames=MAX_FRAMES, opencv_threads=1, blas_threads=1,
        concurrency_reason='one ordered CPU replay beside the separately owned native scene; no new scene or model',
        input_scope='completed eleventh native observations only; current launch is source identity evidence')
    verify_inputs(launch)
    if args.preflight_only:
        print('REACTIVE_FLOOR_TRANSPORT_PREFIX_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            input_and_source_bindings_verified=True, output_created=False, native_execution=False)), flush=True)
        return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('REACTIVE_FLOOR_TRANSPORT_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = replay(); verify_inputs(launch)
        bindings = {name:digest(OUTPUT/name) for name in ('launch.json', DECISIONS)}
        verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='REACTIVE_FLOOR_TRANSPORT_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, report=report, hardware_after=hardware(),
            wall_s=time.perf_counter()-started, model_loaded=False, native_execution=False,
            shadow_replay_only=True, navigation_qualified=False, goal_achieved=False))
        print('REACTIVE_FLOOR_TRANSPORT_PREFIX_COMPLETE', digest(OUTPUT/'result.json'),
            {k:v for k,v in report.items() if k != 'changed_selection'}, flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_REACTIVE_FLOOR_TRANSPORT_PREFIX_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
