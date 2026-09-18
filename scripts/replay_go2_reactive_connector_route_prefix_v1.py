"""Nearer-connector reactive replay, stopped before a changed action outcome."""
import json
import time
import cv2
import torch
from lewm.reactive_connector_round_trip_controller_development import ReactiveConnectorRoundTripController
from lewm.reactive_route_connector_development import nearer_route_target
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows, writer, NAME as DECISIONS
from scripts.run_go2_reactive_nominal_maze_pilot_v1 import OUTPUT as INPUT, CASE
from scripts.read_go2_reactive_nominal_maze_pilot_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE/'go2_reactive_connector_route_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_reactive_connector_route_prefix_v1_2026-09-08.md'
INPUT_SHA = '1a3bf1e0f796d8fe5ae7b0b11feb837b7299064f83f6b9f44457d0040a229dd6'
READOUT_SHA = '4aeea872d44ec90f93d5ff29ce6e144af4b08add3394db0c94bd006c3593f2af'


def replay():
    name, index = CASE, 0; directory = INPUT/name
    controller = ReactiveConnectorRoundTripController(ArticulatedCollisionGeometry(URDF),
        navigation_ticks=NAVIGATION_TICKS, public_mission=public_mission(index))
    assert not hasattr(controller, 'model') and not hasattr(controller, 'residual')
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    assert len(reader.frames) == len(acquisitions) == len(tape)+1
    first_command = first_terminal = None; frames = 0; last = None
    with writer(OUTPUT) as append:
        for i, original in enumerate(read_rows(directory)):
            policy, depth, fast, now = reader.packet(i)
            auxiliary = packet(directory, i, policy, public_acquisition(acquisitions[i]), now_ns=now)
            result = json.loads(json.dumps(controller.observe(policy, depth, fast, now_ns=now, auxiliary_depth=auxiliary)))
            old = original['decision']; selection = result['new_selection']
            for key in ('evidence', 'memory_receipt', 'observed_goal_distance_m', 'auxiliary_floor_partition_receipt', 'mission_receipt'):
                assert result[key] == old[key], ('same causal observed state before intervention', i, key)
            assert not result['learned_model_used'] and not result['candidate_future_outcomes_evaluated']
            assert set(result) == set(old) | {'nearer_observed_route_target_policy_enabled'}
            assert result['nearer_observed_route_target_policy_enabled'] is True
            assert (selection is None) == (old['new_selection'] is None)
            if selection:
                B = controller.mapper.map_from_initial
                expected = nearer_route_target(old['new_selection'], B@controller.mapper.surface.position,
                    B@controller.mapper.surface.rotation, controller.mapper.floor, controller.mapper.occupied)
                assert selection == expected, ('exact nearer-route transformation', i)
                for key in ('proposal', 'current_nominal_clearance', 'current_surface_check'):
                    assert selection[key] == old['new_selection'][key], ('same current geometry and observed route', i, key)
                assert 'prediction' not in selection and 'nominal_path_checks' not in selection
                assert selection['current_geometry_checked'] and not selection['predictive_surface_or_path_gates_applied']
            if result['terminal'] != old['terminal']: first_terminal = i
            if i < len(tape) and result['requested_command'] != tape[i]['requested_command']: first_command = i
            if first_command is None and first_terminal is None:
                for key in old:
                    if key not in ('controller', 'new_selection'):
                        assert result[key] == old[key], ('same complete pre-intervention decision', i, key)
            append(dict(tick=i, decision=result)); frames += 1; last = result
            if i % 100 == 0: print('REACTIVE_CONNECTOR_ROUTE_PREFIX_FRAME', i, flush=True)
            if first_command is not None or first_terminal is not None: break
    assert last is not None and last['failure'] is None
    return dict(case=name, frames=frames, first_requested_command_difference=first_command,
        first_terminal_policy_difference=first_terminal, final_requested_command=last['requested_command'],
        prior_requested_command=tape[frames-1]['requested_command'] if frames-1 < len(tape) else None,
        final_terminal=last['terminal'], changed_selection=last['new_selection'],
        causal_observations_maps_and_mission_exact=True, original_command_outputs_before_intervention_exact=True,
        exact_other_decision_fields_before_intervention=True, measured_route_and_current_geometry_exact=True,
        learned_model_used=False, candidate_future_outcomes_evaluated=False,
        future_constraint_gates_matched=False, measured_geometry_baseline_rule_explicit=True,
        same_nonpredictive_current_geometry_gates=True, original_nominal_radius_preserved=True,
        stopped_before_unexecuted_outcome=first_command is not None or first_terminal is not None,
        unexecuted_outcomes_inferred=False, navigation_verified=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive reactive prefix required')
    bound = []
    for root, sha, status in ((INPUT, INPUT_SHA, 'REACTIVE_NOMINAL_MAZE_PILOT_COMPLETE'),
            (READOUT, READOUT_SHA, 'REACTIVE_NOMINAL_MAZE_READOUT_COMPLETE')):
        verify_artifacts(root, {'result.json': sha}); result = read_json(root, 'result.json'); assert result['status'] == status
        ids = {'result.json': sha, **result.get('artifact_sha256', {})}
        if 'launch_sha256' in result: ids['launch.json'] = result['launch_sha256']
        verify_artifacts(root, ids); bound.append((root, ids, result))
    old = read_json(INPUT, 'launch.json'); verify(old)
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_reactive_connector_route_prefix_v1.py',
        'lewm/tests/test_reactive_route_connector_development.py',
        'docs/go2_reactive_nominal_maze_pilot_result_2026-09-08.md'), bound[-1][2]['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+256*1024**2:
        raise ValueError('bounded reactive replay resources unavailable')
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), hardware=resources,
        replay_input_bindings={str(p): ids for p, ids, _ in bound}, native_execution=False, model_training=False,
        learned_model_used=False, shadow_replay_only=True, replay_workers=1, fresh_replays=1,
        concurrency_reason='one ordered CPU replay alongside separately owned live native scene; no new scene or checkpoint')
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('REACTIVE_CONNECTOR_ROUTE_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = replay(); verify(launch)
        for root, ids, _ in bound: verify_artifacts(root, ids)
        bindings = {n: digest(OUTPUT/n) for n in ('launch.json', DECISIONS)}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='REACTIVE_CONNECTOR_ROUTE_PREFIX_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, report=report,
            wall_s=time.perf_counter()-started, hardware_after=hardware(),
            native_execution=False, learned_model_used=False, model_training=False,
            shadow_replay_only=True, navigation_qualified=False, goal_achieved=False))
        print('REACTIVE_CONNECTOR_ROUTE_PREFIX_COMPLETE', digest(OUTPUT/'result.json'),
            {k: v for k, v in report.items() if k != 'changed_selection'}, flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_REACTIVE_CONNECTOR_ROUTE_PREFIX_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
