"""Actual public-sensor replay until the first unexecuted nominal recovery."""
import json
import time
import cv2
import torch
from lewm.nominal_reentry_round_trip_controller_development import NominalReentryRoundTripController
from lewm.nominal_clearance_reentry_development import reenter
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows, writer, NAME as DECISIONS
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_novel_maze_round_trip_pilot_v1 import OUTPUT as INPUT, CASE, CORRECTION, FITS
from scripts.read_go2_novel_maze_round_trip_pilot_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE/'go2_nominal_reentry_maze_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_nominal_reentry_maze_prefix_v1_2026-09-08.md'
INPUT_SHA = '5874c1fea08b40d90e676b69acb7d3a103d96a6281e0911801bf6bd6e1ab1570'
READOUT_SHA = 'fe2b0fbdadba8f1ae29e67d8897d2ea3bdfe36ec5caa8e4a4a4200a349087bc4'


def replay(admission):
    name, index, variant, condition, model_name = CASE; directory = INPUT/name
    model, c, v = load_assigned(admission, model_name); assert (c, v) == (condition, variant)
    before = state_digest(model.state_dict())
    controller = NominalReentryRoundTripController(model, ArticulatedCollisionGeometry(URDF),
        condition=c, variant=v, persistent=True, navigation_ticks=NAVIGATION_TICKS, public_mission=public_mission(index))
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    assert len(reader.frames) == len(acquisitions) == len(tape)+1
    first_command = first_terminal = None; frames = 0; changed_selection = None; last = None
    with writer(OUTPUT) as append:
        for i, original in enumerate(read_rows(directory)):
            p, d, f, now = reader.packet(i)
            auxiliary = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
            r = json.loads(json.dumps(controller.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary)))
            old = original['decision']; a = old['new_selection']; b = r['new_selection']
            for key in ('evidence', 'memory_receipt', 'observed_goal_distance_m', 'auxiliary_floor_partition_receipt', 'mission_receipt'):
                assert r[key] == old[key], ('unchanged causal observation/map/mission', i, key)
            assert (a is None) == (b is None)
            if a is not None:
                mapper = controller.mapper; B = mapper.map_from_initial
                expected = reenter(a, B@mapper.surface.position, B@mapper.surface.rotation, mapper.occupied)
                assert b == expected, ('exact reentry policy transformation', i)
                for key in ('prediction', 'surface_checks', 'nominal_action_checks', 'nominal_path_checks', 'phase_allowed_actions'):
                    if key in a: assert b[key] == a[key], ('original forecast/constraint preserved', i, key)
            if old['terminal'] != r['terminal']: first_terminal = i
            if i < len(tape) and r['requested_command'] != tape[i]['requested_command']: first_command = i
            if first_command is None and first_terminal is None:
                for key in old:
                    if key != 'controller': assert r[key] == old[key], ('exact pre-intervention decision', i, key)
            append(dict(tick=i, decision=r)); frames += 1; last = r
            if i % 100 == 0: print('NOMINAL_REENTRY_PREFIX_FRAME', i, flush=True)
            if first_command is not None or first_terminal is not None:
                changed_selection = b
                break
    assert state_digest(model.state_dict()) == before and all(p.grad is None for p in model.parameters())
    assert last is not None and last['failure'] is None
    return dict(case=name, model=model_name, frames=frames, model_state_sha256=before,
        model_state_unchanged=True, first_requested_command_difference=first_command,
        first_terminal_policy_difference=first_terminal, final_requested_command=last['requested_command'],
        final_terminal=last['terminal'], prior_requested_command=tape[frames-1]['requested_command'] if frames-1 < len(tape) else None,
        causal_observations_maps_mission_forecasts_and_constraints_exact=True,
        exact_original_decisions_before_intervention=True, reentry_candidates=None if changed_selection is None else changed_selection.get('reentry_candidates'),
        reentry_current_clearance=None if changed_selection is None else changed_selection.get('reentry_current_clearance'),
        stopped_before_unexecuted_outcome=first_command is not None or first_terminal is not None,
        unexecuted_outcomes_inferred=False, native_recovery_verified=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive nominal-reentry replay required')
    bound = []
    for root, sha, status in ((INPUT, INPUT_SHA, 'NOVEL_MAZE_ROUND_TRIP_PILOT_COMPLETE'),
            (READOUT, READOUT_SHA, 'NOVEL_MAZE_ROUND_TRIP_READOUT_COMPLETE')):
        verify_artifacts(root, {'result.json': sha}); result = read_json(root, 'result.json'); assert result['status'] == status
        ids = {'result.json': sha, **result.get('artifact_sha256', {})}
        if 'launch_sha256' in result: ids['launch.json'] = result['launch_sha256']
        verify_artifacts(root, ids); bound.append((root, ids, result))
    old = read_json(INPUT, 'launch.json'); verify(old)
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_nominal_reentry_maze_prefix_v1.py',
        'lewm/tests/test_nominal_clearance_reentry_development.py',
        'docs/go2_novel_maze_round_trip_pilot_result_2026-09-08.md'), bound[-1][2]['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+256*1024**2:
        raise ValueError('replay resource allowance unavailable')
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), hardware=resources,
        replay_input_bindings={str(p): ids for p, ids, _ in bound}, native_execution=False, model_training=False,
        shadow_replay_only=True, replay_workers=1, fresh_replays=1)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('NOMINAL_REENTRY_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = replay(launch['correction_admission'])
        verify(launch)
        for root, ids, _ in bound: verify_artifacts(root, ids)
        verify_artifacts(FITS, launch['correction_admission']['base_admission']['fit_artifact_sha256'])
        verify_artifacts(CORRECTION, launch['correction_admission']['correction_artifact_sha256'])
        bindings = {n: digest(OUTPUT/n) for n in ('launch.json', DECISIONS)}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='NOMINAL_REENTRY_MAZE_PREFIX_COMPLETE', source_sha256=sources,
            artifact_sha256=bindings, report=report, wall_s=time.perf_counter()-started, hardware_after=hardware(),
            native_execution=False, model_training=False, shadow_replay_only=True,
            navigation_qualified=False, goal_achieved=False))
        print('NOMINAL_REENTRY_PREFIX_COMPLETE', digest(OUTPUT/'result.json'), report, flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_NOMINAL_REENTRY_PREFIX_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
