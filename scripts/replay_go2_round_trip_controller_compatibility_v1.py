"""Recorded native sensor/model check of the variable mission controller.

Only the existing single-goal instruction and budget are used for compatibility.
No round-trip or longer-budget native outcome is inferred from these packets.
"""
import json
import time
import cv2
import torch
from lewm.observed_round_trip_controller_development import ObservedRoundTripController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.auxiliary_downward45_packet_replay_development import packet, public_acquisition
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_causal_residual_final_goal_probe_v1 import OUTPUT as INPUT, CASES, CORRECTION, FITS
from scripts.read_go2_causal_residual_final_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE/'go2_round_trip_controller_compatibility_v1_attempt_001'
PROTOCOL = 'docs/go2_round_trip_controller_compatibility_v1_2026-09-08.md'
INPUT_SHA = '42211a96a2be46429b1e6cb92872acc033f84573411968d04d0d5ea8afc2c484'
READOUT_SHA = '2c9f30fd78434d87cb5bac8f5ffb2d18faaf59051165601e9338d596c3886087'


def replay(admission, case):
    name, trial, variant, condition, model_name = case; directory = INPUT/name
    model, c, v = load_assigned(admission, model_name)
    assert (c, v) == (condition, variant)
    before = state_digest(model.state_dict())
    controller = ObservedRoundTripController(model, ArticulatedCollisionGeometry(URDF),
        condition=c, variant=v, persistent=True, navigation_ticks=240,
        public_mission=dict(goal_initial_body_xy_m=[1.2, 0.], return_initial_body_xy_m=[0., 0.],
            require_return_after_goal=False))
    reader = IntentReturnRGBDReplay(directory); original = read_json(directory, 'context_decisions.json')
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    assert len(reader.frames) == len(original) == len(acquisitions) == len(tape)+1
    records = []
    for i in range(len(reader.frames)):
        p, d, f, now = reader.packet(i)
        auxiliary = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
        r = json.loads(json.dumps(controller.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary)))
        old = original[i]['decision']
        for key in old:
            if key != 'controller':
                assert r[key] == old[key], ('original recorded decision field differs', name, i, key)
        if i < len(tape): assert r['requested_command'] == tape[i]['requested_command']
        assert not r['verified_round_trip'] and r['mission_receipt']['phase'] == 'OUTBOUND'
        records.append(r)
    assert state_digest(model.state_dict()) == before and all(p.grad is None for p in model.parameters())
    return dict(decisions=records, model_state_sha256=before, frames=len(records),
        complete_recorded_execution_replayed=True,
        all_original_decision_fields_except_controller_identity_exact=True,
        actual_recorded_commands_exact=True, model_state_unchanged=True,
        return_mission_executed=False, unexecuted_outcomes_inferred=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive compatibility replay required')
    bound = []
    for root, sha, status in ((INPUT, INPUT_SHA, 'CAUSAL_RESIDUAL_FINAL_GOAL_PROBE_COMPLETE'),
            (READOUT, READOUT_SHA, 'CAUSAL_RESIDUAL_FINAL_GOAL_READOUT_COMPLETE')):
        verify_artifacts(root, {'result.json': sha}); result = read_json(root, 'result.json')
        assert result['status'] == status
        ids = {'result.json': sha, **result.get('artifact_sha256', {})}
        if 'launch_sha256' in result: ids['launch.json'] = result['launch_sha256']
        verify_artifacts(root, ids); bound.append((root, ids))
    old = read_json(INPUT, 'launch.json'); verify(old); admission = old['correction_admission']
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_round_trip_controller_compatibility_v1.py',
        'docs/go2_observed_round_trip_controller_source_2026-09-08.md',
        'docs/go2_causal_residual_final_goal_probe_result_2026-09-08.md',
        'lewm/tests/test_observed_round_trip_controller_development.py',
        'lewm/tests/test_observed_round_trip_mission_development.py',
        'lewm/tests/test_mission_target_waypoint_selection_development.py'), old['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+256*1024**2:
        raise ValueError('replay resource allowance unavailable')
    launch = old | dict(source_sha256=sources, output_root=str(OUTPUT), protocol=PROTOCOL, hardware=resources,
        replay_input_bindings={str(p): ids for p, ids in bound}, native_execution=False, model_training=False,
        shadow_replay_only=True, complete_fresh_replays_per_model=1, workers=1)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('ROUND_TRIP_CONTROLLER_COMPATIBILITY_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        reports = []; names = ['launch.json']
        for case in CASES:
            result = replay(admission, case)
            name = case[4]+'_decisions.json'; write_json(OUTPUT/name, result); names.append(name)
            report = {k: v for k, v in result.items() if k != 'decisions'} | dict(case=case[0], model=case[4])
            reports.append(report); print('ROUND_TRIP_CONTROLLER_COMPATIBILITY_MODEL', report, flush=True)
        verify(launch)
        for root, ids in bound: verify_artifacts(root, ids)
        verify_artifacts(CORRECTION, admission['correction_artifact_sha256'])
        verify_artifacts(FITS, admission['base_admission']['fit_artifact_sha256'])
        bindings = {name: digest(OUTPUT/name) for name in names}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='ROUND_TRIP_CONTROLLER_COMPATIBILITY_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, conditions=reports,
            wall_s=time.perf_counter()-started, hardware_after=hardware(), native_execution=False,
            model_training=False, shadow_replay_only=True, return_mission_executed=False,
            independent_maze_evaluation=False, navigation_qualified=False, goal_achieved=False))
        print('ROUND_TRIP_CONTROLLER_COMPATIBILITY_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_ROUND_TRIP_CONTROLLER_COMPATIBILITY_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
