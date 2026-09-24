"""Same supervised model and actual public prefix; contact-cost horizon only."""
import argparse
from itertools import islice
import json
import shutil
import time
import cv2
import torch
from lewm.commitment_contact_controller_development import CommitmentContactController
from lewm.commitment_contact_prefix_development import PrefixComparison, MAX_FRAMES
from lewm.measured_floor_transport_development import current_measured_floor_pose
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.supervised_rollout_maze_study_development import SUPERVISED_STATE, planned_cases
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.read_go2_supervised_maze01_turn_scores_v1 import admit, INPUT, CASE, FIXED, OUTPUT as DIAGNOSTIC
from scripts.run_go2_supervised_rollout_mazes_v1 import verify_inputs as verify_native_context
from scripts.partial_floor_height_scoped_verification_admission_development import admit_benchmark, BENCHMARK_SHA
from scripts.scoped_verification_digest_development import verify_with_scoped_digests
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

OUTPUT = BASE/'go2_supervised_commitment_contact_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_supervised_commitment_contact_prefix_v1_2026-09-09.md'
SOURCE = 'scripts/replay_go2_supervised_commitment_contact_prefix_v1.py'
DIAGNOSTIC_SHA = 'e546494d75770c174453663d14cb5d38754a8f994bf89ba847ab425dbd3ddf8f'
MAX_OUTPUT_BYTES = 4*1024**3
MEMORY_BYTES = 8*1024**3


def verify_input_context(launch):
    verify(launch); verify_artifacts(INPUT, launch['replay_input_bindings'])
    old, record, audit, bindings = admit()
    if (launch['replay_input_bindings'] != bindings or launch['correction_admission'] != old['correction_admission']
            or launch['model_state_sha256'] != SUPERVISED_STATE or launch['planned_case'] != list(planned_cases()[0])):
        raise ValueError('complete completed first worker and same assigned model required')
    verify_artifacts(DIAGNOSTIC, {'result.json': DIAGNOSTIC_SHA})
    diagnostic = read_json(DIAGNOSTIC, 'result.json')
    verify_artifacts(DIAGNOSTIC, diagnostic['artifact_sha256'])
    from scripts.run_go2_successive_choice_maze_development_v1 import verify as verify_sources
    verify_sources(diagnostic['source_sha256'])
    verify_native_context(old)


def verify_inputs(launch):
    if (any(launch['replay_input_bindings'].get(k) != v for k, v in FIXED.items())
            or launch['diagnostic_result_sha256'] != DIAGNOSTIC_SHA
            or launch['verification_benchmark_result_sha256'] != BENCHMARK_SHA):
        raise ValueError('fixed first-worker, diagnostic and scoped benchmark identities required')
    admit_benchmark(); before = fingerprint(launch)
    result, counters = verify_with_scoped_digests(verify_input_context, digest, launch)
    if result is not None or fingerprint(launch) != before: raise ValueError('unchanged original verifier context required')
    print('COMMITMENT_CONTACT_INPUTS_VERIFIED', counters, flush=True)


def replay(launch):
    case = planned_cases()[0]
    model, condition, variant = load_assigned(launch['correction_admission'], case[4])
    before = state_digest(model.state_dict())
    if (condition, variant) != ('supervised_rollout', 'full') or before != SUPERVISED_STATE:
        raise ValueError('exact same assigned supervised model required')
    controller = CommitmentContactController(model, ArticulatedCollisionGeometry(URDF),
        navigation_ticks=NAVIGATION_TICKS, public_mission=public_mission(1), condition=condition, variant=variant, persistent=True)
    directory = INPUT/CASE; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    if len(reader.frames) != 3014 or len(acquisitions) != 3014 or len(tape) != 3013:
        raise ValueError('complete original supervised maze1 population required')
    comparator = PrefixComparison(); frames = forecasts = 0; last = check = None
    with writer(OUTPUT) as append:
        for i, original in enumerate(islice(read_rows(directory), MAX_FRAMES)):
            if original['tick'] != i or original['observation_index'] != i or original['pre_sample_index'] != 749+50*i:
                raise ValueError('ordered actual observation endpoints required')
            if shutil.disk_usage(BASE).free < RESERVE_BYTES+MAX_OUTPUT_BYTES: raise ValueError('replay storage reserve unavailable')
            p, d, f, now = reader.packet(i)
            image, aux = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
            public = fingerprint((p, d, f, image, aux, now))
            live = controller.observe(p, d, f, now_ns=now, auxiliary_depth=aux, auxiliary_rgb=image)
            last = json.loads(json.dumps(live, allow_nan=False))
            try:
                if fingerprint((p, d, f, image, aux, now)) != public: raise ValueError('controller mutated public arrays')
                if tape[i]['completed'] is not True: raise ValueError('completed original command required')
                check = comparator.compare(original['decision'], last, tape[i]['requested_command'], frame=i)
                raw = live['original_visual_evidence']
                if raw['status'] == 'CURRENT_VISUAL_POSE':
                    current_dual_camera_pose(raw, p, image, aux, identity=(0, 0, 0), now_ns=now)
                if live['terminal'] is None: current_measured_floor_pose(live['evidence'], identity=(0, 0, 0), now_ns=now)
            except Exception as error:
                append(dict(tick=i, decision=last, comparison_failure=repr(error), original_requested_command=tape[i]['requested_command']))
                raise
            append(dict(tick=i, decision=last, comparison=check, original_requested_command=tape[i]['requested_command'],
                public_input_arrays_unchanged=True))
            frames += 1; forecasts += int(check['raw_model_forecasts_compared'])
            if (OUTPUT/DECISIONS).stat().st_size > MAX_OUTPUT_BYTES//2: raise ValueError('compressed output headroom exceeded')
            if i % 32 == 0: print('COMMITMENT_CONTACT_PREFIX_FRAME', i, flush=True)
            if check['stop']: break
    if not frames or not check['stop']: raise ValueError('complete prefix until intervention or either terminal required')
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('unchanged model weights and absent gradients required')
    return dict(case=CASE, layout_index=1, frames=frames, maximum_frames=MAX_FRAMES,
        raw_model_forecast_comparisons=forecasts, first_requested_command_difference=comparator.first_command_difference,
        final_requested_command=last['requested_command'], prior_requested_command=original['decision']['requested_command'],
        final_terminal=last['terminal'], prior_terminal=original['decision']['terminal'], final_failure=last['failure'],
        boundary_comparison=check, complete_selection_transform_verified_every_frame=True,
        original_actual_commands_before_intervention_exact=True, all_shared_observed_state_exact=True,
        stopped_at_first_changed_command_or_either_terminal=True, following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True, model_state_sha256=before, model_state_unchanged=True,
        scored_pose_horizon_ns=100_000_000, scored_contact_horizon_ns=100_000_000, path_constraint_horizon_ns=800_000_000,
        contact_scores_calibrated=False, unexecuted_outcomes_inferred=False, native_execution=False, navigation_verified=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--diagnostic-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive new contact-horizon replay required')
    if args.diagnostic_result_sha256 != DIAGNOSTIC_SHA: raise ValueError('exact all-selection diagnostic required')
    resources = hardware()
    if resources['memory_available_bytes'] < MEMORY_BYTES+32*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+MAX_OUTPUT_BYTES:
        raise ValueError('replay plus concurrent native resource allowances unavailable')
    old, record, audit, bindings = admit(); benchmark = admit_benchmark()
    inherited = dict(old['source_sha256'])
    verify_artifacts(DIAGNOSTIC, {'result.json': DIAGNOSTIC_SHA}); diagnostic = read_json(DIAGNOSTIC, 'result.json')
    for source in (benchmark, diagnostic):
        for name, sha in source['source_sha256'].items():
            if name in inherited and inherited[name] != sha: raise ValueError('inherited source conflict: '+name)
            inherited[name] = sha
    sources = discover_sources((PROTOCOL, SOURCE, 'lewm/tests/test_commitment_contact_development.py',
        'lewm/tests/test_commitment_contact_replay_development.py',
        'docs/go2_supervised_maze01_turn_scores_result_2026-09-09.md'), inherited)
    keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules')
    launch = {k: old[k] for k in keys}
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), replay_input_bindings=bindings,
        diagnostic_result_sha256=DIAGNOSTIC_SHA, verification_benchmark_result_sha256=BENCHMARK_SHA,
        correction_admission=old['correction_admission'], model_state_sha256=SUPERVISED_STATE,
        planned_case=list(planned_cases()[0]), implementation_class='CommitmentContactController',
        native_execution=False, model_loaded=True, model_training=False, shadow_replay_only=True,
        replay_workers=1, native_scene_workers=0, maximum_frames=MAX_FRAMES, opencv_threads=1, blas_threads=1,
        output_allowance_bytes=MAX_OUTPUT_BYTES, memory_admission_bytes=MEMORY_BYTES,
        concurrent_native_allowance_bytes=32*1024**3, minimum_free_bytes=RESERVE_BYTES, hardware_before=resources,
        input_scope='completed supervised maze1; stop before any changed-command future or either terminal',
        contact_penalty_coefficient_m=1.2, scored_pose_horizon_ns=100_000_000,
        scored_contact_horizon_ns=100_000_000, path_constraint_horizon_ns=800_000_000,
        observer_mission_memory_and_feasibility_unchanged=True, native_pose_used=False, contact_scores_calibrated=False)
    verify_inputs(launch); resources = hardware(); launch['hardware'] = resources
    if resources['memory_available_bytes'] < MEMORY_BYTES+32*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+MAX_OUTPUT_BYTES:
        raise ValueError('resource allowances unavailable after verification')
    if args.preflight_only:
        print('COMMITMENT_CONTACT_PREFIX_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            input_and_source_bindings_verified=True, output_created=False, native_execution=False)), flush=True); return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('COMMITMENT_CONTACT_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = replay(launch); verify_inputs(launch)
        outputs = {n: digest(OUTPUT/n) for n in ('launch.json', DECISIONS)}; verify_artifacts(OUTPUT, outputs)
        write_json(OUTPUT/'result.json', dict(status='SUPERVISED_COMMITMENT_CONTACT_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=outputs, report=report, diagnostic_result_sha256=DIAGNOSTIC_SHA,
            hardware_after=hardware(), wall_s=time.perf_counter()-started, model_loaded=True, model_training=False,
            native_execution=False, shadow_replay_only=True, navigation_qualified=False, goal_achieved=False))
        print('COMMITMENT_CONTACT_PREFIX_COMPLETE', digest(OUTPUT/'result.json'), report, flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_COMMITMENT_CONTACT_PREFIX_FAILURE', reason=repr(error))); raise


if __name__ == '__main__': main()
