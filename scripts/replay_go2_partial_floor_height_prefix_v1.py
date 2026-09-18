"""Fresh controller replay through the original frame-504 floor failure only."""
import argparse
from copy import deepcopy
from itertools import islice
import json
import shutil
import time
import cv2
import torch
from lewm.partial_floor_height_controller_development import PartialHeightDirectFlowController
from lewm.partial_floor_height_prefix_development import compare_step, validate_live, MAX_FRAMES, BOUNDARY_FRAME
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.diagnose_go2_direct_flow_maze01_floor_conflict_v1 import (
    INPUT, CASE, admit, verify_native, OUTPUT as DIAGNOSIS, verify_all as verify_diagnosis)
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

OUTPUT = BASE/'go2_partial_floor_height_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_partial_floor_height_prefix_v1_2026-09-09.md'
INPUT_SHA = 'd6774bae22cb9effeb0cd85ae255de203de1539541f701d57788b58ab00769de'
DIAGNOSIS_SHA = 'd2512d3241e38ac608aa2185cb62e5aafdd4fc13db962fffc68e45d82eaa992e'
MAX_OUTPUT_BYTES = 512*1024**2
MEMORY_BYTES = 8*1024**3


def verify_inputs(launch):
    verify(launch)
    verify_artifacts(INPUT, launch['replay_input_bindings'])
    verify_native(read_json(INPUT, 'launch.json'))
    verify_artifacts(DIAGNOSIS, launch['diagnosis_bindings'])
    verify_diagnosis(read_json(DIAGNOSIS, 'launch.json'))


def replay(launch):
    model, condition, variant = load_assigned(launch['correction_admission'], CASE[4])
    before = state_digest(model.state_dict())
    if (condition, variant) != (CASE[3], CASE[2]) or before != MODEL_STATE:
        raise ValueError('same frozen assigned model required')
    controller = PartialHeightDirectFlowController(model, ArticulatedCollisionGeometry(URDF),
        navigation_ticks=NAVIGATION_TICKS, public_mission=public_mission(1),
        condition=condition, variant=variant, persistent=True)
    directory = INPUT/CASE[0]; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    if len(reader.frames) != 515 or len(acquisitions) != 515 or len(tape) != 514:
        raise ValueError('complete original paired population required')
    frames = forecasts = exact = 0; last = check = None
    with writer(OUTPUT) as append:
        for i, original in enumerate(islice(read_rows(directory), MAX_FRAMES)):
            if original['tick'] != i or original['pre_sample_index'] != 749+50*i or original['observation_index'] != i:
                raise ValueError('ordered original observation endpoints required')
            if shutil.disk_usage(BASE).free < RESERVE_BYTES+MAX_OUTPUT_BYTES:
                raise ValueError('replay storage reserve unavailable')
            policy, depth, fast, now = reader.packet(i)
            image, auxiliary = packet(directory, i, policy, public_acquisition(acquisitions[i]), now_ns=now)
            inputs = fingerprint((policy, depth, fast, auxiliary, image, now))
            anchor = deepcopy(controller.registration.anchor)
            live = controller.observe(policy, depth, fast, now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image)
            last = json.loads(json.dumps(live, allow_nan=False))
            try:
                if fingerprint((policy, depth, fast, auxiliary, image, now)) != inputs:
                    raise ValueError('controller mutated public inputs')
                if tape[i]['completed'] is not True: raise ValueError('completed original request required')
                check = compare_step(original['decision'], last, tape[i]['requested_command'], frame=i)
                validate_live(live, last, check, policy, image, auxiliary, now_ns=now, prior_anchor=anchor)
                if check['partial_height_admitted'] and controller.registration.anchor != anchor:
                    raise ValueError('partial height cannot replace the full registration anchor')
            except Exception as error:
                append(dict(tick=i, decision=last, comparison_failure=repr(error),
                    original_requested_command=tape[i]['requested_command']))
                raise
            forecasts += int(check['raw_model_forecasts_compared']); exact += int(check['complete_original_decision_exact'])
            append(dict(tick=i, decision=last, comparison=check,
                original_requested_command=tape[i]['requested_command'], public_input_arrays_unchanged=True))
            frames += 1
            if (OUTPUT/DECISIONS).stat().st_size > MAX_OUTPUT_BYTES//2:
                raise ValueError('compressed replay output headroom exceeded')
            if i%32 == 0: print('PARTIAL_FLOOR_HEIGHT_PREFIX_FRAME', i, flush=True)
            if check['stop']: break
    if frames != MAX_FRAMES or not check or not check['boundary_reached']:
        raise ValueError('complete prefix through original failure required')
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('unchanged model weights and absent gradients required')
    return dict(case=CASE[0], layout_index=1, frames=frames, maximum_frames=MAX_FRAMES,
        boundary_frame=BOUNDARY_FRAME, exact_original_decisions=exact, raw_model_forecast_comparisons=forecasts,
        prior_commands_compared=BOUNDARY_FRAME, original_actual_commands_before_intervention_exact=True,
        boundary_comparison=check, final_requested_command=last['requested_command'],
        final_terminal=last['terminal'], final_failure=last['failure'],
        full_controller_recovered_at_boundary=check['controller_recovered'],
        partial_height_receipt=(last['evidence'] or {}).get('partial_floor_height'),
        stopped_at_original_failed_observation=True, following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True, raw_visual_tracker_unchanged=True,
        model_state_sha256=before, model_state_unchanged=True, unexecuted_outcomes_inferred=False,
        native_execution=False, navigation_verified=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fresh height prefix required')
    verify_artifacts(INPUT, {'result.json':INPUT_SHA}); result = read_json(INPUT, 'result.json')
    ids = dict(result['artifact_sha256']); ids['result.json'] = INPUT_SHA; verify_artifacts(INPUT, ids)
    old = read_json(INPUT, 'launch.json'); admit(result, old, read_json(INPUT, CASE[0]+'_audit.json'))
    verify_artifacts(DIAGNOSIS, {'result.json':DIAGNOSIS_SHA}); diagnosis = read_json(DIAGNOSIS, 'result.json')
    diagnosis_ids = dict(diagnosis['artifact_sha256']); diagnosis_ids['result.json'] = DIAGNOSIS_SHA
    inherited = dict(result['source_sha256'])
    for name, sha in diagnosis['source_sha256'].items():
        if name in inherited and inherited[name] != sha: raise ValueError('unchanged predecessor sources required: '+name)
        inherited[name] = sha
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_partial_floor_height_prefix_v1.py',
        'lewm/tests/test_partial_floor_height_development.py',
        'lewm/tests/test_partial_floor_height_prefix_development.py'), inherited)
    keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules')
    launch = {k:old[k] for k in keys}
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), replay_input_bindings=ids,
        diagnosis_bindings=diagnosis_ids, native_result_sha256=INPUT_SHA, diagnosis_result_sha256=DIAGNOSIS_SHA,
        correction_admission=old['correction_admission'], model_state_sha256=MODEL_STATE, planned_case=list(CASE),
        implementation_class='PartialHeightDirectFlowController', native_execution=False, model_loaded=True,
        model_training=False, shadow_replay_only=True, replay_workers=1, maximum_frames=MAX_FRAMES,
        opencv_threads=1, blas_threads=1, output_allowance_bytes=MAX_OUTPUT_BYTES,
        memory_admission_bytes=MEMORY_BYTES, minimum_free_bytes=RESERVE_BYTES,
        input_scope='completed tracking maze1 initial state through frame504 only; no following outcome')
    verify_inputs(launch); resources = hardware(); launch['hardware'] = resources
    if resources['memory_available_bytes'] < MEMORY_BYTES or resources['artifact_free_bytes'] < RESERVE_BYTES+MAX_OUTPUT_BYTES:
        raise ValueError('bounded height replay resources unavailable')
    if args.preflight_only:
        print('PARTIAL_FLOOR_HEIGHT_PREFIX_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            input_and_source_bindings_verified=True, output_created=False, native_execution=False)), flush=True)
        return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('PARTIAL_FLOOR_HEIGHT_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = replay(launch); verify_inputs(launch)
        bindings = {n:digest(OUTPUT/n) for n in ('launch.json', DECISIONS)}; verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='PARTIAL_FLOOR_HEIGHT_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, report=report, hardware_after=hardware(),
            wall_s=time.perf_counter()-started, model_loaded=True, model_training=False,
            native_execution=False, shadow_replay_only=True, navigation_qualified=False, goal_achieved=False))
        print('PARTIAL_FLOOR_HEIGHT_PREFIX_COMPLETE', digest(OUTPUT/'result.json'),
            {k:v for k,v in report.items() if k != 'partial_height_receipt'}, flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_PARTIAL_FLOOR_HEIGHT_PREFIX_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
