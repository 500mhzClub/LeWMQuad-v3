"""Causal actual-packet planning-memory comparison; stop before any changed outcome."""
import argparse
from itertools import islice
import json
import shutil
import time
import cv2
import torch
from lewm.current_observation_planning_controller_development import CurrentObservationPlanningController
from lewm.current_observation_planning_prefix_development import compare_step
from lewm.measured_floor_transport_prefix_development import admit_native
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.replay_go2_reactive_floor_transport_prefix_v1 import (
    INPUT, INPUT_SHA, CANDIDATE_SOURCE, CANDIDATE_LAUNCH, CASE, verify_inputs)
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows, writer, NAME as DECISIONS
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT = BASE/'go2_current_observation_planning_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_current_observation_planning_prefix_v1_2026-09-09.md'
MAX_FRAMES = 64
MAX_OUTPUT_BYTES = 256*1024**2


def replay(launch):
    model,condition,variant = load_assigned(launch['correction_admission'], CASE[4])
    before = state_digest(model.state_dict())
    if (condition,variant) != (CASE[3],CASE[2]) or before != launch['model_state_sha256']:
        raise ValueError('exact assigned learned model required')
    controller = CurrentObservationPlanningController(model,ArticulatedCollisionGeometry(URDF),
        navigation_ticks=NAVIGATION_TICKS,public_mission=public_mission(0),
        condition=condition,variant=variant,persistent=True)
    directory = INPUT/CASE[0]; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory,'auxiliary_camera_audit.json'); tape = read_json(directory,'command_tape.json')
    if not len(reader.frames) == len(acquisitions) == len(tape)+1:
        raise ValueError('complete actual paired observations and command population required')
    first_command = first_terminal = None; frames = forecast_comparisons = 0; last = old = None
    with writer(OUTPUT) as append:
        for i,original in enumerate(islice(read_rows(directory),MAX_FRAMES)):
            if original['tick'] != i or original['pre_sample_index'] != 749+50*i or original['observation_index'] != i:
                raise ValueError('ordered actual observation endpoints required')
            if shutil.disk_usage(BASE).free < RESERVE_BYTES+MAX_OUTPUT_BYTES:
                raise ValueError('replay storage reserve unavailable')
            policy,depth,fast,now = reader.packet(i)
            image,auxiliary = packet(directory,i,policy,public_acquisition(acquisitions[i]),now_ns=now)
            inputs = fingerprint((policy,depth,fast,auxiliary,image,now))
            result = json.loads(json.dumps(controller.observe(policy,depth,fast,now_ns=now,
                auxiliary_depth=auxiliary,auxiliary_rgb=image),allow_nan=False))
            if fingerprint((policy,depth,fast,auxiliary,image,now)) != inputs:
                raise ValueError('controller mutated public input arrays')
            if not tape[i]['completed']: raise ValueError('original command was not completely dispatched')
            old = original['decision']; check = compare_step(old,result,tape[i]['requested_command'],frame=i)
            if check['requested_command_changed']: first_command = i
            if check['terminal_changed']: first_terminal = i
            forecast_comparisons += int(check['raw_model_forecasts_compared'])
            append(dict(tick=i,decision=result,comparison=check,
                original_requested_command=tape[i]['requested_command'],public_input_arrays_unchanged=True))
            frames += 1; last = result
            if (OUTPUT/DECISIONS).stat().st_size > MAX_OUTPUT_BYTES//2:
                raise ValueError('compressed replay output headroom exceeded')
            if i%16 == 0: print('CURRENT_OBSERVATION_PLANNING_PREFIX_FRAME',i,flush=True)
            if first_command is not None or first_terminal is not None: break
    if last is None or last['failure'] is not None:
        raise ValueError('nonempty replay without an internal comparator failure required')
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('unchanged weights and absent gradients required')
    return dict(case=CASE[0],frames=frames,maximum_frames=MAX_FRAMES,
        first_requested_command_difference=first_command,first_terminal_policy_difference=first_terminal,
        final_requested_command=last['requested_command'],prior_requested_command=old['requested_command'],
        final_terminal=last['terminal'],changed_selection=last['new_selection'],
        unchanged_observed_and_executed_residual_state_exact=True,
        raw_model_forecast_comparisons=forecast_comparisons,all_compared_raw_model_forecasts_exact=True,
        original_actual_commands_before_intervention_exact=True,
        stopped_at_first_command_or_terminal_difference=first_command is not None or first_terminal is not None,
        following_recorded_observations_consumed=False,public_input_arrays_unchanged=True,
        model_state_sha256=before,model_state_unchanged=True,
        accumulated_planning_cells_queried=False,persistent_contact_history_retained=True,
        selector_scan_state_retained=True,memoryless_controller=False,
        unexecuted_outcomes_inferred=False,native_execution=False,navigation_verified=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only',action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive planning-memory prefix required')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA}); native = read_json(INPUT,'result.json')
    ids = dict(native['artifact_sha256']); ids['result.json'] = INPUT_SHA; verify_artifacts(INPUT,ids)
    admit_native(native,read_json(INPUT,CASE[0]+'_audit.json'),case=CASE[0])
    verify_artifacts(CANDIDATE_SOURCE,{'launch.json':CANDIDATE_LAUNCH})
    current = read_json(CANDIDATE_SOURCE,'launch.json'); old = read_json(INPUT,'launch.json')
    for name,sha in old['source_sha256'].items():
        if current['source_sha256'].get(name) != sha: raise ValueError('incompatible frozen source: '+name)
    if current['correction_admission'] != old['correction_admission']:
        raise ValueError('unchanged training-only model admission required')
    sources = discover_sources((PROTOCOL,'scripts/replay_go2_current_observation_planning_prefix_v1.py',
        'docs/go2_current_observation_planning_component_v1_2026-09-09.md',
        'lewm/tests/test_current_observation_planning_development.py',
        'lewm/tests/test_current_observation_planning_prefix_development.py'),current['source_sha256'])
    keys = ('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules')
    launch = {k:current[k] for k in keys}
    launch.update(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),
        replay_input_bindings={str(INPUT):ids,str(CANDIDATE_SOURCE):{'launch.json':CANDIDATE_LAUNCH}},
        correction_admission=old['correction_admission'],model_state_sha256=old['prefix_report']['model_state_sha256'],
        implementation_class='CurrentObservationPlanningController',native_execution=False,
        model_loaded=True,model_training=False,shadow_replay_only=True,replay_workers=1,
        maximum_frames=MAX_FRAMES,opencv_threads=1,blas_threads=1,output_allowance_bytes=MAX_OUTPUT_BYTES,
        memory_admission_bytes=8*1024**3,minimum_free_bytes=RESERVE_BYTES,
        concurrency_reason='one ordered CPU model replay beside the existing native audit; no new scene',
        input_scope='completed eleventh episode; current launch is source/runtime identity only')
    verify_inputs(launch); resources = hardware(); launch['hardware'] = resources
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+MAX_OUTPUT_BYTES:
        raise ValueError('bounded planning-memory replay resources unavailable')
    if args.preflight_only:
        print('CURRENT_OBSERVATION_PLANNING_PREFIX_PREFLIGHT',json.dumps(dict(source_count=len(sources),
            hardware=resources,input_and_source_bindings_verified=True,output_created=False,native_execution=False)),flush=True)
        return
    create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch); started = time.perf_counter()
    print('CURRENT_OBSERVATION_PLANNING_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        report = replay(launch); verify_inputs(launch)
        bindings = {n:digest(OUTPUT/n) for n in ('launch.json',DECISIONS)}; verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='CURRENT_OBSERVATION_PLANNING_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=bindings,report=report,hardware_after=hardware(),
            wall_s=time.perf_counter()-started,model_loaded=True,model_training=False,native_execution=False,
            shadow_replay_only=True,navigation_qualified=False,goal_achieved=False))
        print('CURRENT_OBSERVATION_PLANNING_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),
            {k:v for k,v in report.items() if k != 'changed_selection'},flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_CURRENT_OBSERVATION_PLANNING_PREFIX_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
