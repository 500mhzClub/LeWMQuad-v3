"""Causal full-controller replay of the reached-frontier maze3 intervention."""
import argparse
from collections import deque
import json
import time
import numpy as np
import cv2
import torch

from lewm.recent_qualified_direct_flow_controller_development import RecentQualifiedDirectFlowController
from lewm.reached_frontier_recent_qualified_controller_development import ReachedFrontierRecentQualifiedController
from lewm.reached_frontier_prefix_comparison_development import compare_step
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import run_go2_recent_qualified_direct_flow_maze03_pilot_v1 as original
from scripts.all_phase_residual_maze02_native_inputs_development import completed, admit_native
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.maze_decision_stream_development import read_rows, writer, NAME
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.replay_go2_residual_current_observation_planning_prefix_v1 import state_tree
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_reached_frontier_maze03_prefix_v1_attempt_001'
SOURCE = 'scripts/replay_go2_reached_frontier_maze03_prefix_v1.py'
PROTOCOL = 'docs/go2_reached_frontier_maze03_prefix_v1_2026-09-10.md'
TEST = 'lewm/tests/test_reached_frontier_prefix_comparison_development.py'
INPUT_SHA = '330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723'
INPUT_LAUNCH_SHA = '25c34387e44b035f05a4f372ecf70cd337e875de508ba3a9b037ccf741da0eb7'
MODEL_SHA = '4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6'


def authenticate_inputs(sources):
    result, launch, ids = completed(original.OUTPUT, INPUT_SHA, INPUT_LAUNCH_SHA, sources)
    original.verify_inputs(launch)
    name = original.CASE[0]
    receipt = admit_native(result, launch, read_json(original.OUTPUT, name+'_audit.json'),
        read_json(original.OUTPUT, name+'_prefix_comparison.json'),
        read_json(original.OUTPUT, name+'_worker_terminal.json'))
    return dict(result_sha256=INPUT_SHA, original_completion=receipt,
        original_input_verifier_reexecuted=True)


def resources_for(resources):
    if resources['memory_available_bytes'] < 32*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('32GiB available RAM and 40+1GiB artifact envelope required')


def replay(launch):
    old_launch = read_json(original.OUTPUT, 'launch.json')
    name, index, variant, condition, model_name = original.CASE
    models = [load_assigned(old_launch['correction_admission'], model_name) for _ in range(2)]
    if any((c,v) != (condition,variant) or state_digest(m.state_dict()) != MODEL_SHA for m,c,v in models):
        raise ValueError('two fresh identical original assigned models required')
    geometry = ArticulatedCollisionGeometry(URDF)
    options = dict(public_mission=public_mission(index), navigation_ticks=NAVIGATION_TICKS,
        condition=condition, variant=variant, persistent=True)
    baseline = RecentQualifiedDirectFlowController(models[0][0], geometry, **options)
    candidate = ReachedFrontierRecentQualifiedController(models[1][0], geometry, **options)
    directory = original.OUTPUT/name
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    tape = read_json(directory, 'command_tape.json')
    changed = None; frames = forecasts = 0; last = None; reason = None
    frontier_seen = False; first_reached = first_decision_change = None
    with writer(OUTPUT) as append:
        for row in read_rows(directory):
            frame = row['tick']
            if frame >= len(tape) or not tape[frame]['completed']:
                raise ValueError('actual completed original command required at prefix boundary')
            if baseline.residual.pending != candidate.residual.pending:
                raise ValueError('full prior pending forecast differs before current observation')
            p, d, f, now = reader.packet(frame)
            image, auxiliary = packet(directory, frame, p, public_acquisition(acquisitions[frame]), now_ns=now)
            public_before = fingerprint((p,d,f,auxiliary,image,now))
            decisions = [controller.observe(p,d,f,now_ns=now,auxiliary_depth=auxiliary,auxiliary_rgb=image)
                for controller in (baseline,candidate)]
            old, new = decisions
            if json.loads(json.dumps(old)) != row['decision']:
                raise ValueError('fresh original controller does not reproduce complete raw decision: '+str(frame))
            check = compare_step(old,new,tape[frame]['requested_command'],frame=frame,
                frontier_previously_reached=frontier_seen)
            if check['frontier_reached_this_frame']:
                frontier_seen = True
                if first_reached is None: first_reached = frame
            if not check['normalized_complete_decision_exact'] and first_decision_change is None:
                first_decision_change = frame
            if public_before != fingerprint((p,d,f,auxiliary,image,now)):
                raise ValueError('public input bytes changed')
            if baseline.mapper.floor != candidate.mapper.floor or baseline.mapper.occupied != candidate.mapper.occupied:
                raise ValueError('accumulated observation cells changed')
            contact_states = [fingerprint(state_tree(c.memory)) for c in (baseline,candidate)]
            if contact_states[0] != contact_states[1]:
                raise ValueError('complete retained contact evidence differs: '+str(frame))
            append(dict(tick=frame,decision=new,original_requested_command=tape[frame]['requested_command'],
                original_complete_decision_reconstructed=True,comparison=check,
                public_input_sha256=public_before,public_input_arrays_unchanged=True,
                complete_retained_contact_state_sha256=contact_states[0],complete_retained_contact_state_equal=True))
            frames += 1; forecasts += int(check['raw_model_forecasts_compared']); last = new
            if check['requested_command_changed'] or check['terminal_changed']:
                changed = frame; reason = 'first_changed_request_or_terminal'; break
            if old['terminal'] is not None:
                reason = 'unchanged_original_terminal'; break
    if reason is None or last is None:
        raise ValueError('prefix ended without a declared causal boundary')
    for model,_,_ in models:
        if state_digest(model.state_dict()) != MODEL_SHA or any(p.grad is not None for p in model.parameters()):
            raise ValueError('model weights or gradients changed')
    return dict(case=name,frames=frames,raw_model_forecast_comparisons=forecasts,
        first_reached_frontier_frame=first_reached,first_normalized_decision_difference=first_decision_change,
        final_frontier_transition_receipt=last['last_frontier_transition_receipt'],
        first_request_or_terminal_difference=changed,stop_reason=reason,
        stopped_at_first_changed_request_or_terminal=changed is not None,
        following_recorded_observations_consumed=False,original_complete_decisions_reconstructed=True,
        unchanged_observed_and_executed_residual_state_exact=True,complete_retained_contact_state_equal=True,
        accumulated_observation_cells_unchanged=True,public_input_arrays_unchanged=True,
        final_requested_command=last['requested_command'],prior_requested_command=old['requested_command'],
        final_terminal=last['terminal'],changed_selection=last['new_selection'],
        model_state_sha256=MODEL_SHA,model_state_unchanged=True,native_execution=False,
        unexecuted_outcomes_inferred=False,memory_advantage_established=False,navigation_verified=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive prefix attempt; no retry/resume')
    verify_artifacts(original.OUTPUT,{'result.json':INPUT_SHA,'launch.json':INPUT_LAUNCH_SHA})
    result = read_json(original.OUTPUT,'result.json')
    inputs = result['artifact_sha256'] | {'result.json':INPUT_SHA}
    sources = discover_sources((SOURCE,PROTOCOL,TEST,
        'lewm/tests/test_reached_frontier_transition_development.py',
        'docs/go2_reached_frontier_transition_preparation_2026-09-10.md'),result['source_sha256']); verify(sources)
    resources = hardware(); resources_for(resources)
    if args.preflight_only:
        verify_artifacts(original.OUTPUT,inputs)
        print('REACHED_FRONTIER_PREFIX_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            complete_original_verifier_executed=False,model_loaded=False,native_execution=False,output_created=False)),flush=True)
        return
    print('REACHED_FRONTIER_INPUT_ADMISSION_STARTED',len(sources),flush=True)
    admission = authenticate_inputs(sources)
    if admission['result_sha256'] != INPUT_SHA: raise ValueError('exact original completed native episode required')
    verify(sources); resources = hardware(); resources_for(resources)
    create_output(OUTPUT)
    launch = dict(source_sha256=sources,input_artifact_sha256=inputs,input_result_sha256=INPUT_SHA,
        input_admission=admission,output_root=str(OUTPUT),protocol=PROTOCOL,hardware=resources,
        model_state_sha256=MODEL_SHA,native_execution=False,model_training=False,
        stop_at_first_changed_request_or_terminal=True,new_independent_layout_consumed=False)
    write_json(OUTPUT/'launch.json',launch)
    print('REACHED_FRONTIER_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    try:
        report = replay(launch)
        verify(sources); verify_artifacts(original.OUTPUT,inputs)
        final = authenticate_inputs(sources)
        if final != admission: raise ValueError('complete original admission changed')
        ids = {n:digest(artifact_path(OUTPUT,n)) for n in ('launch.json',NAME)}
        verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='REACHED_FRONTIER_MAZE03_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,wall_s=time.perf_counter()-started,
            model_loaded=True,model_training=False,native_execution=False,goal_achieved=False))
        print('REACHED_FRONTIER_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),
            'frames',report['frames'],'changed',report['first_request_or_terminal_difference'],flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_REACHED_FRONTIER_PREFIX_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
