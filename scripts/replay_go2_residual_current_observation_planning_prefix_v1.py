"""Causal original maze2 prefix for the residual planning-map comparator."""
import argparse
from collections import deque
import json
import time
import numpy as np
import cv2
import torch

from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.residual_current_observation_planning_controller_development import ResidualCurrentObservationPlanningController
from lewm.residual_current_observation_planning_prefix_development import compare_step
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import run_go2_residual_anchored_continuation_maze_pilot_v1 as original
from scripts.run_go2_prepared_native_queue_v1 import JOBS, authenticate_completed
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.maze_decision_stream_development import read_rows, writer, NAME
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_residual_current_observation_planning_prefix_v1_attempt_001'
SOURCE = 'scripts/replay_go2_residual_current_observation_planning_prefix_v1.py'
PROTOCOL = 'docs/go2_residual_current_observation_planning_v1_2026-09-10.md'
TEST = 'lewm/tests/test_residual_current_observation_planning_development.py'
INPUT_SHA = '818a598ca6336866cf5f4768c11edaf67c8c1ca60896c305f93f69fd0ed5230c'
INPUT_LAUNCH_SHA = 'c4681e31baaf5dc1c8fa368854ddbcf19866090787741b946c4d37b9a3a477b3'
MODEL_SHA = '4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6'


def state_tree(value):
    if isinstance(value, np.ndarray): return value
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:state_tree(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, deque)): return [state_tree(v) for v in value]
    if isinstance(value, set): return [state_tree(v) for v in sorted(value)]
    if value is None or type(value) in (bool, int, float, str): return value
    if hasattr(value, '__dict__'):
        return dict(type=type(value).__module__+'.'+type(value).__name__, fields=state_tree(vars(value)))
    raise ValueError('unsupported exact contact-state fingerprint type: '+str(type(value)))


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
    baseline = ResidualAnchoredContinuationController(models[0][0], geometry, **options)
    candidate = ResidualCurrentObservationPlanningController(models[1][0], geometry, **options)
    directory = original.OUTPUT/name
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    tape = read_json(directory, 'command_tape.json')
    changed = None; frames = forecasts = 0; last = None; reason = None
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
            check = compare_step(old,new,tape[frame]['requested_command'],frame=frame)
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
    sources = discover_sources((SOURCE,PROTOCOL,TEST),result['source_sha256']); verify(sources)
    resources = hardware(); resources_for(resources)
    if args.preflight_only:
        verify_artifacts(original.OUTPUT,inputs)
        print('RESIDUAL_CURRENT_PLANNING_PREFIX_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            complete_original_verifier_executed=False,model_loaded=False,native_execution=False,output_created=False)),flush=True)
        return
    admission = authenticate_completed(JOBS[1],sources,expected_launch_sha=INPUT_LAUNCH_SHA)
    if admission['result_sha256'] != INPUT_SHA: raise ValueError('exact original completed native episode required')
    verify(sources); resources = hardware(); resources_for(resources)
    create_output(OUTPUT)
    launch = dict(source_sha256=sources,input_artifact_sha256=inputs,input_result_sha256=INPUT_SHA,
        input_admission=admission,output_root=str(OUTPUT),protocol=PROTOCOL,hardware=resources,
        model_state_sha256=MODEL_SHA,native_execution=False,model_training=False,
        stop_at_first_changed_request_or_terminal=True,new_independent_layout_consumed=False)
    write_json(OUTPUT/'launch.json',launch)
    print('RESIDUAL_CURRENT_PLANNING_PREFIX_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    try:
        report = replay(launch)
        verify(sources); verify_artifacts(original.OUTPUT,inputs)
        final = authenticate_completed(JOBS[1],sources,expected_launch_sha=INPUT_LAUNCH_SHA)
        if final != admission: raise ValueError('complete original admission changed')
        ids = {n:digest(artifact_path(OUTPUT,n)) for n in ('launch.json',NAME)}
        verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='RESIDUAL_CURRENT_OBSERVATION_PLANNING_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,wall_s=time.perf_counter()-started,
            model_loaded=True,model_training=False,native_execution=False,goal_achieved=False))
        print('RESIDUAL_CURRENT_PLANNING_PREFIX_COMPLETE',digest(OUTPUT/'result.json'),
            'frames',report['frames'],'changed',report['first_request_or_terminal_difference'],flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_RESIDUAL_CURRENT_PLANNING_PREFIX_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
