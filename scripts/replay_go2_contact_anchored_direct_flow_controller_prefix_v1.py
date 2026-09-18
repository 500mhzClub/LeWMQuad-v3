"""Two complete original-model controllers through the measured contact tracking boundary."""
import argparse
from contextlib import closing
from itertools import islice
import json
import os
from pathlib import Path
import time

import cv2
import torch

from lewm.direct_flow_commitment_contact_controller_development import DirectFlowCommitmentContactController, CONTROLLER, FLAG
from lewm.commitment_contact_anchored_controller_development import CommitmentContactAnchoredController
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose
from lewm.measured_floor_transport_development import current_measured_floor_pose
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import replay_go2_contact_anchored_direct_flow_observer_prefix_v1 as observer
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT

SOURCE = 'scripts/replay_go2_contact_anchored_direct_flow_controller_prefix_v1.py'
TEST = 'lewm/tests/test_contact_anchored_direct_flow_controller_prefix_development.py'
PROTOCOL = 'docs/go2_contact_anchored_direct_flow_controller_prefix_v1_2026-09-11.md'
OUTPUT = BASE/'go2_contact_anchored_direct_flow_controller_prefix_v1_attempt_001'
OBSERVER_SHA = '116ed76d30cf1e9bcbcca01e91618f1a95cfa3c949ec7cd79ec5c6fe5eb56f90'
OBSERVER_LAUNCH = '9ee4ddd13e6672619c4fabb29fcacf9831fbea215377915a2151d6e25da76e62'
OBSERVER_OWNER = dict(pid=2819628, created=1789118183.27, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', observer.SOURCE])
MODEL_SHA = '755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5'
native = observer.probe.diagnosis.native
CASE = native.CASE
BOUNDARY = 561
MAX_OUTPUT_BYTES = 1024**3


def compare(original, candidate, actual, expected_visual, *, frame):
    if (type(frame) is not int or not 0 <= frame <= BOUNDARY
            or original['controller'] != 'commitment_contact_anchored_continuation_controller_v1'
            or candidate['controller'] != CONTROLLER or candidate.get(FLAG) is not True
            or original['requested_command'] != actual):
        raise ValueError('exact assigned original controller and executed command required')
    if candidate['original_visual_evidence'] != expected_visual:
        raise ValueError('complete previously reproduced observer evidence required')
    normalized = dict(candidate); normalized.pop(FLAG); normalized['controller'] = original['controller']
    if frame < BOUNDARY:
        if (normalized != original or original['terminal'] is not None or original['tick'] != frame
                or original['failure'] is not None):
            raise ValueError('complete original decision must match before the observer boundary')
        return dict(stop=False, complete_original_decision_exact=True,
            original_forecast_compared=bool(original['new_selection'] and 'prediction' in original['new_selection']),
            full_controller_recovered=False, requested_command_changed=False)
    old = original['original_visual_evidence']; raw = candidate['original_visual_evidence']
    fallback = raw.get('direct_corner_flow_fallback') or {}
    if (original['terminal'] != 'SENSOR_OR_MODEL_FAILURE' or original['tick'] != frame-1
            or old['status'] != 'VISUAL_TERMINAL_FAILURE'
            or fallback.get('accepted') is not True
            or fallback['original_camera_selection'] != old['camera_selection']
            or fallback['original_auxiliary_continuity'] != old['continuity_evidence']
            or fallback['original_reference_selection'] != old['reference_selection']):
        raise ValueError('exact original failure and authenticated recovered observer boundary required')
    recovered = candidate['terminal'] is None and candidate['failure'] is None
    if recovered:
        selection = candidate['new_selection']
        if (candidate['tick'] != frame or candidate['evidence'] is None
                or raw['status'] != 'CURRENT_VISUAL_POSE' or not selection or 'prediction' not in selection):
            raise ValueError('current registered pose and complete forecast selection required')
        action = selection['action']
        requested = [0.,0.,0.] if action is None else list(candidate_commands(action)[0])
        if candidate['requested_command'] != requested:
            raise ValueError('boundary command must match the selected action')
    elif candidate['terminal'] is None or candidate['requested_command'] != [0.,0.,0.]:
        raise ValueError('unrecovered full controller must stop with a zero command')
    return dict(stop=True, complete_original_decision_exact=False, original_forecast_compared=False,
        full_controller_recovered=recovered, requested_command_changed=candidate['requested_command'] != actual)


def owners_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('original replay boot required')
    if owner_live(OBSERVER_OWNER) or owner_live(observer.previous.RAW_OWNER):
        raise ValueError('original observer and sustained raw replay must have ended')


def prepare():
    verify_artifacts(observer.OUTPUT, {'result.json':OBSERVER_SHA, 'launch.json':OBSERVER_LAUNCH})
    result = read_json(observer.OUTPUT, 'result.json'); launch = read_json(observer.OUTPUT, 'launch.json')
    report = result['report']
    if (result['status'] != 'CONTACT_ANCHORED_DIRECT_FLOW_OBSERVER_PREFIX_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or report != read_json(observer.OUTPUT, 'report.json')
            or report['frames'] != BOUNDARY+1 or report['candidate_exact_original_frames'] != BOUNDARY
            or report['boundary']['frame'] != BOUNDARY or report['boundary_fallback']['accepted'] is not True
            or report['boundary']['candidate_status'] != 'CURRENT_VISUAL_POSE'
            or report['following_recorded_observations_consumed'] is not False):
        raise ValueError('complete accepted original contact observer prefix required')
    verify(result['source_sha256']); verify_artifacts(observer.OUTPUT, result['artifact_sha256'])
    sources = discover_sources((SOURCE, TEST, PROTOCOL), result['source_sha256']); verify(sources)
    return sources, {'result.json':OBSERVER_SHA} | result['artifact_sha256']


def resources():
    hw = observer.original.resources()
    if hw['memory_available_bytes'] < 48*1024**3 or hw['artifact_free_bytes'] < 41*1024**3 or hw['physical_cpus'] < 4:
        raise ValueError('16GiB replay plus 32GiB native RAM, 41GiB disk and four physical CPUs required')
    return hw


def verify_inputs(sources, observer_ids):
    owners_ended(); verify(sources); verify_artifacts(observer.OUTPUT, observer_ids)
    inputs = observer.admit_worker(sources)
    launch = read_json(native.OUTPUT, 'launch.json'); record = read_json(native.OUTPUT, CASE[0]+'_worker_terminal.json')
    if (launch['model_state_sha256'] != MODEL_SHA or record['model_state_sha256'] != MODEL_SHA
            or launch['planned_case'] != list(CASE) or CASE[2:4] != ('full', 'supervised_rollout')):
        raise ValueError('original completed contact worker and full supervised model required')
    return inputs


def replay():
    launch = read_json(native.OUTPUT, 'launch.json')
    models = [native.assigned_model(launch) for _ in range(2)]
    if any(state_digest(m.state_dict()) != MODEL_SHA for m in models):
        raise ValueError('two exact original assigned model copies required')
    options = dict(public_mission=public_mission(2), navigation_ticks=NAVIGATION_TICKS,
        condition=CASE[3], variant=CASE[2], persistent=True)
    controllers = [cls(model, ArticulatedCollisionGeometry(URDF), **options) for cls,model in
        zip((CommitmentContactAnchoredController, DirectFlowCommitmentContactController), models, strict=True)]
    directory = native.OUTPUT/CASE[0]; reader = observer.IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    frames = forecasts = 0
    with observer.writer(OUTPUT) as append, closing(observer.read_rows(directory)) as originals, closing(observer.read_rows(observer.OUTPUT)) as expected:
        for row, witness in zip(islice(originals, BOUNDARY+1), expected, strict=True):
            frame = frames
            if (row['tick'] != frame or witness['tick'] != frame or row['observation_index'] != frame
                    or row['pre_sample_index'] != 749+50*frame or tape[frame]['tick'] != frame
                    or tape[frame]['completed'] is not True or tape[frame]['pre_sample_index'] != 749+50*frame
                    or tape[frame]['post_sample_index'] != 799+50*frame):
                raise ValueError('ordered original actual command endpoints required')
            p,d,fast,now = reader.packet(frame)
            image,aux = observer.packet(directory,frame,p,observer.public_acquisition(acquisitions[frame]),now_ns=now)
            before = observer.fingerprint((p,d,fast,image,aux))
            if before != witness['public_packet_sha256']: raise ValueError('exact observer public inputs required')
            decisions = []
            for controller in controllers:
                live = controller.observe(p,d,fast,auxiliary_rgb=image,auxiliary_depth=aux,now_ns=now)
                if observer.fingerprint((p,d,fast,image,aux)) != before: raise ValueError('controller mutated public inputs')
                raw = live['original_visual_evidence']
                if raw['status'] == 'CURRENT_VISUAL_POSE':
                    current_dual_camera_pose(raw,p,image,aux,identity=(0,0,0),now_ns=now)
                if live['terminal'] is None:
                    current_measured_floor_pose(live['evidence'],identity=(0,0,0),now_ns=now)
                decisions.append(json.loads(json.dumps(live,allow_nan=False)))
            old, serialized = decisions
            try:
                if old != row['decision']: raise ValueError('complete original controller decision did not reproduce')
                if old['original_visual_evidence'] != witness['original']:
                    raise ValueError('same original observer history required')
                check = compare(old,serialized,tape[frame]['requested_command'],witness['candidate'],frame=frame)
            except Exception as error:
                append(dict(tick=frame,original=old,decision=serialized,comparison_failure=repr(error))); raise
            append(dict(tick=frame,decision=serialized,comparison=check,
                complete_original_decision_reconstructed=True,
                public_input_sha256=before,public_input_arrays_unchanged=True,
                original_requested_command=tape[frame]['requested_command']))
            frames += 1; forecasts += int(check['original_forecast_compared'])
            if (OUTPUT/observer.NAME).stat().st_size > MAX_OUTPUT_BYTES:
                raise ValueError('bounded controller output exceeded')
            if frame % 50 == 0 or check['stop']: print('CONTACT_FLOW_CONTROLLER_FRAME',frame,flush=True)
            if check['stop']: break
    if frames != BOUNDARY+1 or forecasts != BOUNDARY-3 or not check['stop']:
        raise ValueError('complete fixed prefix and original forecast population required')
    if any(state_digest(m.state_dict()) != MODEL_SHA or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('original model state or gradients changed')
    return dict(frames=frames,exact_candidate_original_decisions=BOUNDARY,original_forecasts_compared=forecasts,
        complete_original_controller_decisions_reconstructed=True,two_fresh_original_model_copies=True,
        boundary_comparison=check,boundary_terminal=serialized['terminal'],boundary_failure=serialized['failure'],
        boundary_selected_action=serialized['selected_action'],boundary_requested_command=serialized['requested_command'],
        model_state_sha256=MODEL_SHA,model_state_unchanged=True,following_recorded_observations_consumed=False,
        actual_original_commands_before_intervention_exact=True,new_command_executed=False,
        original_failed_outcome_preserved=True,native_execution=False,navigation_qualified=False,goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only',action='store_true'); args=parser.parse_args()
    env = dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k,v in env.items()) or cv2.ocl.useOpenCL():
        raise ValueError('assertions and fixed CPU environment required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive full contact controller prefix required')
    sources, observer_ids = prepare(); owners_ended(); hw = resources()
    if args.source_preflight_only:
        print('CONTACT_FLOW_CONTROLLER_SOURCE_PREFLIGHT',len(sources),json.dumps(hw),flush=True); return
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    inputs = verify_inputs(sources,observer_ids); hw = resources(); owners_ended(); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,observer_artifact_sha256=observer_ids,
        input_artifact_sha256=inputs,hardware=hw,protocol=PROTOCOL,model_state_sha256=MODEL_SHA,
        case=list(CASE),boundary_frame=BOUNDARY,opencv_threads=1,blas_threads=1,environment=env,
        previous_observer_owner=OBSERVER_OWNER,boot_id=BOOT,owner_pid=os.getpid(),
        implementation_class='DirectFlowCommitmentContactController',two_fresh_original_model_copies=True,
        original_completed_worker_admitted=True,actual_assigned_model_loader_required=True,
        full_training_ancestry_reexecuted=False,final_parent_completion_authenticated=False,
        native_execution=False,model_training=False,maximum_output_bytes=MAX_OUTPUT_BYTES,automatic_retry=False))
    print('CONTACT_FLOW_CONTROLLER_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True); start=time.perf_counter()
    try:
        report=replay(); write_json(OUTPUT/'report.json',report)
        if verify_inputs(sources,observer_ids) != inputs: raise ValueError('original input admission changed')
        ids={n:digest(OUTPUT/n) for n in ('launch.json',observer.NAME,'report.json')};verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='CONTACT_ANCHORED_DIRECT_FLOW_CONTROLLER_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,wall_s=time.perf_counter()-start,
            native_execution=False,goal_achieved=False))
        print('CONTACT_FLOW_CONTROLLER_COMPLETE',digest(OUTPUT/'result.json'),report,flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_CONTACT_FLOW_CONTROLLER_PREFIX_FAILURE',
            reason=repr(error),automatic_retry=False,original_work_retained=True));raise


if __name__ == '__main__': main()
