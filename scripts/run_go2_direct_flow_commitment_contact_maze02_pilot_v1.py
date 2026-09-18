"""Fresh contact-plus-flow physics after completed raw replay and the original sustained-turn queue."""
import argparse
import contextlib
import multiprocessing
import resource
import time
import json
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import cv2
import numpy as np
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_residual_maze02_study_development import resources_for as cohort_resources
from lewm.all_phase_residual_maze02_readout_development import case_readout
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS, require_raw_audit
from scripts import direct_flow_commitment_contact_native_inputs_development as inputs
from scripts.direct_flow_commitment_contact_native_prefix_development import compare, boundary
from scripts.direct_flow_commitment_contact_maze02_episode_development import collect, artifacts
from scripts.direct_flow_commitment_contact_maze02_audit_development import audit
from scripts.all_phase_planner_model_admission_development import load_assigned
from lewm.all_phase_planner_model_adapter_development import AllPhasePlannerModel
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.supervised_commitment_contact_queue_gate_development import require_native_idle
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_direct_flow_commitment_contact_maze02_pilot_v1_attempt_001'
SOURCE = 'scripts/run_go2_direct_flow_commitment_contact_maze02_pilot_v1.py'
PROTOCOL = 'docs/go2_direct_flow_commitment_contact_maze02_pilot_v1_2026-09-11.md'
PREPARATION = 'docs/go2_direct_flow_commitment_contact_native_inputs_preparation_2026-09-11.json'
PREPARATION_SHA = '4cca1b1448e9e07d2b4cf5e121a75cfc40440bb7f9b9f9197ee25d36d401ce37'
TESTS = ('lewm/tests/test_direct_flow_commitment_contact_native_launcher_development.py',
    'lewm/tests/test_direct_flow_commitment_contact_native_prefix_development.py',
    'lewm/tests/test_direct_flow_commitment_contact_native_source_development.py')
CASE = ('full_supervised_direct_flow_commitment_contact_maze_02', *inputs.replay.native.CASE[1:])
WORKER_STATUS = 'DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_COLLECTED_AND_RAW_AUDITED'


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    preparation = json.loads((ROOT/PREPARATION).read_text())
    if preparation['status'] != 'DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_INPUTS_PREPARED':
        raise ValueError('tested contact-plus-flow input preparation required')
    sources = inputs.prepared_sources((SOURCE, PROTOCOL, PREPARATION, *TESTS))
    if any(sources.get(n) != h for n,h in preparation['source_sha256'].items()):
        raise ValueError('same frozen contact-plus-flow input preparation required')
    return sources


def verify_inputs(launch, *, full=False):
    verify_ordered_launch(launch)
    expected = dict(planned_case=list(CASE), output_root=str(OUTPUT),
        implementation_class='DirectFlowCommitmentContactController',
        scene_specification=specification(2), public_mission=public_mission(2),
        model_state_sha256=inputs.replay.MODEL_SHA, robot_urdf_sha256=digest(URDF),
        navigation_ticks=NAVIGATION_TICKS, native_scene_workers=1, opencv_threads=1, blas_threads=1,
        maximum_tasks_per_process=1, physics_paused_during_compute=True,
        renderer_capture_witnesses_enabled=True, fresh_controller_and_memory=True,
        ordinary_waypoint_commitment_contact_enabled=True, direct_corner_flow_missingness_fallback_enabled=True, planner_interface_adapter_enabled=True,
        native_execution=True, model_training=False, independent_layout_development_execution=False,
        reused_development_layout=True)
    if any(launch[k] != v or type(launch[k]) is not type(v) for k, v in expected.items()):
        raise ValueError('exact prospective contact-plus-flow native definition required')
    admission = launch['input_admission']; boundary(admission['prefix_report'])
    inputs.verify_bound(admission, launch['source_sha256'])
    if full and inputs.admit(admission['raw_prefix_result_sha256'], admission['sustained_wait_result_sha256'],
            launch['source_sha256']) != admission:
        raise ValueError('complete original contact-plus-flow input admission changed')


def assigned_model(launch):
    model, condition, variant = load_assigned(launch['input_admission']['correction_admission'], CASE[4])
    if (type(model) is not AllPhasePlannerModel or model.training
            or (condition, variant) != (CASE[3], CASE[2])
            or state_digest(model.state_dict()) != inputs.replay.MODEL_SHA):
        raise ValueError('original assigned contact-plus-flow model state required')
    return model


def require_worker(record, report, prefix_report):
    if (record['status'] != WORKER_STATUS or 'failure' in record or record['case'] != CASE[0]
            or record['layout_index'] != CASE[1] or record['model_name'] != CASE[4]
            or record['condition'] != CASE[3] or record['variant'] != CASE[2]
            or record['model_state_sha256'] != inputs.replay.MODEL_SHA
            or record['model_state_unchanged'] is not True or report['layout_index'] != CASE[1]
            or report['ordinary_waypoint_commitment_contact_enabled'] is not True
            or report['direct_corner_flow_missingness_fallback_enabled'] is not True
            or record['collection']['status'] != 'DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_TERMINAL_AUDIT_REQUIRED'
            or record['collection']['ordinary_waypoint_commitment_contact_enabled'] is not True
            or record['collection']['direct_corner_flow_missingness_fallback_enabled'] is not True):
        raise ValueError('complete exact contact-plus-flow collection and raw audit required')
    require_raw_audit(record, report, learned=True)
    expected = bool(report['native_evaluation']['native_round_trip_candidate_pass']
        and report['strict_physical_visibility_pass'] and not report['hard_measurement_failed_frames'])
    if record['verified_round_trip'] is not expected:
        raise ValueError('unchanged joint native and visibility success criteria required')
    receipt = record['prefix_comparison']; frames, changed = boundary(prefix_report)
    for key in ('physical_and_public_prefix_exact', 'all_preintervention_requested_commands_exact',
            'complete_candidate_decisions_match_prospective_prefix', 'candidate_intervention_command_completed',
            'observer_and_full_controller_intervention'):
        if receipt[key] is not True: raise ValueError('complete physical contact-plus-flow intervention required: '+key)
    expected_receipt = dict(common_prefix_frames=frames, first_intervention_frame=changed,
        physical_prefix_samples=750+50*changed, original_forecasts_compared=prefix_report['original_forecasts_compared'],
        candidate_intervention_command=prefix_report['boundary_requested_command'],
        intervention_command_changed=prefix_report['boundary_comparison']['requested_command_changed'],
        recovered_observation_frame=changed, boundary_command_samples_present=50,
        following_physical_outcomes_compared=False, navigation_verified=False, unexecuted_outcomes_inferred=False)
    if any(receipt[k] != v or type(receipt[k]) is not type(v) for k,v in expected_receipt.items()):
        raise ValueError('exact prospective physical boundary and forecast count required')



def worker(launch_sha):
    name, index, variant, condition, model_name = CASE
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    record = dict(case=name, layout_index=index, variant=variant, condition=condition, model_name=model_name,
        status='DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_WORKER_FAILED', artifact_sha256={}); started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha}); launch = read_json(OUTPUT, 'launch.json')
            verify_inputs(launch); model = assigned_model(launch); before = state_digest(model.state_dict())
            result = collect(index, launch['source_sha256'][PROTOCOL], output=OUTPUT, model=model,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            if state_digest(model.state_dict()) != before: raise ValueError('collection changed assigned contact-plus-flow model')
            ids = {name+'/'+n: digest(OUTPUT/name/n) for n in artifacts(index, result)}
            record.update(collection=result, artifact_sha256=dict(ids)); verify_artifacts(OUTPUT, ids)
            report = audit(index, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT,
                model=assigned_model(launch), robot_geometry=ArticulatedCollisionGeometry(URDF),
                episode_name=name, condition=condition, variant=variant)
            n = name+'_audit.json'; write_json(OUTPUT/n, report); ids[n] = digest(OUTPUT/n)
            record['artifact_sha256'] = dict(ids)
            receipt = compare(inputs.replay.native.OUTPUT/inputs.replay.native.CASE[0], OUTPUT/name,
                inputs.replay.OUTPUT, launch['input_admission']['prefix_report'])
            n = name+'_prefix_comparison.json'; write_json(OUTPUT/n, receipt); ids[n] = digest(OUTPUT/n)
            record['artifact_sha256'] = dict(ids)
            with np.load(OUTPUT/name/'physics_trace.npz', allow_pickle=False) as saved:
                readout = case_readout(report, result, saved['physics_contact'])
            n = name+'_readout.json'; write_json(OUTPUT/n, readout); ids[n] = digest(OUTPUT/n)
            record.update(status=WORKER_STATUS, artifact_sha256=ids, prefix_comparison=receipt, readout=readout,
                model_state_sha256=before, model_state_unchanged=True, ordinary_waypoint_commitment_contact_enabled=True, direct_corner_flow_missingness_fallback_enabled=True,
                **{k: report[k] for k in OUTCOME_KEYS})
            require_worker(record, report, launch['input_admission']['prefix_report'])
            verify_inputs(launch); verify_artifacts(OUTPUT, ids)
        except Exception as error:
            import traceback
            traceback.print_exc(); record.update(status='DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_WORKER_FAILED', failure=repr(error))
    record.update(wall_s=time.perf_counter()-started,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-prefix-result-sha256'); parser.add_argument('--sustained-wait-result-sha256')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--source-preflight-only', action='store_true'); modes.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive contact-plus-flow native attempt; no retry/resume')
    sources = prepared_sources(); resources = hardware(); cohort_resources(resources, 1)
    if args.source_preflight_only:
        print('DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_SOURCE_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            native_execution=False, output_created=False, complete_input_admission_performed=False)), flush=True); return
    if not args.raw_prefix_result_sha256 or not args.sustained_wait_result_sha256:
        raise ValueError('exact completed raw-prefix result and final original sustained-turn waiter identities required')
    print('DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_INPUT_ADMISSION_STARTED', len(sources), flush=True)
    admission = inputs.admit(args.raw_prefix_result_sha256, args.sustained_wait_result_sha256, sources)
    old = read_json(inputs.replay.native.OUTPUT, 'launch.json')
    keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment')
    launch = {k: old[k] for k in keys}
    launch.update(source_sha256=sources, protocol=PROTOCOL, output_root=str(OUTPUT), input_admission=admission,
        planned_case=list(CASE), implementation_class='DirectFlowCommitmentContactController',
        scene_specification=specification(2), public_mission=public_mission(2), robot_urdf_sha256=digest(URDF),
        model_state_sha256=inputs.replay.MODEL_SHA, navigation_ticks=NAVIGATION_TICKS,
        native_scene_workers=1, opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1,
        physics_paused_during_compute=True, renderer_capture_witnesses_enabled=True, fresh_controller_and_memory=True,
        ordinary_waypoint_commitment_contact_enabled=True, direct_corner_flow_missingness_fallback_enabled=True, planner_interface_adapter_enabled=True,
        native_execution=True, model_training=False, independent_layout_development_execution=False,
        reused_development_layout=True, navigation_qualified=False, real_time_qualified=False,
        hardware_qualified=False, goal_achieved=False)
    verify_inputs(launch); assigned_model(launch)
    resources = hardware(); launch['hardware'] = resources; launch['resource_admission'] = cohort_resources(resources, 1)
    if args.preflight_only:
        print('DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            complete_input_admission_performed=True, native_execution=False, output_created=False)), flush=True); return
    require_native_idle(); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); launch_sha = digest(OUTPUT/'launch.json')
    print('DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_LAUNCHED', launch_sha, flush=True); started = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, **hardware()))+'\n'); monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        print('DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_TERMINAL', record['status'], record.get('verified_round_trip'), record.get('failure'), flush=True)
        if record['status'] != WORKER_STATUS:
            raise ValueError('contact-plus-flow collection/audit/prefix incomplete: '+str(record.get('failure')))
        require_worker(record, read_json(OUTPUT, CASE[0]+'_audit.json'), admission['prefix_report'])
        ids = dict(record['artifact_sha256'])
        for n in ('launch.json', 'resource_monitor.jsonl', CASE[0]+'_worker.log', CASE[0]+'_worker_terminal.json'):
            ids[n] = digest(OUTPUT/n)
        verify_inputs(launch, full=True); verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='DIRECT_FLOW_COMMITMENT_CONTACT_MAZE02_PILOT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, conditions=[record], wall_s=time.perf_counter()-started,
            prospective_prefix_result_sha256=admission['raw_prefix_result_sha256'],
            raw_prefix_result_sha256=args.raw_prefix_result_sha256, sustained_wait_result_sha256=args.sustained_wait_result_sha256,
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, ordinary_waypoint_commitment_contact_enabled=True, direct_corner_flow_missingness_fallback_enabled=True, model_training=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_DIRECT_FLOW_COMMITMENT_CONTACT_NATIVE_FAILURE', reason=repr(error), automatic_retry=False)); raise


if __name__ == '__main__': main()
