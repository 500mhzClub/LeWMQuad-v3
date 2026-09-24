"""Serial process lifecycle for the prospective fixed independent population.

No command-line entry point or final launch admission. A frozen final launcher
must provide its source-bound verifier of inputs, policy review and queue end.
The collector, auditor, assignments and evidence readers are fixed imports.
"""
import contextlib
from copy import deepcopy
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import cv2
import psutil
import torch

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_round_trip_comparison_study_development import CASES, manifest, require_case, resources_for
from lewm.independent_round_trip_population_readout_development import WORKER_STATUS
from scripts.independent_round_trip_adapter_multiarm_episode_development import collect
from scripts.independent_round_trip_adapter_multiarm_audit_development import audit
from scripts import independent_round_trip_population_case_evidence_development as evidence
from scripts.independent_round_trip_paired_startup_development import reference_case
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.supervised_commitment_contact_queue_gate_development import require_native_idle

SOURCE = 'scripts/independent_round_trip_population_runtime_development.py'
TEST = 'lewm/tests/test_independent_round_trip_population_runtime_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_population_runtime_v1_2026-09-10.md'
FAILED = 'INDEPENDENT_POPULATION_WORKER_FAILED'
COMPLETE = 'INDEPENDENT_ROUND_TRIP_POPULATION_NATIVE_V1_COMPLETE'
EXECUTION_STATUS = 'INDEPENDENT_POPULATION_CASE_COLLECTION_AND_RAW_AUDIT_RETURNED'
FIXED_RUNTIME = dict(native_scene_workers=1, maximum_tasks_per_process=1,
    opencv_threads=1, blas_threads=1, torch_threads=1,
    physics_paused_during_compute=True, fresh_controller_and_memory_per_case=True,
    unplanned_controller_or_model_changes_between_cases=False,
    model_training=False, automatic_retry=False, resume=False)


def require_verifier(verifier, launch):
    if (type(verifier) is not types.FunctionType or verifier.__closure__ is not None
            or '<locals>' in verifier.__qualname__):
        raise ValueError('source-bound top-level final launch verifier required')
    source = inspect.getsourcefile(verifier)
    if source is None:
        raise ValueError('ordinary repository verifier source required')
    path = Path(source).absolute()
    if path.resolve() != path or not path.is_relative_to(ROOT):
        raise ValueError('nonsymlink repository verifier source required')
    name = str(path.relative_to(ROOT))
    if (launch['runtime_verifier'] != dict(source=name, function=verifier.__name__)
            or name not in launch['source_sha256']):
        raise ValueError('exact final verifier must belong to the frozen source closure')
    verify({name: launch['source_sha256'][name]})


def checked_launch(output, launch_sha256, verifier, *, full=False):
    if not __debug__:
        raise ValueError('assertions must remain enabled for the original auditors')
    if any(os.environ.get(name) != value for name, value in {
            'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
            'PYTHONHASHSEED':'0'}.items()):
        raise ValueError('fixed BLAS thread counts and Python hash seed required')
    output = validate_root(output)
    verify_artifacts(output, {'launch.json': launch_sha256})
    launch = evidence.read_json(output, 'launch.json')
    sources = launch['source_sha256']; verify(sources)
    for name in (SOURCE, TEST, PROTOCOL, evidence.SOURCE, evidence.TEST, evidence.PROTOCOL):
        if name not in sources:
            raise ValueError('complete frozen runtime and case-evidence sources required')
    expected = dict(output_root=str(output), ordered_cases=manifest()['ordered_cases'],
        runtime=FIXED_RUNTIME, final_policy_review_completed=True,
        complete_input_admission_performed=True, native_queue_completion_verified=True,
        robot_urdf_sha256=digest(URDF))
    if any(type(launch.get(k)) is not type(v) or
            json.dumps(launch[k], sort_keys=True, allow_nan=False) != json.dumps(v, sort_keys=True, allow_nan=False)
            for k, v in expected.items()):
        raise ValueError('fixed admitted population launch required')
    if launch['protocol'] not in sources:
        raise ValueError('final execution protocol must be source-bound')
    require_verifier(verifier, launch)
    before = deepcopy(launch)
    # Flags above are consistency checks. The frozen final verifier must inspect
    # the actual input, queue and review evidence, and reject incomplete work.
    if verifier(launch, full=full) is not None or launch != before:
        raise ValueError('final launch verifier must raise on failure and leave the launch unchanged')
    verify(sources); verify_artifacts(output, {'launch.json': launch_sha256})
    return launch


def case_worker(output, case, launch_sha256, reference_worker_sha256, verifier):
    output = validate_root(output); require_case(case)
    for suffix in ('_worker.log', '_worker_terminal.json', '_worker_execution.json', '_worker_failure.json'):
        path = output/(case.name+suffix)
        if path.exists() or path.is_symlink():
            raise ValueError('fresh exclusive worker required; never overwrite or resume')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter(); stage = 'launch_admission'
    collection = report = None; collection_ids = {}; failure = None
    identity = dict(worker_pid=os.getpid(), worker_create_time=psutil.Process().create_time())
    log_name = case.name+'_worker.log'
    with (output/log_name).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            launch = checked_launch(output, launch_sha256, verifier)
            stage = 'completed_reference_admission'
            evidence.reference_record(output, case, launch_sha256, reference_worker_sha256)
            correction = launch['input_admission']['factory_correction_admission']
            definition = launch['source_sha256'][launch['protocol']]
            stage = 'native_collection'
            collection = collect(case, definition, output=output,
                geometry=ArticulatedCollisionGeometry(URDF), correction_admission=correction)
            stage = 'pre_audit_collection_binding'
            collection_ids = evidence.bind_collection(output, case, collection)
            stage = 'original_raw_sensor_controller_and_physics_audit'
            report = audit(case, collection, definition, input_root=output,
                robot_geometry=ArticulatedCollisionGeometry(URDF), correction_admission=correction)
            stage = 'post_audit_launch_admission'
            checked_launch(output, launch_sha256, verifier)
        except BaseException as error:
            failure = dict(error=repr(error), traceback=traceback.format_exc())
            print(failure['traceback'], flush=True)
    log_sha = digest(artifact_path(output, log_name))
    try:
        if failure is not None:
            raise RuntimeError('collection or original raw audit did not complete')
        stage = 'audited_case_persistence'
        record = evidence.persist_audited_case(output, case, collection, report,
            launch_sha256=launch_sha256, collection_artifact_sha256=collection_ids,
            worker_log_sha256=log_sha, reference_worker_sha256=reference_worker_sha256)
        stage = 'worker_execution_receipt'
        terminal_sha = digest(artifact_path(output, case.name+'_worker_terminal.json'))
        execution = dict(status=EXECUTION_STATUS, case=case.name, launch_sha256=launch_sha256,
            worker_terminal_sha256=terminal_sha, **identity,
            collection_returned=True, original_raw_audit_returned=True,
            same_process_collection_and_raw_audit=True,
            collection_artifact_sha256=collection_ids, worker_log_sha256=log_sha,
            wall_s=time.perf_counter()-started,
            maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        name = case.name+'_worker_execution.json'; write_json(output/name, execution)
        return dict(case=case.name, status=record['status'], worker_terminal_sha256=terminal_sha,
            worker_execution_sha256=digest(artifact_path(output, name)))
    except BaseException as error:
        if failure is None: failure = dict(error=repr(error), traceback=traceback.format_exc())
        failure.update(status=FAILED, case=case.name, stage=stage, launch_sha256=launch_sha256,
            **identity, collection=collection, raw_audit_report=report,
            known_collection_artifact_sha256=collection_ids, worker_log_sha256=log_sha,
            wall_s=time.perf_counter()-started, automatic_retry=False, evidence_preserved=True)
        name = case.name+'_worker_failure.json'; write_json(output/name, failure)
        return dict(case=case.name, status=FAILED, failure_sha256=digest(artifact_path(output, name)))


def accept_worker(output, case, returned, launch_sha256, reference_worker_sha256):
    output = validate_root(output); require_case(case)
    failure_path = output/(case.name+'_worker_failure.json')
    if failure_path.exists() or failure_path.is_symlink():
        raise ValueError('worker failure preserved; cannot count this case as complete')
    if returned.get('case') != case.name or returned.get('status') != WORKER_STATUS:
        raise ValueError('same assigned successfully audited worker required')
    name = case.name+'_worker_execution.json'
    verify_artifacts(output, {name: returned['worker_execution_sha256']})
    execution = evidence.read_json(output, name)
    record, _, _, ids = evidence.read_audited_case(output, case,
        returned['worker_terminal_sha256'], launch_sha256=launch_sha256)
    expected = dict(status=EXECUTION_STATUS, case=case.name, launch_sha256=launch_sha256,
        worker_terminal_sha256=returned['worker_terminal_sha256'],
        collection_returned=True, original_raw_audit_returned=True,
        same_process_collection_and_raw_audit=True, worker_log_sha256=record['worker_log_sha256'],
        collection_artifact_sha256={n: ids[n] for n in evidence.collection_names(case, record['collection'])})
    if (any(type(execution.get(k)) is not type(v) or execution[k] != v for k, v in expected.items())
            or record['reference_worker_sha256'] != reference_worker_sha256):
        raise ValueError('original collection, audit and reference worker receipts must agree')
    if (type(execution['worker_pid']) is not int or execution['worker_pid'] <= 0
            or execution['worker_pid'] == os.getpid()
            or type(execution['worker_create_time']) is not float
            or not 0 < execution['worker_create_time'] < float('inf')):
        raise ValueError('separate fresh worker process identity required')
    if (type(execution['wall_s']) not in (int, float) or not 0 <= execution['wall_s'] < float('inf')
            or type(execution['maximum_rss_bytes']) is not int or execution['maximum_rss_bytes'] <= 0):
        raise ValueError('finite measured worker duration and peak memory required')
    ids[name] = returned['worker_execution_sha256']
    return record, execution, ids


def run_population(output, launch_sha256, verifier):
    """Called only by a final admitted launcher, on its new launch-only root."""
    output = validate_root(output)
    # Filename-only freshness check at the explicit attempt root. Never descend
    # into an unexpected directory or read an unexpected file, protected or not.
    if {path.name for path in output.iterdir()} != {'launch.json'}:
        raise ValueError('exclusive full original population required; no retry or resume')
    completed = []; ordered = []; bindings = {}; worker_identities = set()
    started = time.perf_counter(); stage = 'full_initial_launch_admission'; active_case = None
    try:
        launch = checked_launch(output, launch_sha256, verifier, full=True)
        with (output/'resource_monitor.jsonl').open('x') as monitor:
            for case in CASES:
                active_case = case.name; stage = 'before_case_admission'
                checked_launch(output, launch_sha256, verifier)
                measured = hardware(); admitted = resources_for(measured, completed); require_native_idle()
                monitor.write(json.dumps(dict(case=case.name, stage='before_case', hardware=measured,
                    resource_admission=admitted))+'\n'); monitor.flush()
                first = reference_case(case)
                reference_sha = None if first == case else dict(ordered)[first.name]
                stage = 'fresh_worker_collection_and_raw_audit'
                with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                        max_tasks_per_child=1) as pool:
                    future = pool.submit(case_worker, output, case, launch_sha256, reference_sha, verifier)
                    while True:
                        done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                        if done:
                            returned = future.result(); break
                        monitor.write(json.dumps(dict(case=case.name, stage='worker_active',
                            elapsed_s=time.perf_counter()-started, hardware=hardware()))+'\n'); monitor.flush()
                stage = 'parent_case_authentication'
                checked_launch(output, launch_sha256, verifier)
                record, execution, ids = accept_worker(output, case, returned, launch_sha256, reference_sha)
                identity = (execution['worker_pid'], execution['worker_create_time'])
                if identity in worker_identities:
                    raise ValueError('controller/model process reused across assigned cases')
                worker_identities.add(identity)
                name = case.name+'_parent_completion.json'
                parent = dict(case=case.name, worker_terminal_sha256=returned['worker_terminal_sha256'],
                    worker_execution_sha256=returned['worker_execution_sha256'],
                    verified_round_trip=record['verified_round_trip'], scientific_success_required=False)
                write_json(output/name, parent); ids[name] = digest(artifact_path(output, name))
                bindings.update(ids); completed.append(case.name)
                ordered.append([case.name, returned['worker_terminal_sha256']])
                print('INDEPENDENT_POPULATION_CASE_COMPLETE', case.name, record['verified_round_trip'], flush=True)
        stage = 'complete_population_authentication'
        population = evidence.complete_saved_population(output, ordered, launch_sha256=launch_sha256)
        for name, sha in population['artifact_sha256'].items():
            if bindings.get(name) != sha:
                raise ValueError('final population differs from original parent case bindings')
        stage = 'full_final_launch_admission'
        checked_launch(output, launch_sha256, verifier, full=True)
        bindings['resource_monitor.jsonl'] = digest(artifact_path(output, 'resource_monitor.jsonl'))
        verify_artifacts(output, bindings)
        result = dict(status=COMPLETE, source_sha256=launch['source_sha256'], launch_sha256=launch_sha256,
            artifact_sha256=bindings, ordered_worker_sha256=ordered, population_readout=population['summary'],
            all_original_raw_audits_returned=True, all_fixed_cases_executed=True,
            completed_episodes=len(completed), independent_layout_units=8,
            fresh_worker_processes=len(worker_identities), native_execution=True, model_training=False,
            automatic_retry=False, wall_s=time.perf_counter()-started,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
        write_json(output/'result.json', result)
        return result
    except BaseException as error:
        write_json(output/'failure.json', dict(status='TERMINAL_INDEPENDENT_POPULATION_NATIVE_FAILURE',
            stage=stage, active_case=active_case, completed_cases=completed,
            completed_worker_sha256=ordered, launch_sha256=launch_sha256,
            reason=repr(error), traceback=traceback.format_exc(),
            automatic_retry=False, evidence_preserved=True))
        raise
