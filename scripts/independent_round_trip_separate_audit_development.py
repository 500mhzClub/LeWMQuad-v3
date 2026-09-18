"""Fresh-process raw audit of a parent-confirmed completed collection.

No collector or scheduler is invoked. Saved audit evidence is not parent
confirmation of the audit child's exit, nor global single-scene admission.
"""
import contextlib
import json
import multiprocessing
import os
import resource
import time
import traceback

import cv2
import psutil
import torch
from scripts import independent_round_trip_collection_process_development as lifecycle
from scripts import independent_round_trip_collection_handoff_development as handoff
from scripts import independent_round_trip_population_case_evidence_development as evidence
from scripts.independent_round_trip_adapter_multiarm_audit_development import audit
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_round_trip_comparison_study_development import require_case
from lewm.independent_round_trip_population_readout_development import WORKER_STATUS

SOURCE = 'scripts/independent_round_trip_separate_audit_development.py'
TEST = 'lewm/tests/test_independent_round_trip_separate_audit_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_separate_audit_v1_2026-09-11.md'
PREPARATION = 'docs/go2_independent_round_trip_collection_process_preparation_2026-09-11.json'
PREPARATION_SHA = '317ce035af1e8f85e5930f005057638e46bd28bb002f6afb0d2c177b688cae11'
EXECUTION = '_separate_audit_execution.json'
FAILED = 'INDEPENDENT_POPULATION_SEPARATE_RAW_AUDIT_FAILED'
STATUS = 'INDEPENDENT_POPULATION_SEPARATE_RAW_AUDIT_RETURNED'


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']
    verify(inherited)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), inherited)
    verify(sources)
    return sources


def same_fields(actual, expected, message):
    if any(type(actual.get(k)) is not type(v) or actual[k] != v for k, v in expected.items()):
        raise ValueError(message)


def admit_collection(output, case, confirmation_sha, launch_sha, reference_sha, verifier):
    """Authenticate the parent's zero-exit record and the full original handoff."""
    output = validate_root(output); require_case(case)
    name = case.name+lifecycle.CONFIRMATION
    verify_artifacts(output, {name:confirmation_sha, 'launch.json':launch_sha})
    confirmed = evidence.read_json(output, name)
    same_fields(confirmed, dict(status=lifecycle.STATUS, case=case.name,
        boot_id=handoff.BOOT, launch_sha256=launch_sha, reference_worker_sha256=reference_sha,
        original_collector_ended=True, owned_process_exitcode=0, parent_verified_zero_exit=True,
        complete_collection_bindings_verified=True, raw_audit_completed=False,
        audited_episode_complete=False, native_scene_ownership_released=False,
        audit_execution_permitted=False, automatic_retry=False), 'exact parent zero-exit confirmation required')
    handoff.require_identity(confirmed['parent_owner'])
    handoff.require_identity(confirmed['collection_owner'])
    registration_name = case.name+lifecycle.REGISTRATION
    handoff_name = case.name+handoff.SUFFIX
    ids = confirmed['artifact_sha256']
    if set(ids) != {registration_name, handoff_name, 'launch.json'} or ids['launch.json'] != launch_sha:
        raise ValueError('exact original registration, handoff and launch bindings required')
    verify_artifacts(output, ids)
    registered = evidence.read_json(output, registration_name)
    same_fields(registered, dict(status='INDEPENDENT_POPULATION_COLLECTION_PROCESS_REGISTERED',
        start_method='spawn', automatic_retry=False, parent_verified_zero_exit=False,
        raw_audit_completed=False, audited_episode_complete=False,
        native_scene_ownership_released=False, audit_execution_permitted=False),
        'original pending collection registration required')
    for key in ('case', 'boot_id', 'launch_sha256', 'reference_worker_sha256',
            'parent_owner', 'collection_owner', 'source_sha256'):
        if registered[key] != confirmed[key]:
            raise ValueError('parent registration and confirmation disagree: '+key)
    admitted = handoff.read_for_audit(output, case, ids[handoff_name], launch_sha, reference_sha, verifier)
    record = admitted['collection_handoff']
    for key in ('case', 'boot_id', 'launch_sha256', 'reference_worker_sha256',
            'collection_owner', 'source_sha256'):
        if record[key] != confirmed[key]:
            raise ValueError('original collection and confirmation disagree: '+key)
    if any(n not in record['source_sha256'] for n in
            (SOURCE, TEST, PROTOCOL, lifecycle.SOURCE, lifecycle.TEST, lifecycle.PROTOCOL)):
        raise ValueError('separate audit and lifecycle sources must be frozen in launch')
    verify_artifacts(output, ids | {name:confirmation_sha})
    return confirmed, record


def require_audit_child(confirmation):
    """The registering parent must launch a fresh spawn child for this audit."""
    parent = multiprocessing.parent_process()
    current = psutil.Process()
    owner = confirmation['parent_owner']
    if (parent is None or multiprocessing.get_start_method() != 'spawn'
            or parent.pid != owner['pid'] or current.ppid() != owner['pid']
            or current.pid in (owner['pid'], confirmation['collection_owner']['pid'])
            or not handoff.owner_live(owner)):
        raise ValueError('fresh audit spawn child of the original live parent required')
    return lifecycle.identity(current)


def fresh(output, case):
    for suffix in ('_worker.log', '_worker_terminal.json', '_worker_failure.json',
            '_worker_execution.json', EXECUTION, '_audit.json', '_readout.json', '_startup_comparison.json'):
        path = output/(case.name+suffix)
        if path.exists() or path.is_symlink():
            raise ValueError('fresh exclusive audit evidence required; no retry or resume')


def audit_worker(output, case, confirmation_sha, launch_sha, reference_sha, verifier):
    """Run the fixed full auditor; caller owns process scheduling and exit checks."""
    output = validate_root(output); require_case(case); fresh(output, case)
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter(); stage = 'parent_confirmation_admission'
    collection = report = None; collection_ids = {}; failure = None; owner = None
    confirmation = None
    log_name = case.name+'_worker.log'
    with (output/log_name).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            confirmation, handoff_record = admit_collection(output, case, confirmation_sha,
                launch_sha, reference_sha, verifier)
            stage = 'fresh_audit_process_admission'
            owner = require_audit_child(confirmation)
            launch = handoff.checked_launch(output, launch_sha, verifier)
            collection = handoff_record['collection']
            collection_ids = {n:handoff_record['artifact_sha256'][n]
                for n in evidence.collection_names(case, collection)}
            stage = 'original_raw_sensor_controller_and_physics_audit'
            report = audit(case, collection, launch['source_sha256'][launch['protocol']],
                input_root=output, robot_geometry=ArticulatedCollisionGeometry(URDF),
                correction_admission=launch['input_admission']['factory_correction_admission'])
            stage = 'post_audit_collection_admission'
            after, _ = admit_collection(output, case, confirmation_sha, launch_sha, reference_sha, verifier)
            if after != confirmation or require_audit_child(after) != owner:
                raise ValueError('original collection or audit process changed')
        except BaseException as error:
            failure = dict(error=repr(error), traceback=traceback.format_exc())
            print(failure['traceback'], flush=True)
    log_sha = digest(artifact_path(output, log_name))
    try:
        if failure is not None:
            raise RuntimeError('separate original raw audit did not complete')
        stage = 'audited_case_persistence'
        record = evidence.persist_audited_case(output, case, collection, report,
            launch_sha256=launch_sha, collection_artifact_sha256=collection_ids,
            worker_log_sha256=log_sha, reference_worker_sha256=reference_sha)
        stage = 'separate_audit_execution_receipt'
        terminal_sha = digest(artifact_path(output, case.name+'_worker_terminal.json'))
        execution = dict(status=STATUS, case=case.name, launch_sha256=launch_sha,
            reference_worker_sha256=reference_sha, collection_confirmation_sha256=confirmation_sha,
            worker_terminal_sha256=terminal_sha, source_sha256=handoff_record['source_sha256'],
            boot_id=handoff.BOOT, audit_owner=owner, collection_owner=confirmation['collection_owner'],
            parent_owner=confirmation['parent_owner'], original_raw_audit_returned=True,
            same_process_collection_and_raw_audit=False, native_collection_called=False,
            collection_artifact_sha256=collection_ids, worker_log_sha256=log_sha,
            wall_s=time.perf_counter()-started,
            maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            parent_verified_audit_zero_exit=False, parallel_speedup_measured=False,
            cpu_only_execution_qualified=False, automatic_retry=False)
        verify_artifacts(output, {case.name+lifecycle.CONFIRMATION:confirmation_sha})
        name = case.name+EXECUTION; write_json(output/name, execution)
        return dict(status=record['status'], case=case.name, worker_terminal_sha256=terminal_sha,
            separate_audit_execution_sha256=digest(artifact_path(output, name)))
    except BaseException as error:
        if failure is None: failure = dict(error=repr(error), traceback=traceback.format_exc())
        failure.update(status=FAILED, case=case.name, stage=stage, launch_sha256=launch_sha,
            collection_confirmation_sha256=confirmation_sha, audit_owner=owner,
            collection=collection, raw_audit_report=report,
            known_collection_artifact_sha256=collection_ids, worker_log_sha256=log_sha,
            wall_s=time.perf_counter()-started, automatic_retry=False, evidence_preserved=True)
        name = case.name+'_worker_failure.json'; write_json(output/name, failure)
        return dict(case=case.name, status=FAILED, failure_sha256=digest(artifact_path(output, name)))


def read_completed_audit(output, case, returned, confirmation_sha, launch_sha, reference_sha, verifier):
    """Authenticate ended audit evidence; caller must still verify owned zero exit."""
    output = validate_root(output); require_case(case)
    failure = output/(case.name+'_worker_failure.json')
    if failure.exists() or failure.is_symlink():
        raise ValueError('audit failure preserved; case cannot be accepted')
    same_fields(returned, dict(case=case.name, status=WORKER_STATUS), 'completed assigned raw audit required')
    name = case.name+EXECUTION
    verify_artifacts(output, {name:returned['separate_audit_execution_sha256']})
    execution = evidence.read_json(output, name)
    handoff.require_identity(execution['audit_owner'])
    if handoff.owner_live(execution['audit_owner']):
        raise ValueError('original audit worker remains live')
    confirmation, original = admit_collection(output, case, confirmation_sha, launch_sha, reference_sha, verifier)
    expected = dict(status=STATUS, case=case.name, launch_sha256=launch_sha,
        reference_worker_sha256=reference_sha, collection_confirmation_sha256=confirmation_sha,
        worker_terminal_sha256=returned['worker_terminal_sha256'], source_sha256=original['source_sha256'],
        boot_id=handoff.BOOT, collection_owner=confirmation['collection_owner'],
        parent_owner=confirmation['parent_owner'], original_raw_audit_returned=True,
        same_process_collection_and_raw_audit=False, native_collection_called=False,
        parent_verified_audit_zero_exit=False, parallel_speedup_measured=False,
        cpu_only_execution_qualified=False, automatic_retry=False,
        collection_artifact_sha256={n:original['artifact_sha256'][n]
            for n in evidence.collection_names(case, original['collection'])})
    same_fields(execution, expected, 'original separate audit execution evidence required')
    if execution['audit_owner']['pid'] in (confirmation['parent_owner']['pid'], confirmation['collection_owner']['pid']):
        raise ValueError('distinct original audit worker required')
    if (type(execution['wall_s']) not in (int, float) or not 0 <= execution['wall_s'] < float('inf')
            or type(execution['maximum_rss_bytes']) is not int or execution['maximum_rss_bytes'] <= 0):
        raise ValueError('finite audit time and positive measured memory required')
    record, _, _, ids = evidence.read_audited_case(output, case,
        returned['worker_terminal_sha256'], launch_sha256=launch_sha)
    if (record['reference_worker_sha256'] != reference_sha
            or record['worker_log_sha256'] != execution['worker_log_sha256']):
        raise ValueError('same original reference and closed audit log required')
    ids.update(original['artifact_sha256'])
    ids.update(confirmation['artifact_sha256'])
    ids.update({case.name+lifecycle.CONFIRMATION:confirmation_sha,
        name:returned['separate_audit_execution_sha256']})
    verify_artifacts(output, ids)
    if handoff.owner_live(execution['audit_owner']):
        raise ValueError('original audit ownership changed during evidence admission')
    return dict(record=record, execution=execution, artifact_sha256=ids,
        saved_audit_evidence_authenticated=True, original_raw_audit_reexecuted=False,
        parent_verified_audit_zero_exit=False, population_case_accepted=False)
