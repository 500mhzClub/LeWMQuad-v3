"""Bind finished collection bytes for a future separate-process raw audit.

This is an evidence boundary, not a scheduler or permission to execute an audit.
The future coordinator must additionally verify its owned child's zero exit.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path

import psutil
from scripts import independent_round_trip_population_runtime_development as runtime
from scripts import independent_round_trip_population_case_evidence_development as evidence
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_round_trip_comparison_study_development import require_case

SOURCE = 'scripts/independent_round_trip_collection_handoff_development.py'
TEST = 'lewm/tests/test_independent_round_trip_collection_handoff_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_collection_handoff_v1_2026-09-10.md'
PREPARATION = 'docs/go2_independent_round_trip_population_runtime_verification_2026-09-10.json'
PREPARATION_SHA = 'f5b25eb2c15f86e2e0948a62b39f4e859002f3c3d896ab773b14837b225ce8e7'
STATUS = 'INDEPENDENT_POPULATION_COLLECTION_COMPLETE_RAW_AUDIT_PENDING'
SUFFIX = '_collection_handoff.json'
LOG_SUFFIX = '_collection_worker.log'
FAILURE_SUFFIX = '_collection_worker_failure.json'


def prepared_sources():
    verify({PREPARATION: PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']
    verify(inherited)
    sources = discover_sources((SOURCE,TEST,PROTOCOL,PREPARATION),inherited)
    verify(sources)
    return sources


def checked_launch(output, launch_sha, verifier):
    launch = runtime.checked_launch(output,launch_sha,verifier)
    if any(n not in launch['source_sha256'] for n in (SOURCE,TEST,PROTOCOL)):
        raise ValueError('collection handoff source, tests and protocol must be frozen in launch')
    return launch


def require_identity(owner):
    if (type(owner) is not dict or set(owner) != {'pid','created','command'}
            or type(owner['pid']) is not int or owner['pid'] <= 0
            or type(owner['created']) is not float or not math.isfinite(owner['created']) or owner['created'] <= 0
            or type(owner['command']) is not list or not owner['command']
            or any(type(v) is not str or not v for v in owner['command'])):
        raise ValueError('complete exact collection process identity required')


def require_pending(record, case, launch_sha, reference_sha):
    if (record['status'] != STATUS or record['case'] != case.name
            or record['launch_sha256'] != launch_sha
            or record['reference_worker_sha256'] != reference_sha
            or record['boot_id'] != BOOT or record['collection_returned'] is not True
            or record['raw_audit_completed'] is not False or record['audited_episode_complete'] is not False
            or record['native_scene_ownership_released'] is not False
            or record['parent_verified_zero_exit'] is not False
            or record['audit_execution_permitted'] is not False):
        raise ValueError('same pending-audit collection without execution or audit claims required')
    require_identity(record['collection_owner'])


def publish(output, case, collection, launch_sha, reference_sha, verifier):
    """Call in the collector after closing its log and before process exit."""
    output=validate_root(output);require_case(case)
    for suffix in (SUFFIX,FAILURE_SUFFIX,'_worker_terminal.json'):
        p=output/(case.name+suffix)
        if p.exists() or p.is_symlink():
            raise ValueError('fresh collection handoff; preserve failure and existing evidence')
    launch=checked_launch(output,launch_sha,verifier)
    evidence.reference_record(output,case,launch_sha,reference_sha)
    process=psutil.Process()
    if process.pid != os.getpid() or Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('same-boot actual collector process required')
    owner=dict(pid=process.pid,created=process.create_time(),command=process.cmdline())
    require_identity(owner)
    log_name=case.name+LOG_SUFFIX;log_path=artifact_path(output,log_name)
    if any(Path(f.path)==log_path for f in process.open_files()):
        raise ValueError('collector log must be closed before handoff binding')
    ids=evidence.bind_collection(output,case,collection)
    ids.update({'launch.json':launch_sha,log_name:digest(log_path)})
    verify_artifacts(output,ids)
    record=dict(status=STATUS,case=case.name,collection=deepcopy(collection),
        source_sha256=launch['source_sha256'],launch_sha256=launch_sha,
        reference_worker_sha256=reference_sha,artifact_sha256=ids,
        collection_log_sha256=ids[log_name],collection_owner=owner,boot_id=BOOT,
        collection_returned=True,raw_audit_completed=False,audited_episode_complete=False,
        native_scene_ownership_released=False,parent_verified_zero_exit=False,
        audit_execution_permitted=False)
    require_pending(record,case,launch_sha,reference_sha)
    verify_artifacts(output,ids);verify(launch['source_sha256'])
    write_json(output/(case.name+SUFFIX),record)
    return record


def read_for_audit(output, case, handoff_sha, launch_sha, reference_sha, verifier):
    """Require ended original collector and unchanged bytes; never launch work."""
    output=validate_root(output);require_case(case)
    failure=output/(case.name+FAILURE_SUFFIX)
    if failure.exists() or failure.is_symlink():
        raise ValueError('original collection failure retained; no audit handoff')
    name=case.name+SUFFIX
    verify_artifacts(output,{'launch.json':launch_sha,name:handoff_sha})
    record=evidence.read_json(output,name);require_pending(record,case,launch_sha,reference_sha)
    # Check process identity before expensive source/reference/data admission.
    if owner_live(record['collection_owner']):
        raise ValueError('original collector remains live; scene ownership not released')
    launch=checked_launch(output,launch_sha,verifier)
    if record['source_sha256'] != launch['source_sha256']:
        raise ValueError('same frozen collection and audit launch sources required')
    expected=set(evidence.collection_names(case,record['collection'])) | {'launch.json',case.name+LOG_SUFFIX}
    ids=record['artifact_sha256']
    if (set(ids) != expected or ids['launch.json'] != launch_sha
            or ids[case.name+LOG_SUFFIX] != record['collection_log_sha256']):
        raise ValueError('all original collection artifacts and closed log required')
    verify_artifacts(output,ids)
    if evidence.read_json(output,case.name+'/result.json') != record['collection']:
        raise ValueError('same completed on-disk collection receipt required')
    evidence.reference_record(output,case,launch_sha,reference_sha)
    verify_artifacts(output,ids | {name:handoff_sha});verify(launch['source_sha256'])
    if owner_live(record['collection_owner']):
        raise ValueError('original collection ownership changed during admission')
    return dict(collection_handoff=record,collection_handoff_sha256=handoff_sha,
        original_collector_ended=True,complete_collection_bindings_verified=True,
        raw_audit_completed=False,audited_episode_complete=False,
        parent_verified_zero_exit=False,audit_execution_permitted=False)
