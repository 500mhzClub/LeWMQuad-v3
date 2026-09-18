"""Persist and authenticate audited cases for the prospective fixed population.

No scene, model, process or execution entry point. The final launcher must own
policy/input admission, collection, original raw audit and process scheduling.
These functions cannot turn saved receipt checks into a fresh raw audit.
"""
from copy import deepcopy
import json
import numpy as np

from lewm.independent_round_trip_comparison_study_development import CASES, require_case
from lewm.independent_round_trip_multiarm_contract_development import require_collection, treatment
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS
from lewm.independent_round_trip_population_readout_development import (
    WORKER_STATUS, case_readout, require_completed_case, complete_population)
from scripts.independent_round_trip_adapter_multiarm_episode_development import artifacts
from scripts.independent_round_trip_paired_startup_development import reference_case, compare_case_startup
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

SOURCE = 'scripts/independent_round_trip_population_case_evidence_development.py'
TEST = 'lewm/tests/test_independent_round_trip_population_case_evidence_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_population_case_evidence_v1_2026-09-10.md'
SUFFIXES = ('_audit.json', '_startup_comparison.json', '_readout.json', '_worker.log')


def read_json(output, name):
    return json.loads(artifact_path(output, name).read_text())


def collection_names(case, collection):
    require_collection(case, collection)
    names = [case.name+'/'+name for name in artifacts(case, collection)]
    if len(names) != len(set(names)):
        raise ValueError('unique complete collector artifact roster required')
    return names


def bind_collection(output, case, collection):
    """Freeze all persisted collector bytes before invoking the raw auditor."""
    output = validate_root(output)
    names = collection_names(case, collection)
    if read_json(output, case.name+'/result.json') != collection:
        raise ValueError('the original on-disk collection receipt is required')
    return {name: digest(artifact_path(output, name)) for name in names}


def contacts(output, case):
    with np.load(artifact_path(output, case.name+'/physics_trace.npz'), allow_pickle=False) as saved:
        return saved['physics_contact'].copy()


def reference_record(output, case, launch_sha256, reference_worker_sha256):
    first = reference_case(case)
    if first == case:
        if reference_worker_sha256 is not None:
            raise ValueError('first case has no previously completed reference worker')
        return None
    if reference_worker_sha256 is None:
        raise ValueError('exact completed fixed reference worker identity required')
    record, _, _, _ = read_audited_case(output, first, reference_worker_sha256,
        launch_sha256=launch_sha256)
    return record


def persist_audited_case(output, case, collection, report, *, launch_sha256,
        collection_artifact_sha256, worker_log_sha256, reference_worker_sha256=None):
    """Called after the original audit and after closing the worker log.

The pre-audit collection hash map is mandatory. Failed validation leaves raw
evidence in place and never publishes a successful worker terminal.
"""
    output = validate_root(output); arm = require_case(case)
    names = collection_names(case, collection)
    if set(collection_artifact_sha256) != set(names):
        raise ValueError('exact complete pre-audit collection bindings required')
    for suffix in (*SUFFIXES[:3], '_worker_terminal.json'):
        target = output/(case.name+suffix)
        if target.exists() or target.is_symlink():
            raise ValueError('exclusive case evidence; no overwrite, repair or resume')
    initial = dict(collection_artifact_sha256)
    initial.update({'launch.json': launch_sha256, case.name+'_worker.log': worker_log_sha256})
    verify_artifacts(output, initial)
    if read_json(output, case.name+'/result.json') != collection:
        raise ValueError('original persisted collection receipt required')
    first = reference_record(output, case, launch_sha256, reference_worker_sha256)
    reference = collection if first is None else first['collection']
    startup = compare_case_startup(output, case, reference, collection)
    contact = contacts(output, case)
    readout = case_readout(report, collection, contact)
    record = dict(status=WORKER_STATUS, **treatment(case), collection=deepcopy(collection),
        model_state_sha256=arm.model_state_sha256,
        **{key: deepcopy(report[key]) for key in OUTCOME_KEYS},
        startup_comparison=startup, readout=readout, launch_sha256=launch_sha256,
        reference_worker_sha256=reference_worker_sha256, worker_log_sha256=worker_log_sha256)
    require_completed_case(case, record, report, contact, reference)
    bindings = dict(collection_artifact_sha256)
    bindings[case.name+'_worker.log'] = worker_log_sha256
    for suffix, value in zip(SUFFIXES[:3], (report, startup, readout), strict=True):
        name = case.name+suffix
        write_json(output/name, value); bindings[name] = digest(artifact_path(output, name))
    # Recheck the original collection, closed log and launch after all readers.
    verify_artifacts(output, initial)
    verify_artifacts(output, bindings)
    record['artifact_sha256'] = bindings
    write_json(output/(case.name+'_worker_terminal.json'), record)
    return record


def read_audited_case(output, case, worker_sha256, *, launch_sha256):
    """Authenticate saved raw evidence and reconstruct its recorded readout.

This does not reexecute the raw sensor/controller auditor. The final launcher's
source and policy checks remain necessary in addition to these file bindings.
"""
    output = validate_root(output); require_case(case)
    terminal = case.name+'_worker_terminal.json'
    anchors = {'launch.json': launch_sha256, terminal: worker_sha256}
    verify_artifacts(output, anchors)
    record = read_json(output, terminal)
    if record['launch_sha256'] != launch_sha256:
        raise ValueError('worker from the same fixed launch required')
    names = collection_names(case, record['collection']) + [case.name+s for s in SUFFIXES]
    bindings = record['artifact_sha256']
    if set(bindings) != set(names) or bindings[case.name+'_worker.log'] != record['worker_log_sha256']:
        raise ValueError('every original collection, audit, startup, readout and log binding required')
    verify_artifacts(output, bindings)
    if (record['collection'] != read_json(output, case.name+'/result.json')
            or record['startup_comparison'] != read_json(output, case.name+'_startup_comparison.json')
            or record['readout'] != read_json(output, case.name+'_readout.json')):
        raise ValueError('worker and original persisted case receipts must agree')
    first = reference_record(output, case, launch_sha256, record['reference_worker_sha256'])
    reference = record['collection'] if first is None else first['collection']
    report = read_json(output, case.name+'_audit.json'); contact = contacts(output, case)
    require_completed_case(case, record, report, contact, reference)
    verify_artifacts(output, bindings)
    verify_artifacts(output, anchors)
    return record, report, contact, dict(bindings, **anchors)


def complete_saved_population(output, ordered_worker_sha256, *, launch_sha256):
    """Reconstruct the whole population from exact, caller-bound worker identities."""
    if (type(ordered_worker_sha256) not in (list, tuple)
            or len(ordered_worker_sha256) != len(CASES)
            or any(type(row) not in (list, tuple) or len(row) != 2 for row in ordered_worker_sha256)
            or [row[0] for row in ordered_worker_sha256] != [case.name for case in CASES]):
        raise ValueError('all 32 exact ordered worker identities required')
    expected = dict(ordered_worker_sha256)
    records = []; reports = []; populations = []; bindings = {}
    for case, (_, sha) in zip(CASES, ordered_worker_sha256, strict=True):
        record, report, contact, ids = read_audited_case(output, case, sha, launch_sha256=launch_sha256)
        first = reference_case(case)
        reference_sha = None if first == case else expected[first.name]
        if record['reference_worker_sha256'] != reference_sha:
            raise ValueError('same original roster reference worker required for every arm')
        for name, value in ids.items():
            if name in bindings and bindings[name] != value:
                raise ValueError('conflicting original population artifact identities')
            bindings[name] = value
        records.append(record); reports.append(report); populations.append(contact)
    summary = complete_population(records, reports, populations)
    verify_artifacts(output, bindings)
    return dict(summary=summary, artifact_sha256=bindings,
        ordered_worker_sha256=[list(row) for row in ordered_worker_sha256],
        original_raw_audit_reexecuted=False, saved_case_evidence_authenticated=True,
        final_policy_review_performed=False, population_execution_permitted=False)
