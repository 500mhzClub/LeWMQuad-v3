"""Parent-owned spawn lifecycle evidence for a future split collection runtime.

No launcher, scheduler, raw-audit acceptance, or global native-idle claim.
"""
from dataclasses import dataclass
import json
import multiprocessing.context
import os
from pathlib import Path

import psutil
from scripts import independent_round_trip_collection_handoff_development as handoff
from scripts.navigation_artifact_root_development import validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_round_trip_comparison_study_development import require_case

SOURCE = 'scripts/independent_round_trip_collection_process_development.py'
TEST = 'lewm/tests/test_independent_round_trip_collection_process_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_collection_process_v1_2026-09-11.md'
PREPARATION = 'docs/go2_independent_round_trip_collection_handoff_preparation_2026-09-10.json'
PREPARATION_SHA = '228400f4ade1cb6cf3340a0a8ade07c66899f4d418c4b7777e87ff72fae68a99'
REGISTRATION = '_collection_process_registration.json'
CONFIRMATION = '_collection_process_zero_exit.json'
STATUS = 'INDEPENDENT_POPULATION_COLLECTION_PARENT_CONFIRMED_ZERO_EXIT'
_tickets = {}


@dataclass(frozen=True, eq=False)
class Ticket:
    process: object
    output: Path
    case: object
    launch_sha: str
    reference_sha: object
    registration_sha: str


def prepared_sources():
    verify({PREPARATION: PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']
    verify(inherited)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), inherited)
    verify(sources)
    return sources


def identity(process):
    owner = dict(pid=process.pid, created=process.create_time(), command=process.cmdline())
    handoff.require_identity(owner)
    return owner


def require_owned(process):
    if (type(process) is not multiprocessing.context.SpawnProcess
            or process._parent_pid != os.getpid()
            or Path('/proc/sys/kernel/random/boot_id').read_text().strip() != handoff.BOOT):
        raise ValueError('same-boot original parent-owned spawn handle required')


def fresh(output, case):
    for suffix in (CONFIRMATION, handoff.FAILURE_SUFFIX, '_worker_terminal.json'):
        path = output/(case.name+suffix)
        if path.exists() or path.is_symlink():
            raise ValueError('preserve existing completion or failure; no replacement')


def register(process, output, case, launch_sha, reference_sha, verifier):
    """Parent registers a live child; caller must keep it alive until this returns."""
    require_owned(process)
    output = validate_root(output); require_case(case); fresh(output, case)
    registration = output/(case.name+REGISTRATION)
    if (process in _tickets or registration.exists() or registration.is_symlink()
            or (output/(case.name+handoff.SUFFIX)).exists()):
        raise ValueError('fresh live collection registration required; no retry or resume')
    if not process.is_alive() or process.exitcode is not None:
        raise ValueError('register original collector while live')
    child = psutil.Process(process.pid)
    owner = identity(child)
    if child.ppid() != os.getpid():
        raise ValueError('actual direct child of registering parent required')
    parent = identity(psutil.Process())
    launch = handoff.checked_launch(output, launch_sha, verifier)
    if any(n not in launch['source_sha256'] for n in (SOURCE, TEST, PROTOCOL)):
        raise ValueError('collection process source, tests and protocol must be frozen in launch')
    handoff.evidence.reference_record(output, case, launch_sha, reference_sha)
    if not process.is_alive() or not handoff.owner_live(owner):
        raise ValueError('collector must remain live through registration')
    record = dict(status='INDEPENDENT_POPULATION_COLLECTION_PROCESS_REGISTERED',
        case=case.name, launch_sha256=launch_sha, reference_worker_sha256=reference_sha,
        collection_owner=owner, parent_owner=parent, boot_id=handoff.BOOT,
        source_sha256=launch['source_sha256'], start_method='spawn', automatic_retry=False,
        parent_verified_zero_exit=False, raw_audit_completed=False,
        audited_episode_complete=False, native_scene_ownership_released=False,
        audit_execution_permitted=False)
    write_json(registration, record)
    ticket = Ticket(process, output, case, launch_sha, reference_sha, digest(registration))
    _tickets[process] = ticket
    return ticket


def confirm(ticket, handoff_sha, verifier):
    """Confirm a registered handle's zero exit and all original collection bytes."""
    if type(ticket) is not Ticket or _tickets.get(ticket.process) is not ticket:
        raise ValueError('original unconsumed in-memory process ticket required; no resume')
    process = ticket.process
    require_owned(process)
    # exitcode comes from this parent's actual multiprocessing wait handle.
    if process.is_alive() or type(process.exitcode) is not int or process.exitcode != 0:
        raise ValueError('original owned collector must have exited normally with code zero')
    output = validate_root(ticket.output); case = ticket.case; require_case(case)
    fresh(output, case)
    registration_name = case.name+REGISTRATION
    verify_artifacts(output, {registration_name:ticket.registration_sha})
    registration = handoff.evidence.read_json(output, registration_name)
    if registration['parent_owner'] != identity(psutil.Process()):
        raise ValueError('same original registering parent identity required')
    if registration['collection_owner']['pid'] != process.pid:
        raise ValueError('same registered child handle required')
    admitted = handoff.read_for_audit(output, case, handoff_sha, ticket.launch_sha,
        ticket.reference_sha, verifier)
    record = admitted['collection_handoff']
    for key in ('case', 'launch_sha256', 'reference_worker_sha256', 'collection_owner',
            'boot_id', 'source_sha256'):
        if record[key] != registration[key]:
            raise ValueError('collection handoff differs from parent registration: '+key)
    bindings = {registration_name:ticket.registration_sha,
        case.name+handoff.SUFFIX:handoff_sha, 'launch.json':ticket.launch_sha}
    verify_artifacts(output, bindings)
    fresh(output, case)
    result = dict(status=STATUS, case=case.name, boot_id=handoff.BOOT,
        launch_sha256=ticket.launch_sha, reference_worker_sha256=ticket.reference_sha,
        source_sha256=record['source_sha256'], artifact_sha256=bindings,
        collection_owner=record['collection_owner'], parent_owner=registration['parent_owner'],
        original_collector_ended=True, owned_process_exitcode=process.exitcode,
        parent_verified_zero_exit=True, complete_collection_bindings_verified=True,
        raw_audit_completed=False, audited_episode_complete=False,
        native_scene_ownership_released=False, audit_execution_permitted=False,
        automatic_retry=False)
    write_json(output/(case.name+CONFIRMATION), result)
    del _tickets[process]
    return result
