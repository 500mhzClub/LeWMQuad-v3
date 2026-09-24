"""Parent acceptance of a registered separate audit child and its full evidence."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

import psutil
from scripts import independent_round_trip_collection_process_development as lifecycle
from scripts import independent_round_trip_collection_handoff_development as handoff
from scripts import independent_round_trip_separate_audit_development as separate
from scripts import independent_round_trip_population_case_evidence_development as evidence
from scripts.navigation_artifact_root_development import validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_round_trip_comparison_study_development import require_case

SOURCE = 'scripts/independent_round_trip_audit_process_development.py'
TEST = 'lewm/tests/test_independent_round_trip_audit_process_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_audit_process_v1_2026-09-11.md'
PREPARATION = 'docs/go2_independent_round_trip_separate_audit_preparation_2026-09-11.json'
PREPARATION_SHA = 'd99d32d2591657fa2c9dd427784138819b7966212a61c8d745c1aeacc0e974c5'
REGISTRATION = '_audit_process_registration.json'
COMPLETION = '_parent_completion.json'
STATUS = 'INDEPENDENT_POPULATION_SEPARATE_AUDIT_PARENT_ACCEPTED'
_tickets = {}


@dataclass(frozen=True, eq=False)
class Ticket:
    process: object
    output: Path
    case: object
    confirmation_sha: str
    launch_sha: str
    reference_sha: object
    registration_sha: str


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']
    verify(inherited)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), inherited)
    verify(sources)
    return sources


def no_parent_completion(output, case):
    path = output/(case.name+COMPLETION)
    if path.exists() or path.is_symlink():
        raise ValueError('preserve original parent completion; no replacement')


def register(process, output, case, confirmation_sha, launch_sha, reference_sha, verifier):
    """Register the actual live spawn child before allowing its audit to begin."""
    lifecycle.require_owned(process)
    output = validate_root(output); require_case(case)
    separate.fresh(output, case); no_parent_completion(output, case)
    path = output/(case.name+REGISTRATION)
    if process in _tickets or path.exists() or path.is_symlink():
        raise ValueError('fresh audit registration required; no retry or resume')
    if not process.is_alive() or process.exitcode is not None:
        raise ValueError('register the original audit child while live')
    child = psutil.Process(process.pid); owner = lifecycle.identity(child)
    parent = lifecycle.identity(psutil.Process())
    if child.ppid() != os.getpid():
        raise ValueError('actual direct audit child required')
    confirmation, original = separate.admit_collection(output, case, confirmation_sha,
        launch_sha, reference_sha, verifier)
    if confirmation['parent_owner'] != parent or owner['pid'] == confirmation['collection_owner']['pid']:
        raise ValueError('same collection parent and distinct audit child required')
    sources = original['source_sha256']
    if any(n not in sources for n in (SOURCE, TEST, PROTOCOL)):
        raise ValueError('audit process source, tests and protocol must be frozen in launch')
    if not process.is_alive() or not handoff.owner_live(owner):
        raise ValueError('audit child must remain live through registration')
    separate.fresh(output, case)
    registered = dict(status='INDEPENDENT_POPULATION_AUDIT_PROCESS_REGISTERED',
        case=case.name, launch_sha256=launch_sha, reference_worker_sha256=reference_sha,
        collection_confirmation_sha256=confirmation_sha, boot_id=handoff.BOOT,
        audit_owner=owner, parent_owner=parent, collection_owner=confirmation['collection_owner'],
        source_sha256=sources, start_method='spawn', automatic_retry=False,
        parent_verified_audit_zero_exit=False, population_case_accepted=False)
    write_json(path, registered)
    ticket = Ticket(process, output, case, confirmation_sha, launch_sha, reference_sha, digest(path))
    _tickets[process] = ticket
    return ticket


def accept(ticket, returned, verifier):
    """Accept only the owned zero-exit child with authenticated original results."""
    if type(ticket) is not Ticket or _tickets.get(ticket.process) is not ticket:
        raise ValueError('original unconsumed audit process ticket required; no resume')
    process = ticket.process; lifecycle.require_owned(process)
    if process.is_alive() or type(process.exitcode) is not int or process.exitcode != 0:
        raise ValueError('original owned audit child must have exited normally with code zero')
    output = validate_root(ticket.output); case = ticket.case; require_case(case)
    no_parent_completion(output, case)
    name = case.name+REGISTRATION
    verify_artifacts(output, {name:ticket.registration_sha})
    registered = evidence.read_json(output, name)
    if (registered['parent_owner'] != lifecycle.identity(psutil.Process())
            or registered['audit_owner']['pid'] != process.pid):
        raise ValueError('same original parent and registered child handle required')
    admitted = separate.read_completed_audit(output, case, returned, ticket.confirmation_sha,
        ticket.launch_sha, ticket.reference_sha, verifier)
    execution = admitted['execution']
    for key in ('case', 'boot_id', 'launch_sha256', 'reference_worker_sha256',
            'collection_confirmation_sha256', 'audit_owner', 'parent_owner',
            'collection_owner', 'source_sha256'):
        if registered[key] != execution[key]:
            raise ValueError('audited result differs from original child registration: '+key)
    ids = dict(admitted['artifact_sha256']); ids[name] = ticket.registration_sha
    verify_artifacts(output, ids); verify(execution['source_sha256'])
    result = dict(status=STATUS, case=case.name, launch_sha256=ticket.launch_sha,
        reference_worker_sha256=ticket.reference_sha, boot_id=handoff.BOOT,
        collection_confirmation_sha256=ticket.confirmation_sha,
        worker_terminal_sha256=returned['worker_terminal_sha256'],
        separate_audit_execution_sha256=returned['separate_audit_execution_sha256'],
        source_sha256=execution['source_sha256'], artifact_sha256=ids,
        audit_owner=execution['audit_owner'], parent_owner=execution['parent_owner'],
        collection_owner=execution['collection_owner'], owned_audit_exitcode=process.exitcode,
        parent_verified_audit_zero_exit=True, population_case_accepted=True,
        verified_round_trip=admitted['record']['verified_round_trip'],
        scientific_success_required=False, same_process_collection_and_raw_audit=False,
        automatic_retry=False, native_scene_ownership_released=False,
        real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
    no_parent_completion(output, case)
    write_json(output/(case.name+COMPLETION), result)
    del _tickets[process]
    return result
