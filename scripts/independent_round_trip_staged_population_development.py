"""Bounded spawn driver for the fixed study; no final launcher or CLI.

Both final population admission and separate-audit overlap qualification must
be supplied by frozen source-bound verifiers. Neither is implemented here.
"""
import contextlib
from copy import deepcopy
from dataclasses import dataclass
import json
import multiprocessing
import os
import time
import traceback

import cv2
import psutil
import torch
from scripts import independent_round_trip_population_runtime_development as runtime
from scripts import independent_round_trip_collection_handoff_development as handoff
from scripts import independent_round_trip_collection_process_development as collection_process
from scripts import independent_round_trip_separate_audit_development as separate
from scripts import independent_round_trip_audit_process_development as audit_process
from scripts import independent_round_trip_population_schedule_development as scheduling
from scripts import independent_round_trip_population_case_evidence_development as evidence
from scripts.independent_round_trip_adapter_multiarm_episode_development import collect
from scripts.run_go2_prepared_native_queue_v1 import competitors
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_round_trip_comparison_study_development import CASES, require_case

SOURCE = 'scripts/independent_round_trip_staged_population_development.py'
TEST = 'lewm/tests/test_independent_round_trip_staged_population_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_staged_population_v1_2026-09-11.md'
PREPARATION = 'docs/go2_independent_round_trip_population_schedule_preparation_2026-09-11.json'
PREPARATION_SHA = '23b91e4fec1d1d3ec9b7fbd77ad9cc916d4f95018a09c17355804f20d687f465'
COMPLETE = 'INDEPENDENT_ROUND_TRIP_STAGED_POPULATION_NATIVE_V1_COMPLETE'
FIXED = dict(maximum_active_collectors=1, maximum_active_auditors=1,
    maximum_unaccepted_cases=2, spawn_start_barriers=True,
    first_arm_parent_acceptance_required=True, automatic_retry=False, resume=False,
    overlap_requires_cpu_only_audit_qualification=True, available_memory_minimum_bytes=64*1024**3,
    failure_stops_dispatch=True, existing_workers_drain_after_failure=True)


def prepared_sources():
    verify({PREPARATION:PREPARATION_SHA})
    inherited = json.loads((ROOT/PREPARATION).read_text())['source_sha256']
    verify(inherited)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PREPARATION), inherited)
    verify(sources)
    return sources


def checked_launch(output, launch_sha, verifier, overlap_verifier, *, full=False):
    launch = runtime.checked_launch(output, launch_sha, verifier, full=full)
    if (json.dumps(launch.get('staged_runtime'), sort_keys=True) != json.dumps(FIXED, sort_keys=True)
            or any(n not in launch['source_sha256'] for n in (SOURCE, TEST, PROTOCOL,
                scheduling.SOURCE, scheduling.TEST, scheduling.PROTOCOL,
                audit_process.SOURCE, audit_process.TEST, audit_process.PROTOCOL))):
        raise ValueError('fixed staged runtime and complete frozen driver sources required')
    # Reuse the original function/source binding check for the second verifier.
    bound = dict(launch, runtime_verifier=launch['overlap_verifier'])
    runtime.require_verifier(overlap_verifier, bound)
    before = deepcopy(launch)
    if overlap_verifier(launch, full=full) is not None or launch != before:
        raise ValueError('overlap verifier must authenticate CPU-only audit evidence without mutating launch')
    verify(launch['source_sha256']); verify_artifacts(output, {'launch.json':launch_sha})
    return launch


def require_native_slot(audit_ticket=None):
    """Exclude only this parent's exact registered audit; never exempt spawn generally."""
    allowed = None
    if audit_ticket is not None:
        if (type(audit_ticket) is not audit_process.Ticket
                or audit_process._tickets.get(audit_ticket.process) is not audit_ticket):
            raise ValueError('original active audit ticket required for native slot admission')
        collection_process.require_owned(audit_ticket.process)
        name = audit_ticket.case.name+audit_process.REGISTRATION
        verify_artifacts(audit_ticket.output, {name:audit_ticket.registration_sha})
        registered = evidence.read_json(audit_ticket.output, name)
        if (registered['parent_owner'] != collection_process.identity(psutil.Process())
                or registered['audit_owner']['pid'] != audit_ticket.process.pid):
            raise ValueError('same original owned audit process required')
        if handoff.owner_live(registered['audit_owner']):
            allowed = registered['audit_owner']
    for other in competitors():
        if (allowed is None or other != dict(pid=allowed['pid'], started=allowed['created'], command=allowed['command'])):
            raise ValueError('another native runner or unregistered worker remains live')


def await_start(release, cancel, original_parent):
    while not release.wait(1):
        if not handoff.owner_live(original_parent):
            raise ValueError('original parent ended before worker registration')
    if cancel.is_set() or not handoff.owner_live(original_parent):
        raise ValueError('worker start cancelled or original parent ended')


def require_registration(role, output, case, launch_sha, reference_sha, confirmation_sha, original_parent):
    suffix = collection_process.REGISTRATION if role == 'collection' else audit_process.REGISTRATION
    record = evidence.read_json(output, case.name+suffix)
    key = 'collection_owner' if role == 'collection' else 'audit_owner'
    expected_status = ('INDEPENDENT_POPULATION_COLLECTION_PROCESS_REGISTERED' if role == 'collection'
        else 'INDEPENDENT_POPULATION_AUDIT_PROCESS_REGISTERED')
    if (record['status'] != expected_status or record['case'] != case.name
            or record['launch_sha256'] != launch_sha or record['reference_worker_sha256'] != reference_sha
            or record['parent_owner'] != original_parent or record['boot_id'] != handoff.BOOT
            or record[key] != collection_process.identity(psutil.Process())
            or (role == 'audit' and record['collection_confirmation_sha256'] != confirmation_sha)):
        raise ValueError('same actual registered worker and launch required before release')
    launch = evidence.read_json(output, 'launch.json')
    verify_artifacts(output, {'launch.json':launch_sha})
    if record['source_sha256'] != launch['source_sha256']:
        raise ValueError('same frozen worker registration sources required')
    verify(record['source_sha256'])


def collection_worker(output, case, launch_sha, reference_sha, verifier):
    """Collect once, close its log, and publish the existing complete handoff."""
    output = validate_root(output); require_case(case)
    name = case.name+handoff.LOG_SUFFIX
    for suffix in (handoff.LOG_SUFFIX, handoff.SUFFIX, handoff.FAILURE_SUFFIX):
        path = output/(case.name+suffix)
        if path.exists() or path.is_symlink(): raise ValueError('fresh collector required; no retry')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    collection = None; failure = None; stage = 'collection_launch_admission'
    with (output/name).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            launch = handoff.checked_launch(output, launch_sha, verifier)
            evidence.reference_record(output, case, launch_sha, reference_sha)
            stage = 'native_collection'
            collection = collect(case, launch['source_sha256'][launch['protocol']], output=output,
                geometry=ArticulatedCollisionGeometry(URDF),
                correction_admission=launch['input_admission']['factory_correction_admission'])
        except BaseException as error:
            failure = dict(error=repr(error), traceback=traceback.format_exc())
            print(failure['traceback'], flush=True)
    try:
        if failure is not None: raise RuntimeError('original collection failed')
        stage = 'closed_collection_handoff'
        handoff.publish(output, case, collection, launch_sha, reference_sha, verifier)
    except BaseException as error:
        if failure is None: failure = dict(error=repr(error), traceback=traceback.format_exc())
        failure.update(status='INDEPENDENT_POPULATION_COLLECTION_WORKER_FAILED', case=case.name,
            stage=stage, collection=collection, launch_sha256=launch_sha,
            collection_owner=collection_process.identity(psutil.Process()),
            collection_log_sha256=digest(artifact_path(output, name)), automatic_retry=False, evidence_preserved=True)
        write_json(output/(case.name+handoff.FAILURE_SUFFIX), failure)
        raise RuntimeError('original collection did not produce a valid handoff') from error


def worker_entry(role, output, case, launch_sha, reference_sha, confirmation_sha, verifier,
        original_parent, ready, release, cancel):
    output = validate_root(output); require_case(case)
    if role not in ('collection', 'audit'): raise ValueError('fixed worker role required')
    try:
        ready.send(collection_process.identity(psutil.Process())); ready.close()
        await_start(release, cancel, original_parent)
        require_registration(role, output, case, launch_sha, reference_sha, confirmation_sha, original_parent)
        if role == 'collection':
            collection_worker(output, case, launch_sha, reference_sha, verifier)
        else:
            returned = separate.audit_worker(output, case, confirmation_sha, launch_sha, reference_sha, verifier)
            if returned['status'] != separate.WORKER_STATUS:
                raise RuntimeError('original separate audit worker failed')
    except BaseException as error:
        path = output/(case.name+'_'+role+'_entry_failure.json')
        if not path.exists() and not path.is_symlink():
            write_json(path, dict(status='INDEPENDENT_POPULATION_STAGE_ENTRY_FAILED', role=role,
                case=case.name, reason=repr(error), traceback=traceback.format_exc(),
                owner=collection_process.identity(psutil.Process()), automatic_retry=False, evidence_preserved=True))
        raise


@dataclass
class Active:
    process: object
    ready: object
    release: object
    cancel: object
    case: object
    reference_sha: object
    confirmation_sha: object
    ticket: object = None


class Driver:
    def __init__(self, output, launch_sha, verifier, overlap_verifier, monitor):
        self.output = output; self.launch_sha = launch_sha; self.verifier = verifier
        self.overlap_verifier = overlap_verifier; self.monitor = monitor
        self.schedule = scheduling.PopulationSchedule(); self.active = {}; self.bindings = {}
        self.last_monitor = 0.; self.identities = set()

    def check(self, full=False):
        return checked_launch(self.output, self.launch_sha, self.verifier, self.overlap_verifier, full=full)

    def sample(self, *, force=False):
        now = time.monotonic()
        if force or now-self.last_monitor >= 15:
            self.monitor.write(json.dumps(dict(monotonic_s=now, schedule=self.schedule.snapshot(),
                active_owners={k:dict(pid=v.process.pid, alive=v.process.is_alive()) for k,v in self.active.items()},
                hardware=hardware()))+'\n'); self.monitor.flush(); self.last_monitor = now

    def start(self, role, case):
        self.check()
        reference = self.schedule.reference(case)
        confirmation = None
        if role == 'collection':
            self.schedule.resources_before_collection(hardware())
            require_native_slot(self.active['audit'].ticket if 'audit' in self.active else None)
            self.schedule.collection_started(case, reference)
        else:
            confirmation = self.schedule.pending['collection_confirmation_sha256']
            self.schedule.audit_started(case, confirmation)
        context = multiprocessing.get_context('spawn')
        receive, send = context.Pipe(duplex=False); release = context.Event(); cancel = context.Event()
        original_parent = collection_process.identity(psutil.Process())
        process = context.Process(target=worker_entry, args=(role, self.output, case, self.launch_sha,
            reference, confirmation, self.verifier, original_parent, send, release, cancel))
        active = Active(process, receive, release, cancel, case, reference, confirmation)
        try:
            process.start(); send.close(); self.active[role] = active
            while not receive.poll(1):
                if not process.is_alive(): raise ValueError('original child ended before ready handshake')
                self.sample()
            owner = receive.recv(); receive.close()
            if owner != collection_process.identity(psutil.Process(process.pid)):
                raise ValueError('ready handshake differs from actual child identity')
            identity = (owner['pid'], owner['created'])
            if identity in self.identities: raise ValueError('fresh process identity required for every stage')
            self.identities.add(identity)
            if role == 'collection':
                active.ticket = collection_process.register(process, self.output, case, self.launch_sha,
                    reference, self.verifier)
            else:
                active.ticket = audit_process.register(process, self.output, case, confirmation,
                    self.launch_sha, reference, self.verifier)
            release.set()
        except BaseException:
            send.close()
            if role not in self.active:
                receive.close(); self.schedule.failed(role, case, 'spawn failed before an owned child was available')
            elif not release.is_set():
                cancel.set(); release.set()
            raise

    def poll(self):
        for role, active in tuple(self.active.items()):
            if active.process.is_alive(): continue
            active.process.join()
            try:
                if active.ticket is None: raise ValueError('child ended without completed parent registration')
                if role == 'collection':
                    handoff_sha = digest(artifact_path(self.output, active.case.name+handoff.SUFFIX))
                    collection_process.confirm(active.ticket, handoff_sha, self.verifier)
                    sha = digest(artifact_path(self.output, active.case.name+collection_process.CONFIRMATION))
                    self.schedule.collection_finished(active.case, sha)
                else:
                    name = active.case.name+separate.EXECUTION
                    execution = evidence.read_json(self.output, name)
                    returned = dict(status=separate.WORKER_STATUS, case=active.case.name,
                        worker_terminal_sha256=execution['worker_terminal_sha256'],
                        separate_audit_execution_sha256=digest(artifact_path(self.output, name)))
                    accepted = audit_process.accept(active.ticket, returned, self.verifier)
                    self.schedule.audit_finished(active.case, accepted)
                    self.bindings.update(accepted['artifact_sha256'])
                    name = active.case.name+audit_process.COMPLETION
                    self.bindings[name] = digest(artifact_path(self.output, name))
            except BaseException as error:
                self.schedule.failed(role, active.case, repr(error))
            finally:
                registry = collection_process._tickets if role == 'collection' else audit_process._tickets
                registry.pop(active.process, None)
                active.ready.close(); active.process.close(); del self.active[role]


def drive(driver):
    """Run to completion or drain original workers after the first failure."""
    while True:
        try:
            driver.poll()
            action, name = driver.schedule.next_action()
            if action in ('complete', 'failed'): return action
            if action in ('collection', 'audit'):
                driver.start(action, next(case for case in CASES if case.name == name))
            else:
                time.sleep(1)
            driver.sample()
        except BaseException as error:
            driver.schedule.stop_dispatch(repr(error))


def run_population(output, launch_sha, verifier, overlap_verifier):
    output = validate_root(output)
    if {p.name for p in output.iterdir()} != {'launch.json'}:
        raise ValueError('fresh launch-only staged population root required; no retry or resume')
    started = time.perf_counter(); driver = None
    try:
        launch = checked_launch(output, launch_sha, verifier, overlap_verifier, full=True)
        with (output/'resource_monitor.jsonl').open('x') as monitor:
            driver = Driver(output, launch_sha, verifier, overlap_verifier, monitor)
            if drive(driver) != 'complete': raise RuntimeError('original staged population failed after draining')
        ordered = [[row['case'], row['worker_terminal_sha256']] for row in driver.schedule.accepted]
        population = evidence.complete_saved_population(output, ordered, launch_sha256=launch_sha)
        if any(driver.bindings.get(k) != v for k,v in population['artifact_sha256'].items()):
            raise ValueError('population readout differs from original parent acceptance')
        checked_launch(output, launch_sha, verifier, overlap_verifier, full=True)
        driver.bindings['resource_monitor.jsonl'] = digest(artifact_path(output, 'resource_monitor.jsonl'))
        verify_artifacts(output, driver.bindings)
        result = dict(status=COMPLETE, source_sha256=launch['source_sha256'], launch_sha256=launch_sha,
            artifact_sha256=driver.bindings, ordered_worker_sha256=ordered, population_readout=population['summary'],
            completed_episodes=len(ordered), independent_layout_units=8, fresh_stage_processes=len(driver.identities),
            all_fixed_cases_executed=True, all_original_raw_audits_returned=True,
            same_process_collection_and_raw_audit=False, native_execution=True, model_training=False,
            automatic_retry=False, wall_s=time.perf_counter()-started, parallel_speedup_measured=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
        write_json(output/'result.json', result)
        return result
    except BaseException as error:
        # drive drains every original started worker before returning failure.
        write_json(output/'failure.json', dict(status='TERMINAL_INDEPENDENT_STAGED_POPULATION_FAILURE',
            launch_sha256=launch_sha, reason=repr(error), traceback=traceback.format_exc(),
            schedule=None if driver is None else driver.schedule.snapshot(),
            known_accepted_artifact_sha256={} if driver is None else driver.bindings,
            automatic_retry=False, evidence_preserved=True))
        raise
