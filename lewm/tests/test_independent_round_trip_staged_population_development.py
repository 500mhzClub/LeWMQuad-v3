"""Driver orchestration with synthetic processes/data and real spawn barriers."""
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import multiprocessing
import os
from types import SimpleNamespace
import time

import psutil
import pytest
from scripts import independent_round_trip_staged_population_development as staged
from scripts import independent_round_trip_collection_handoff_development as handoff
from scripts import independent_round_trip_collection_process_development as collection_process
from scripts import independent_round_trip_audit_process_development as audit_process
from scripts import independent_round_trip_separate_audit_development as separate
from scripts import independent_round_trip_population_runtime_development as runtime
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest
from lewm.tests.test_independent_round_trip_population_runtime_development import (
    environment, install_worker_stubs, rewrite_launch, synthetic_verifier)
from lewm.tests.test_independent_round_trip_population_schedule_development import completion
from lewm.tests.test_independent_round_trip_collection_process_development import workers
from lewm.independent_round_trip_comparison_study_development import CASES


def hashed(value): return sha256(value.encode()).hexdigest()


def overlap_ok(launch, *, full=False):
    if not launch.get('synthetic_overlap_evidence'):
        raise ValueError('synthetic CPU-only qualification missing')


def overlap_false(launch, *, full=False): return False


def overlap_mutates(launch, *, full=False): launch['mutated'] = True


def barrier_probe(connection, release, cancel, owner):
    connection.send('ready')
    try:
        staged.await_start(release, cancel, owner)
        connection.send('released')
    except ValueError:
        connection.send('rejected')
    finally:
        connection.close()


@pytest.mark.parametrize('cancelled', [False, True])
def test_real_spawn_waits_for_explicit_parent_release_or_cancellation(cancelled):
    context = multiprocessing.get_context('spawn'); receive, send = context.Pipe(duplex=False)
    release = context.Event(); cancel = context.Event()
    owner = collection_process.identity(psutil.Process())
    child = context.Process(target=barrier_probe, args=(send, release, cancel, owner))
    child.start(); send.close()
    try:
        assert receive.poll(15) and receive.recv() == 'ready'
        assert not receive.poll(.1) and child.is_alive()
        if cancelled: cancel.set()
        release.set()
        assert receive.poll(5)
        assert receive.recv() == ('rejected' if cancelled else 'released')
        child.join(5); assert child.exitcode == 0
    finally:
        cancel.set(); release.set(); child.join(5)
        if child.is_alive(): child.terminate(); child.join(5)
        receive.close(); child.close()


@pytest.fixture
def admitted_launch(environment):
    env = environment
    names = (handoff.SOURCE, handoff.TEST, handoff.PROTOCOL,
        staged.SOURCE, staged.TEST, staged.PROTOCOL, staged.scheduling.SOURCE,
        staged.scheduling.TEST, staged.scheduling.PROTOCOL,
        audit_process.SOURCE, audit_process.TEST, audit_process.PROTOCOL)
    env['launch']['source_sha256'].update({n:digest(ROOT/n) for n in names})
    env['launch'].update(staged_runtime=deepcopy(staged.FIXED), synthetic_overlap_evidence=True,
        overlap_verifier=dict(source=staged.TEST, function='overlap_ok'))
    env['sha'] = rewrite_launch(env)
    return env


def test_source_bound_overlap_verifier_checks_evidence_in_addition_to_final_admission(admitted_launch):
    env = admitted_launch
    assert staged.checked_launch(env['output'], env['sha'], synthetic_verifier, overlap_ok, full=True)
    env['sha'] = rewrite_launch(env, synthetic_overlap_evidence=False)
    with pytest.raises(ValueError, match='qualification missing'):
        staged.checked_launch(env['output'], env['sha'], synthetic_verifier, overlap_ok)


@pytest.mark.parametrize('verifier', [overlap_false, overlap_mutates])
def test_overlap_verifier_cannot_return_false_or_change_launch(admitted_launch, verifier):
    env = admitted_launch
    sha = rewrite_launch(env, overlap_verifier=dict(source=staged.TEST, function=verifier.__name__))
    with pytest.raises(ValueError): staged.checked_launch(env['output'], sha, synthetic_verifier, verifier)


def test_changed_staged_runtime_or_unbound_verifier_rejected(admitted_launch):
    env = admitted_launch; changed = deepcopy(staged.FIXED); changed['maximum_active_collectors'] = True
    sha = rewrite_launch(env, staged_runtime=changed)
    with pytest.raises(ValueError, match='fixed staged runtime'):
        staged.checked_launch(env['output'], sha, synthetic_verifier, overlap_ok)
    sha = rewrite_launch(env, staged_runtime=deepcopy(staged.FIXED))
    with pytest.raises(ValueError): staged.checked_launch(env['output'], sha, synthetic_verifier, lambda *a:None)


def test_collector_closes_log_and_publishes_full_handoff_without_audit(admitted_launch, monkeypatch):
    env = admitted_launch; events = install_worker_stubs(env, monkeypatch)
    monkeypatch.setattr(staged, 'collect', runtime.collect)
    monkeypatch.setattr(staged, 'ArticulatedCollisionGeometry', runtime.ArticulatedCollisionGeometry)
    staged.collection_worker(env['output'], CASES[0], env['sha'], None, synthetic_verifier)
    assert events == ['collect']
    record = handoff.evidence.read_json(env['output'], CASES[0].name+handoff.SUFFIX)
    assert record['collection_owner']['pid'] == os.getpid()
    assert not record['raw_audit_completed'] and not record['parent_verified_zero_exit']
    handoff.verify_artifacts(env['output'], record['artifact_sha256'])


def test_collector_failure_has_non_successful_return_and_preserved_log(admitted_launch, monkeypatch):
    env = admitted_launch; install_worker_stubs(env, monkeypatch, failure='collection')
    monkeypatch.setattr(staged, 'collect', runtime.collect)
    monkeypatch.setattr(staged, 'ArticulatedCollisionGeometry', runtime.ArticulatedCollisionGeometry)
    with pytest.raises(RuntimeError): staged.collection_worker(env['output'], CASES[0], env['sha'], None, synthetic_verifier)
    record = handoff.evidence.read_json(env['output'], CASES[0].name+handoff.FAILURE_SUFFIX)
    assert record['evidence_preserved'] and not record['automatic_retry']
    assert not (env['output']/(CASES[0].name+handoff.SUFFIX)).exists()


def test_cancelled_entry_never_calls_collection_and_retains_failure(environment, monkeypatch):
    release = multiprocessing.get_context('spawn').Event(); release.set()
    cancel = multiprocessing.get_context('spawn').Event(); cancel.set()
    ready = SimpleNamespace(send=lambda value:None, close=lambda:None)
    monkeypatch.setattr(staged, 'collection_worker', lambda *a:pytest.fail('cancelled worker collected'))
    owner = collection_process.identity(psutil.Process())
    with pytest.raises(ValueError, match='cancelled'):
        staged.worker_entry('collection', environment['output'], CASES[0], 'a'*64,
            None, None, synthetic_verifier, owner, ready, release, cancel)
    assert (environment['output']/(CASES[0].name+'_collection_entry_failure.json')).exists()


class FakeEvent:
    def __init__(self): self.value = False
    def set(self): self.value = True
    def is_set(self): return self.value


@pytest.fixture
def driver(environment, monkeypatch):
    output = environment['output']; launched = []; processes = []; faults = {}
    def owner(process): return dict(pid=process.pid, created=float(process.pid), command=['synthetic', str(process.pid)])
    monkeypatch.setattr(staged, 'psutil', SimpleNamespace(Process=lambda pid=None:SimpleNamespace(pid=pid or 4000)))
    monkeypatch.setattr(collection_process, 'identity', owner)
    monkeypatch.setattr(staged, 'hardware', lambda:dict(memory_available_bytes=80*1024**3,
        artifact_free_bytes=600*1024**3, physical_cpus=16))
    monkeypatch.setattr(staged, 'time', SimpleNamespace(sleep=lambda seconds:None, monotonic=time.monotonic,
        perf_counter=time.perf_counter))
    instance = staged.Driver(output, 'a'*64, None, None, StringIO())
    monkeypatch.setattr(instance, 'check', lambda full=False:{})
    monkeypatch.setattr(instance, 'sample', lambda **kwargs:None)
    def native_slot(ticket=None):
        assert not any(p.role == 'collection' and not p.finished for p in processes)
        assert ticket is None or ticket.process.role == 'audit'
    monkeypatch.setattr(staged, 'require_native_slot', native_slot)
    class FakeProcess:
        def __init__(self, target, args):
            assert target is staged.worker_entry
            self.role, _, self.case = args[:3]; self.release, self.cancel = args[-2:]
            self.pid = 5000+len(processes); self.finished = False; self.exitcode = None
            self.remaining = 2 if self.role == 'collection' else 7
            processes.append(self)
        def start(self): launched.append((self.role, CASES.index(self.case)))
        def is_alive(self):
            if self.finished: return False
            if not self.release.is_set(): return True
            self.remaining -= 1
            if self.remaining > 0: return True
            self.finished = True
            self.exitcode = 1 if self.cancel.is_set() or faults.get((self.role, CASES.index(self.case))) else 0
            if self.exitcode == 0:
                if self.role == 'collection':
                    (output/(self.case.name+handoff.SUFFIX)).write_text('{}\n')
                else:
                    (output/(self.case.name+separate.EXECUTION)).write_text(json.dumps(
                        dict(worker_terminal_sha256=hashed(self.case.name+' terminal')))+'\n')
            return False
        def join(self): assert self.finished
        def close(self): assert self.finished
    class Context:
        def Pipe(self, duplex):
            assert duplex is False
            return (SimpleNamespace(poll=lambda seconds:True, recv=lambda:owner(processes[-1]), close=lambda:None),
                SimpleNamespace(close=lambda:None))
        Event = FakeEvent
        Process = FakeProcess
    monkeypatch.setattr(staged, 'multiprocessing', SimpleNamespace(get_context=lambda name:Context()))
    def register(process, *args):
        if faults.get(('registration', CASES.index(process.case))): raise ValueError('registration rejected')
        return SimpleNamespace(process=process)
    monkeypatch.setattr(collection_process, 'register', register)
    monkeypatch.setattr(audit_process, 'register', register)
    def confirm(ticket, sha, verifier):
        assert ticket.process.finished and ticket.process.exitcode == 0
        (output/(ticket.process.case.name+collection_process.CONFIRMATION)).write_text('{}\n')
    monkeypatch.setattr(collection_process, 'confirm', confirm)
    def accept(ticket, returned, verifier):
        assert ticket.process.finished and ticket.process.exitcode == 0
        case = ticket.process.case
        record = completion(instance.schedule, case)
        record['collection_confirmation_sha256'] = instance.schedule.audit['collection_confirmation_sha256']
        record['artifact_sha256'] = {}
        (output/(case.name+audit_process.COMPLETION)).write_text(json.dumps(record)+'\n')
        return record
    monkeypatch.setattr(audit_process, 'accept', accept)
    return instance, launched, processes, faults


def test_driver_runs_all_32_cases_through_64_distinct_stages(driver):
    instance, launched, processes, _ = driver
    assert staged.drive(instance) == 'complete'
    assert len(processes) == len(instance.identities) == 64
    assert not instance.active and len(instance.schedule.accepted) == 32
    for role in ('collection', 'audit'):
        assert [index for kind,index in launched if kind == role] == list(range(32))


@pytest.mark.parametrize('stage,index', [('collection', 0), ('collection', 2), ('audit', 0), ('audit', 1), ('audit', 31)])
def test_driver_drains_existing_children_and_never_restarts_failed_stage(driver, stage, index):
    instance, launched, processes, faults = driver; faults[(stage, index)] = True
    assert staged.drive(instance) == 'failed'
    assert not instance.active and all(p.finished for p in processes)
    assert launched.count((stage, index)) == 1
    assert len(instance.schedule.accepted) == index
    if stage == 'audit' and index == 1:
        assert ('collection', 2) in launched and ('audit', 2) not in launched
        assert instance.schedule.pending['case'] == CASES[2].name


def test_registration_failure_cancels_unreleased_child_and_drains_it(driver):
    instance, launched, processes, faults = driver; faults[('registration', 0)] = True
    assert staged.drive(instance) == 'failed'
    assert launched == [('collection', 0)]
    assert processes[0].cancel.is_set() and processes[0].finished and not instance.active


def test_unregistered_native_worker_is_never_generically_exempted(monkeypatch):
    monkeypatch.setattr(staged, 'competitors', lambda:[dict(pid=8888, started=1., command=['multiprocessing.spawn'])])
    with pytest.raises(ValueError, match='unregistered worker'): staged.require_native_slot()
    with pytest.raises(ValueError, match='original active audit ticket'):
        staged.require_native_slot(SimpleNamespace(process=object()))


def test_only_exact_registered_live_owned_audit_is_exempted(environment, workers, monkeypatch):
    child = workers(); output = environment['output']; case = CASES[0]
    record = dict(audit_owner=child[2], parent_owner=collection_process.identity(psutil.Process()))
    path = output/(case.name+audit_process.REGISTRATION); path.write_text(json.dumps(record)+'\n')
    ticket = audit_process.Ticket(child[0], output, case, 'a'*64, 'b'*64, None, digest(path))
    monkeypatch.setattr(audit_process, '_tickets', {child[0]:ticket})
    observed = [dict(pid=child[2]['pid'], started=child[2]['created'], command=child[2]['command'])]
    monkeypatch.setattr(staged, 'competitors', lambda:observed)
    staged.require_native_slot(ticket)
    observed.append(dict(pid=8888, started=1., command=['unregistered native runner']))
    with pytest.raises(ValueError, match='unregistered worker'): staged.require_native_slot(ticket)
    observed.pop(); observed[0]['started'] += 1
    with pytest.raises(ValueError, match='unregistered worker'): staged.require_native_slot(ticket)
    observed[0]['started'] -= 1
    path.write_text('changed original registration')
    with pytest.raises(ValueError): staged.require_native_slot(ticket)


def test_release_cannot_override_original_parent_death(monkeypatch):
    monkeypatch.setattr(handoff, 'owner_live', lambda owner:False)
    release = SimpleNamespace(wait=lambda seconds:True)
    cancel = SimpleNamespace(is_set=lambda:False)
    with pytest.raises(ValueError, match='original parent ended'):
        staged.await_start(release, cancel, {})
