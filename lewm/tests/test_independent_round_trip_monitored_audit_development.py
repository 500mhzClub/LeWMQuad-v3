"""CPU receipt enforcement on synthetic data and real owned spawn handles."""
from copy import deepcopy
import json
import pickle
from io import StringIO
from types import SimpleNamespace

import pytest
import torch
from scripts import independent_round_trip_monitored_audit_development as monitored
from scripts import independent_round_trip_monitored_staged_population_development as driver
from scripts import independent_round_trip_audit_process_development as parent
from scripts import independent_round_trip_separate_audit_development as separate
from scripts import independent_round_trip_collection_process_development as lifecycle
from scripts import independent_round_trip_collection_handoff_development as handoff
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest
from lewm.tests.test_independent_round_trip_population_case_evidence_development import saved
from lewm.tests.test_independent_round_trip_collection_handoff_development import setup
from lewm.tests.test_independent_round_trip_separate_audit_development import fixture as base_fixture, worker
from lewm.tests.test_independent_round_trip_audit_process_development import fixture as parent_fixture
from lewm.tests.test_independent_round_trip_collection_process_development import workers, finish
from lewm.tests.test_independent_round_trip_population_runtime_development import environment
from lewm.tests.test_independent_round_trip_staged_population_development import driver as legacy_driver


@pytest.fixture
def fixture(parent_fixture, monkeypatch):
    f = parent_fixture; output = f['output']; case = f['case']
    for key,value in monitored.ENVIRONMENT.items(): monkeypatch.setenv(key,value)
    names = (monitored.SOURCE, monitored.TEST, monitored.PROTOCOL, driver.SOURCE,
        monitored.cpu.SOURCE, monitored.cpu.TEST, monitored.cpu.PROTOCOL)
    f['launch']['source_sha256'].update({n:digest(ROOT/n) for n in names})
    f['launch']['audit_cpu_monitor'] = deepcopy(monitored.FIXED)
    (output/'launch.json').write_text(json.dumps(f['launch'])+'\n')
    f['launch_sha'] = digest(output/'launch.json')
    for suffix in (handoff.SUFFIX, lifecycle.REGISTRATION, lifecycle.CONFIRMATION):
        path = output/(case.name+suffix); record = json.loads(path.read_text())
        record['source_sha256'] = f['launch']['source_sha256']; record['launch_sha256'] = f['launch_sha']
        if 'artifact_sha256' in record: record['artifact_sha256']['launch.json'] = f['launch_sha']
        if suffix == lifecycle.CONFIRMATION:
            for prior in (handoff.SUFFIX, lifecycle.REGISTRATION):
                name = case.name+prior; record['artifact_sha256'][name] = digest(output/name)
            f['confirmation'] = record
        path.write_text(json.dumps(record)+'\n')
    f['confirmation_sha'] = digest(output/(case.name+lifecycle.CONFIRMATION))
    return f


def run(f):
    return monitored.audit_worker(f['output'], f['case'], f['confirmation_sha'], f['launch_sha'], None, None)


def read(f, returned):
    return monitored.read_completed_audit(f['output'], f['case'], returned,
        f['confirmation_sha'], f['launch_sha'], None, None)


def register(f, child):
    return monitored.parent_register(child[0], f['output'], f['case'], f['confirmation_sha'],
        f['launch_sha'], None, None)


def test_actual_worker_wrapper_monitor_and_complete_saved_report(fixture, worker):
    returned = run(fixture); assert returned['status'] == monitored.WORKER_STATUS
    admitted = read(fixture, returned)
    assert admitted['actual_raw_audit_cpu_scope_authenticated']
    name = fixture['case'].name+monitored.MONITOR
    assert admitted['artifact_sha256'][name] == returned['cpu_monitor_sha256']
    record = json.loads((fixture['output']/name).read_text())
    assert record['complete_raw_audit_report_sha256'] == monitored.report_hash(fixture['report'])
    assert record['monitor']['python_calls'] > 0
    assert not admitted['population_case_accepted']
    assert worker == ['geometry','original_audit']
    with pytest.raises(ValueError, match='no retry'): run(fixture)


@pytest.mark.parametrize('fault', ['device','caught_device','error','profile_removed','nonfinite_report'])
def test_monitor_or_audit_failure_retained_and_unacceptable(fixture, worker, monkeypatch, fault):
    def fail(*args, **kwargs):
        if fault == 'error': raise RuntimeError('synthetic original raw audit failure')
        if fault == 'nonfinite_report': return {'synthetic':float('nan')}
        if fault == 'profile_removed':
            import sys
            sys.setprofile(None)
        else:
            try: torch.empty(1, device='meta')
            except ValueError:
                if fault != 'caught_device': raise
        return deepcopy(fixture['report'])
    monkeypatch.setattr(separate, 'audit', fail)
    returned = run(fixture)
    assert returned['status'] == monitored.FAILED
    record = json.loads((fixture['output']/(fixture['case'].name+monitored.MONITOR)).read_text())
    assert record['status'] == 'INDEPENDENT_RAW_AUDIT_CPU_SCOPE_FAILED'
    if fault == 'nonfinite_report': assert record['report_encoding_error']
    else: assert not record['monitor']['scope_returned_without_error']
    assert (fixture['output']/(fixture['case'].name+'_worker_failure.json')).exists()
    with pytest.raises((ValueError,KeyError)): read(fixture, returned)


@pytest.mark.parametrize('fault', ['case','owner','report','violations','scope','count','null_monitor'])
def test_rebound_monitor_cannot_change_execution_identity_or_report(fixture, worker, fault):
    returned = run(fixture); name = fixture['case'].name+monitored.MONITOR
    path = fixture['output']/name; record = json.loads(path.read_text())
    if fault == 'case': record['case'] = 'another_case'
    elif fault == 'owner': record['audit_owner']['created'] += 1
    elif fault == 'report': record['complete_raw_audit_report_sha256'] = 'a'*64
    elif fault == 'violations': record['monitor']['violations'] = ['synthetic violation']
    elif fault == 'scope': record['monitor']['scope_returned_without_error'] = False
    elif fault == 'count': record['monitor']['torch_operation_count'] += 1
    else: record['monitor'] = None
    path.write_text(json.dumps(record)+'\n'); returned['cpu_monitor_sha256'] = digest(path)
    with pytest.raises(ValueError): read(fixture, returned)


def test_original_unmonitored_completion_cannot_pass_successor(fixture, worker):
    returned = separate.audit_worker(fixture['output'], fixture['case'], fixture['confirmation_sha'],
        fixture['launch_sha'], None, None)
    assert returned['status'] == separate.WORKER_STATUS
    with pytest.raises((ValueError,KeyError)): read(fixture, returned)


@pytest.mark.parametrize('code', [0,17])
def test_parent_requires_owned_zero_exit_and_binds_monitor(fixture, worker, workers, monkeypatch, code):
    child = workers(code); ticket = register(fixture, child)
    monkeypatch.setattr(separate, 'require_audit_child', lambda confirmation:child[2])
    returned = run(fixture); finish(child)
    if code:
        with pytest.raises(ValueError, match='code zero'): monitored.parent_accept(ticket, returned, None)
        assert not (fixture['output']/(fixture['case'].name+parent.COMPLETION)).exists()
    else:
        accepted = monitored.parent_accept(ticket, returned, None)
        assert accepted['population_case_accepted'] and not accepted['verified_round_trip']
        assert accepted['artifact_sha256'][fixture['case'].name+monitored.MONITOR] == returned['cpu_monitor_sha256']
        with pytest.raises(ValueError, match='unconsumed'): monitored.parent_accept(ticket, returned, None)


def test_parent_rejects_changed_cpu_receipt(fixture, worker, workers, monkeypatch):
    child = workers(); ticket = register(fixture, child)
    monkeypatch.setattr(separate, 'require_audit_child', lambda confirmation:child[2])
    returned = run(fixture); finish(child)
    (fixture['output']/(fixture['case'].name+monitored.MONITOR)).write_text('changed CPU evidence')
    with pytest.raises(ValueError): monitored.parent_accept(ticket, returned, None)
    assert not (fixture['output']/(fixture['case'].name+parent.COMPLETION)).exists()


def test_cloned_function_preserves_code_defaults_and_original_globals():
    before = dict(separate.audit_worker.__globals__); replacement = object()
    clone = monitored.isolated(separate.audit_worker, audit=replacement)
    assert clone.__code__ is separate.audit_worker.__code__ and clone.__globals__['audit'] is replacement
    assert all(separate.audit_worker.__globals__[k] is v for k,v in before.items())
    assert all(clone.__globals__[k] is v for k,v in before.items() if k != 'audit')
    clone = monitored.isolated(driver.staged.checked_launch)
    assert clone.__kwdefaults__ == {'full':False}
    assert pickle.loads(pickle.dumps(driver.worker_entry)) is driver.worker_entry


def test_driver_entry_routes_original_barrier_to_monitored_worker(monkeypatch):
    calls = []
    def fake(*args): calls.append(args); return 'original_body_result'
    def isolated(function, **overrides):
        assert function is driver.staged.worker_entry and overrides == {'separate':monitored}
        return fake
    monkeypatch.setattr(monitored, 'isolated', isolated)
    assert driver.worker_entry('synthetic') == 'original_body_result' and calls == [('synthetic',)]


def test_driver_poll_passes_cpu_receipt_to_parent(fixture, worker, workers, monkeypatch):
    child = workers(); ticket = register(fixture, child)
    monkeypatch.setattr(separate, 'require_audit_child', lambda confirmation:child[2])
    run(fixture); finish(child)
    # Driver owns closing; use a proxy for that one operation because the test
    # fixture retains the actual handle to perform its final cleanup.
    class ProcessProxy:
        def is_alive(self): return child[0].is_alive()
        def join(self): child[0].join()
        def close(self): pass
    process = ProcessProxy()
    active = SimpleNamespace(process=process, ready=SimpleNamespace(close=lambda:None),
        case=fixture['case'], ticket=ticket)
    d = driver.Driver(fixture['output'], fixture['launch_sha'], None, None, StringIO())
    accepted = []
    d.active['audit'] = active
    d.schedule = SimpleNamespace(audit_finished=lambda case,result:accepted.append(result),
        failed=lambda *a:pytest.fail(str(a)))
    d.poll()
    assert len(accepted) == 1 and not d.active
    assert fixture['case'].name+monitored.MONITOR in d.bindings


def test_missing_runtime_policy_or_environment_rejected(fixture, monkeypatch):
    monkeypatch.setenv('OPENCV_OPENCL_RUNTIME','enabled')
    with pytest.raises(ValueError, match='fixed CPU environment'): monitored.require_environment()
    monkeypatch.setenv('OPENCV_OPENCL_RUNTIME','disabled')
    fixture['launch']['audit_cpu_monitor']['monitoring_failure_rejects_case'] = False
    path = fixture['output']/'launch.json'; path.write_text(json.dumps(fixture['launch'])+'\n')
    with pytest.raises(ValueError, match='fixed monitored auditor'):
        monitored.require_launch(fixture['output'], digest(path))


@pytest.fixture
def staged_driver(legacy_driver, monkeypatch):
    instance, launched, processes, faults = legacy_driver
    instance.__class__ = driver.Driver
    context = driver.staged.multiprocessing.get_context('spawn')
    original_process = context.Process; targets = []
    def process(target, args):
        assert target is driver.worker_entry; targets.append(target)
        p = original_process(driver.staged.worker_entry, args)
        alive = p.is_alive
        def is_alive():
            result = alive()
            if not result and p.role == 'audit' and p.exitcode == 0:
                path = instance.output/(p.case.name+monitored.MONITOR)
                if not path.exists(): path.write_text('{"synthetic_monitor":true}\n')
            return result
        p.is_alive = is_alive
        return p
    monkeypatch.setattr(driver.staged, 'multiprocessing', SimpleNamespace(get_context=lambda name:
        SimpleNamespace(Process=process, Pipe=context.Pipe, Event=context.Event)))
    monkeypatch.setattr(driver, 'require_native_slot', driver.staged.require_native_slot)
    monkeypatch.setattr(monitored, 'parent_register', parent.register)
    old_accept = parent.accept
    def accept(ticket, returned, verifier):
        name = ticket.process.case.name+monitored.MONITOR
        assert returned['cpu_monitor_sha256'] == digest(instance.output/name)
        result = old_accept(ticket, returned, verifier)
        result['artifact_sha256'][name] = returned['cpu_monitor_sha256']
        return result
    monkeypatch.setattr(monitored, 'parent_accept', accept)
    return instance, launched, processes, faults, targets


@pytest.mark.parametrize('failure', [None,('audit',1),('audit',31),('collection',2),('registration',0)])
def test_monitored_driver_complete_population_or_drain_failure(staged_driver, failure):
    instance, launched, processes, faults, targets = staged_driver
    if failure: faults[failure] = True
    status = driver.staged.drive(instance)
    assert status == ('failed' if failure else 'complete')
    assert not instance.active and all(p.finished for p in processes)
    assert len(targets) == len(processes) == len(instance.identities)
    assert len(launched) == len(set(launched))
    if failure is None:
        assert len(processes) == 64 and len(instance.schedule.accepted) == 32
        assert len([n for n in instance.bindings if n.endswith(monitored.MONITOR)]) == 32
    elif failure == ('audit',1):
        assert ('collection',2) in launched and ('audit',2) not in launched
