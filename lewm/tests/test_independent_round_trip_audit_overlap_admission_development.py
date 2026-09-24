"""Proof scope, exact runtime binding and independent final-admission checks."""
from copy import deepcopy
import json

import pytest
from scripts import independent_round_trip_audit_overlap_admission_development as overlap
from lewm.tests.test_independent_round_trip_population_runtime_development import (
    environment, synthetic_verifier, rewrite_launch)


def monitor():
    return dict(scope_returned_without_error=True, violations=[], initial_opencl_enabled=False,
        opencl_disabled_inside_scope=True, original_opencl_setting_restored=True,
        tensor_inputs_outputs_and_device_requests_checked=True, genesis_runtime_calls_prohibited=True,
        accelerator_initialization_prohibited=True, new_python_threads_prohibited=True,
        os_device_access_isolated=False, full_raw_auditor_qualified=False, overlap_execution_permitted=False,
        torch_operation_count=2, python_calls=10, native_python_calls=5, torch_operations={'synthetic':2})


@pytest.mark.parametrize('fault', ['violation','opencl','counter','bool_counter','no_calls','broader_claim'])
def test_invalid_monitor_cannot_become_overlap_evidence(fault):
    m = monitor()
    if fault == 'violation': m['violations'] = ['device']
    elif fault == 'opencl': m['initial_opencl_enabled'] = True
    elif fault == 'counter': m['torch_operation_count'] += 1
    elif fault == 'bool_counter': m['python_calls'] = True
    elif fault == 'no_calls': m['python_calls'] = 0
    else: m['full_raw_auditor_qualified'] = True
    with pytest.raises(ValueError): overlap.require_cpu(m, learned=True)


def test_reactive_does_not_require_neural_tensor_operations():
    m = monitor(); m['torch_operation_count'] = 0; m['torch_operations'] = {}
    overlap.require_cpu(m, learned=False)
    with pytest.raises(ValueError): overlap.require_cpu(m, learned=True)


def factory():
    originals = [{'synthetic_arm':c.arm_name} for c in overlap.startup.CASES[:4]]
    reports = {c.arm_name:monitor() for c in overlap.startup.CASES[:4]}
    streams = {c.arm_name:[{'tick':i,'decision':{'synthetic':True}} for i in range(4)] for c in overlap.startup.CASES[:4]}
    result = dict(status='INDEPENDENT_FACTORY_CPU_MONITOR_STARTUP_V1_COMPLETE',
        reference_result_sha256=overlap.startup.REFERENCE_SHA, original_startup_outputs_unchanged=True,
        full_raw_auditor_qualified=False, overlap_execution_permitted=False, native_execution=False,
        independent_layout_sensor_data_consumed=False, command_executed=False, goal_achieved=False,
        reports=[dict(arm=c.arm_name, original_report=o, cpu_monitor=reports[c.arm_name],
            complete_saved_startup_decisions_exact=True, frames=4, full_raw_auditor_qualified=False,
            overlap_execution_permitted=False) for c,o in zip(overlap.startup.CASES[:4], originals, strict=True)])
    return result, {'reports':originals}, reports, streams, deepcopy(streams)


@pytest.mark.parametrize('fault', [None,'missing_arm','changed_stream','changed_report','scope'])
def test_all_fixed_arms_and_complete_saved_decisions_required(fault):
    args = factory()
    if fault == 'missing_arm': args[0]['reports'].pop()
    elif fault == 'changed_stream': next(iter(args[3].values()))[-1]['decision'] = {'changed':True}
    elif fault == 'changed_report': args[0]['reports'][0]['original_report'] = {'changed':True}
    elif fault == 'scope': args[0]['full_raw_auditor_qualified'] = True
    if fault:
        with pytest.raises(ValueError): overlap.require_startup(*args)
    else: overlap.require_startup(*args)


def short():
    m = monitor(); audit = {'verified_round_trip':False,'synthetic_physics':{'contact':False}}
    r = dict(status='SHORT_COMPLETE_RAW_AUDIT_CPU_MONITOR_V1_COMPLETE',
        reference_result_sha256=overlap.short.REFERENCE_SHA, complete_original_raw_inputs_rehashed_before_and_after=True,
        independent_multiarm_raw_audit_qualified=False, overlap_execution_permitted=False,
        native_execution=False, independent_layout_sensor_data_consumed=False, goal_achieved=False,
        report=dict(complete_original_audit_report_exact=True, original_failed_navigation_preserved=True,
            observations=14, physics_samples=1400, command_intervals=13,
            model_loading_and_geometry_construction_monitored=True, cpu_monitor=m,
            independent_multiarm_raw_audit_qualified=False, overlap_execution_permitted=False, command_executed=False))
    return r,m,audit,deepcopy(audit)


@pytest.mark.parametrize('fault', [None,'partial','changed_physics','success','claimed_full_audit'])
def test_complete_short_audit_and_negative_outcome_required(fault):
    args = short()
    if fault == 'partial': args[0]['report']['observations'] = 13
    elif fault == 'changed_physics': args[2]['synthetic_physics']['contact'] = True
    elif fault == 'success': args[2]['verified_round_trip'] = args[3]['verified_round_trip'] = True
    elif fault == 'claimed_full_audit': args[0]['independent_multiarm_raw_audit_qualified'] = True
    if fault:
        with pytest.raises(ValueError): overlap.require_short(*args)
    else: overlap.require_short(*args)


@pytest.fixture(scope='module')
def sources(): return overlap.prepared_sources()


@pytest.fixture
def launch(sources, monkeypatch):
    for k,v in overlap.monitored.ENVIRONMENT.items(): monkeypatch.setenv(k,v)
    return dict(source_sha256=deepcopy(sources), audit_cpu_monitor=deepcopy(overlap.monitored.FIXED),
        staged_runtime=deepcopy(overlap.driver.staged.FIXED), overlap_evidence=deepcopy(overlap.EVIDENCE),
        population_entrypoint=deepcopy(overlap.ENTRYPOINT),
        overlap_verifier=dict(source=overlap.SOURCE,function='verify_overlap'))


@pytest.mark.parametrize('full', [False,True])
def test_source_bound_verifier_calls_actual_evidence_reader_without_mutation(launch, monkeypatch, full):
    calls = []
    monkeypatch.setattr(overlap,'admit_proofs',lambda sources,**kwargs:calls.append((deepcopy(sources),kwargs)))
    before = deepcopy(launch)
    assert overlap.verify_overlap(launch, full=full) is None
    assert launch == before and calls == [(launch['source_sha256'], {'full':full})]


@pytest.mark.parametrize('fault', ['source','monitor','capacity','entrypoint','evidence','verifier','boolean'])
def test_changed_execution_or_evidence_is_rejected_before_probe_admission(launch, monkeypatch, fault):
    if fault == 'source': launch['source_sha256'][overlap.driver.SOURCE] = 'a'*64
    elif fault == 'monitor': launch['audit_cpu_monitor']['monitoring_failure_rejects_case'] = False
    elif fault == 'capacity': launch['staged_runtime']['maximum_active_collectors'] = 2
    elif fault == 'entrypoint': launch['population_entrypoint']['source'] = overlap.driver.staged.SOURCE
    elif fault == 'evidence': launch['overlap_evidence']['factory_cpu_startup_result_sha256'] = 'b'*64
    elif fault == 'verifier': launch['overlap_verifier']['function'] = 'another_function'
    else: launch['audit_cpu_monitor']['actual_independent_auditor_monitored'] = 1
    monkeypatch.setattr(overlap,'admit_proofs',lambda *a,**k:pytest.fail('changed launch admitted'))
    with pytest.raises(ValueError): overlap.verify_overlap(launch)


def test_evidence_failure_propagates(launch, monkeypatch):
    def rejected(*a,**k): raise ValueError('synthetic changed completed proof')
    monkeypatch.setattr(overlap,'admit_proofs',rejected)
    with pytest.raises(ValueError,match='changed completed proof'): overlap.verify_overlap(launch)


def test_real_driver_still_requires_separate_final_verifier(environment, launch, monkeypatch):
    env = environment; env['launch'].update(launch); env['sha'] = rewrite_launch(env)
    calls=[]
    monkeypatch.setattr(overlap,'admit_proofs',lambda *a,**k:calls.append(k))
    with pytest.raises(ValueError,match='source-bound top-level'):
        overlap.driver.checked_launch(env['output'],env['sha'],None,overlap.verify_overlap)
    assert not calls
    assert overlap.driver.checked_launch(env['output'],env['sha'],synthetic_verifier,overlap.verify_overlap)
    assert calls == [{'full':False}]


def test_entry_calls_fixed_driver_and_preserves_final_verifier(monkeypatch):
    final=object(); calls=[]
    monkeypatch.setattr(overlap.driver,'run_population',lambda *args:calls.append(args) or 'synthetic_result')
    assert overlap.run_population('output','sha',final) == 'synthetic_result'
    assert calls == [('output','sha',final,overlap.verify_overlap)]
