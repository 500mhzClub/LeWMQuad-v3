"""Startup protocol checks and small synthetic monitored scopes only."""
from copy import deepcopy
import json

import pytest
import torch
from scripts import replay_go2_independent_factory_cpu_monitor_startup_v1 as probe


def reference():
    sources = {'synthetic.py':'a'*64}
    launch = dict(source_sha256=sources, original_result_sha256=probe.original.evidence.ORIGINAL_SHA,
        original_raw_startup_case=probe.original.INPUT_CASE,
        planned_cases=[vars(c) for c in probe.CASES[:4]])
    reports = [dict(arm=c.arm_name, assigned_case=c.name, frames=4,
        model_state_sha256=probe.require_case(c).model_state_sha256, model_state_unchanged=True,
        no_packet_after_new_planning_command_consumed=True, command_executed=False,
        independent_layout_sensor_data_consumed=False) for c in probe.CASES[:4]]
    result = dict(status='INDEPENDENT_ADAPTER_FACTORY_STARTUP_V1_COMPLETE', source_sha256=sources,
        artifact_sha256={'launch.json':probe.REFERENCE_LAUNCH_SHA,
            **{c.arm_name+'/'+probe.NAME:'b'*64 for c in probe.CASES[:4]}}, reports=reports,
        native_execution=False, model_training=False, new_layout_sensor_data_consumed=False,
        new_layout_navigation_execution=False, goal_achieved=False)
    return result, launch, sources


def test_exact_reference_structure_accepted():
    probe.require_reference(*reference())


@pytest.mark.parametrize('fault', ['extra_artifact','source','frames','model','order','command','scope'])
def test_changed_reference_rejected(fault):
    result, launch, sources = deepcopy(reference())
    if fault == 'extra_artifact': result['artifact_sha256']['unexpected.json'] = 'c'*64
    elif fault == 'source': result['source_sha256'] = {'changed.py':'c'*64}
    elif fault == 'frames': result['reports'][0]['frames'] = 5
    elif fault == 'model': result['reports'][0]['model_state_sha256'] = 'c'*64
    elif fault == 'order': result['reports'].reverse()
    elif fault == 'command': result['reports'][0]['command_executed'] = True
    else: result['new_layout_sensor_data_consumed'] = True
    with pytest.raises(ValueError): probe.require_reference(result, launch, sources)


def test_original_startup_body_and_globals_preserved(tmp_path, monkeypatch):
    old_root = probe.original.OUTPUT; original_globals = probe.original.run_case.__globals__
    monkeypatch.setattr(probe, 'OUTPUT', tmp_path/'new_output')
    clone = probe.isolated_case()
    assert clone.__code__ is probe.original.run_case.__code__
    assert clone.__globals__ is not original_globals and clone.__globals__['OUTPUT'] == probe.OUTPUT
    assert probe.original.OUTPUT == old_root and original_globals['OUTPUT'] == old_root
    assert all(clone.__globals__[key] is value for key,value in original_globals.items() if key != 'OUTPUT')


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'OUTPUT', tmp_path)
    report = {'synthetic':True}
    rows = [dict(tick=i, decision={'synthetic':True}) for i in range(4)]
    def install(*, changed=None, failure=False):
        def execute(case, admission):
            if failure: torch.empty(1, device='meta')
            if case.arm_name != 'reactive': torch.ones(1) + 1
            directory = tmp_path/case.arm_name; directory.mkdir()
            written = deepcopy(rows)
            if changed == 'rows': written[-1]['decision']['synthetic'] = False
            with probe.original.writer(directory) as append:
                for row in written: append(row)
            return {'changed':True} if changed == 'report' else deepcopy(report)
        monkeypatch.setattr(probe, 'isolated_case', lambda:execute)
    install()
    return report, rows, install


@pytest.mark.parametrize('case', probe.CASES[:4])
def test_all_fixed_arm_scopes_record_monitor_and_compare_outputs(synthetic, case):
    report, rows, _ = synthetic
    result = probe.monitored_case(case, {}, report, rows)
    assert result['complete_saved_startup_decisions_exact'] and result['frames'] == 4
    assert not result['full_raw_auditor_qualified'] and not result['overlap_execution_permitted']
    monitor = json.loads((probe.OUTPUT/(case.arm_name+'_cpu_monitor.json')).read_text())
    assert monitor['scope_returned_without_error'] and not monitor['initial_opencl_enabled']
    assert bool(monitor['torch_operation_count']) == (case.arm_name != 'reactive')


@pytest.mark.parametrize('changed', ['rows','report'])
def test_changed_output_does_not_become_a_successful_cpu_qualification(synthetic, changed):
    report, rows, install = synthetic; install(changed=changed)
    with pytest.raises(ValueError, match='changed complete saved startup evidence'):
        probe.monitored_case(probe.CASES[0], {}, report, rows)


def test_monitor_failure_receipt_is_retained(synthetic):
    report, rows, install = synthetic; install(failure=True); case = probe.CASES[0]
    with pytest.raises(ValueError, match='non-CPU device request'):
        probe.monitored_case(case, {}, report, rows)
    monitor = json.loads((probe.OUTPUT/(case.arm_name+'_cpu_monitor.json')).read_text())
    assert monitor['violations'] and not monitor['scope_returned_without_error']
