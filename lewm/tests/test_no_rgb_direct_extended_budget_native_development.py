"""Launcher/worker boundary tests use synthetic dependencies, never a scene."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts import run_go2_no_rgb_direct_extended_budget_maze02_pilot_v1 as native


def resources():
    return dict(artifact_free_bytes=55*1024**3, memory_available_bytes=32*1024**3)


@pytest.mark.parametrize('key', ['artifact_free_bytes', 'memory_available_bytes'])
def test_complete_extended_resource_allowance_is_required(key):
    r = resources(); native.resource_admission(r)
    r[key] -= 1
    with pytest.raises(ValueError): native.resource_admission(r)


def test_input_owner_guard_precedes_expensive_admission(monkeypatch):
    def busy(): raise ValueError('live original queue')
    monkeypatch.setattr(native.inputs.queue, 'owners_ended', busy)
    monkeypatch.setattr(native.inputs, 'original_batch', lambda *a, **kw:pytest.fail('must not inspect large inputs'))
    with pytest.raises(ValueError, match='live original'): native.inputs.admit('a', {}, 'b', {})


def test_full_admission_includes_original_model_and_all_four_queued_stages(monkeypatch):
    calls = []; inputs = native.inputs
    monkeypatch.setattr(inputs.queue, 'owners_ended', lambda: calls.append('extended owners'))
    monkeypatch.setattr(inputs.queue.original, 'owners_ended', lambda: calls.append('original owners'))
    monkeypatch.setattr(inputs, 'verify', lambda s:None)
    def batch(sha, sources, *, full):
        assert full is True; calls.append('batch'); return dict(batch_result_sha256=sha)
    def original(ids, *, adapter_batch_result_sha256, sources, full):
        assert full is True and set(ids) == {'frontier', 'hold', 'contact'}
        calls.append('three stages'); return {'batch':adapter_batch_result_sha256}
    def extended(admission, sha, *, sources, full):
        assert full is True and sha == 'tracking'; calls.append('tracking'); return {'complete':True}
    monkeypatch.setattr(inputs, 'original_batch', batch)
    monkeypatch.setattr(inputs.queue.original, 'admit', original)
    monkeypatch.setattr(inputs.queue, 'admit', extended)
    a = inputs.admit('batch', dict(frontier='f', hold='h', contact='c'), 'tracking', {})
    assert calls == ['extended owners', 'original owners', 'batch', 'three stages', 'tracking']
    assert a['all_scientific_failures_retained'] and not a['queued_controller_changes_adopted']


def fixture_record(reached=False):
    collection = dict(navigation_ticks=4000, decisions=4014 if reached else 100,
        mission_receipt=dict(frame=4003 if reached else 89),
        status='RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED',
        storage_allowance_bytes=14*1024**3)
    report = dict(raw_sensor_reconstruction_pass=True, raw_command_audit_pass=True,
        raw_model_command_replay_pass=True, model_state_unchanged=True, verified_round_trip=False,
        native_evaluation=dict(native_round_trip_candidate_pass=False), strict_physical_visibility_pass=False,
        hard_measurement_failed_frames=[20], renderer_capture_audit={})
    if reached:
        receipt = dict(status='COMPLETE_ACTUAL_PREFIX_COMPARISON', actual_paired_execution_compared=True,
            frames=3004, physical_prefix_samples=150900, normalized_budget_paths=list(native.prefix.BUDGET_PATHS),
            following_observations_compared=False, unexecuted_outcomes_inferred=False,
            all_preboundary_decisions_exact=False, all_preboundary_commands_exact=True, physical_prefix_exact=True,
            all_through_boundary_public_packets_exact=True, budget_only_preboundary_execution_supported=False)
    else:
        receipt = native.prefix_result(collection, {}, {})
    record = dict(status=native.WORKER_STATUS, case=native.CASE[0], layout_index=2,
        variant=native.CASE[2], condition=native.CASE[3], model_name=native.CASE[4],
        model_state_sha256=native.inputs.MODEL_SHA, model_state_unchanged=True,
        collection=collection, prefix_comparison=receipt, **{k:report[k] for k in native.OUTCOME_KEYS})
    return record, report


@pytest.mark.parametrize('reached', [False, True])
def test_negative_native_and_prefix_results_remain_complete_scientific_evidence(reached):
    record, report = fixture_record(reached)
    native.require_worker(record, report)
    record['verified_round_trip'] = True
    with pytest.raises(ValueError): native.require_worker(record, report)


def test_visibility_failure_cannot_be_promoted_by_physical_candidate():
    record, report = fixture_record()
    record['native_evaluation']['native_round_trip_candidate_pass'] = True
    report['native_evaluation']['native_round_trip_candidate_pass'] = True
    native.require_worker(record, report)
    record['verified_round_trip'] = report['verified_round_trip'] = True
    with pytest.raises(ValueError): native.require_worker(record, report)


def test_prefix_equality_claim_cannot_hide_an_earlier_divergence():
    record, report = fixture_record(True)
    record['prefix_comparison']['budget_only_preboundary_execution_supported'] = True
    with pytest.raises(ValueError): native.require_worker(record, report)


@pytest.mark.parametrize('fault', [None, 'prefix_failure'])
def test_worker_persists_full_audit_before_prefix_and_keeps_failure_evidence(tmp_path, monkeypatch, fault):
    record, report = fixture_record(False); output = tmp_path/'attempt'; output.mkdir()
    monkeypatch.setattr(native, 'OUTPUT', output)
    launch = dict(source_sha256={native.PROTOCOL:'a'*64}, input_admission={})
    (output/'launch.json').write_text(json.dumps(launch))
    monkeypatch.setattr(native, 'verify_inputs', lambda l:None)
    monkeypatch.setattr(native, 'verify_artifacts', lambda *a:None)
    monkeypatch.setattr(native, 'hardware', resources)
    monkeypatch.setattr(native, 'assigned_model', lambda l:NS(state_dict=lambda:{}))
    monkeypatch.setattr(native, 'state_digest', lambda s:native.inputs.MODEL_SHA)
    monkeypatch.setattr(native, 'ArticulatedCollisionGeometry', lambda u:object())
    monkeypatch.setattr(native, 'case_readout', lambda *a:dict(verified_round_trip=False))
    def collect(*a, **kw):
        p = output/native.CASE[0]; p.mkdir()
        np.savez_compressed(p/'physics_trace.npz', physics_contact=np.zeros(900))
        return deepcopy(record['collection'])
    monkeypatch.setattr(native.pipeline, 'collect', collect)
    monkeypatch.setattr(native.pipeline, 'artifacts', lambda *a:['physics_trace.npz'])
    monkeypatch.setattr(native.pipeline, 'audit', lambda *a, **kw:deepcopy(report))
    original_prefix = native.prefix_result
    def checked_prefix(*a):
        assert (output/(native.CASE[0]+'_audit.json')).is_file()
        assert (output/(native.CASE[0]+'_readout.json')).is_file()
        if fault: raise ValueError('synthetic prefix failure')
        return original_prefix(*a)
    monkeypatch.setattr(native, 'prefix_result', checked_prefix)
    result = native.worker('a'*64)
    assert json.loads((output/(native.CASE[0]+'_worker_terminal.json')).read_text()) == result
    assert native.CASE[0]+'_audit.json' in result['artifact_sha256']
    assert native.CASE[0]+'_readout.json' in result['artifact_sha256']
    assert result['verified_round_trip'] is False
    if fault:
        assert result['status'] == 'NO_RGB_DIRECT_EXTENDED_BUDGET_WORKER_FAILED'
        assert 'synthetic prefix failure' in result['failure']
    else:
        assert result['status'] == native.WORKER_STATUS
        assert result['prefix_comparison']['status'] == 'ORIGINAL_BUDGET_BOUNDARY_NOT_REACHED'


def test_source_preflight_does_not_admit_inputs_construct_model_or_create_output(tmp_path, monkeypatch, capsys):
    output = tmp_path/'absent'
    monkeypatch.setattr(native, 'OUTPUT', output)
    monkeypatch.setattr(native, 'validate_root', lambda *a, **kw:None)
    monkeypatch.setattr(native.inputs, 'prepared_sources', lambda *a:{'synthetic':'a'*64})
    monkeypatch.setattr(native.inputs, 'admit', lambda *a:pytest.fail('no input admission'))
    monkeypatch.setattr(native, 'assigned_model', lambda *a:pytest.fail('no model'))
    monkeypatch.setattr(native, 'create_output', lambda *a:pytest.fail('no output'))
    monkeypatch.setattr(native, 'hardware', resources)
    monkeypatch.setattr(sys, 'argv', ['launcher', '--source-preflight-only'])
    native.main()
    assert 'EXTENDED_BUDGET_NATIVE_SOURCE_PREFLIGHT' in capsys.readouterr().out and not output.exists()


def test_existing_attempt_is_rejected_before_source_or_input_work(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'OUTPUT', tmp_path)
    monkeypatch.setattr(native, 'validate_root', lambda *a, **kw:None)
    monkeypatch.setattr(native.inputs, 'prepared_sources', lambda *a:pytest.fail('existing attempt must stop first'))
    monkeypatch.setattr(sys, 'argv', ['launcher', '--source-preflight-only'])
    with pytest.raises(ValueError, match='exclusive'): native.main()


@pytest.mark.parametrize('mode', ['--preflight-only', 'execute'])
def test_full_preflight_and_busy_native_gate_do_not_create_an_episode(tmp_path, monkeypatch, mode):
    output = tmp_path/'absent'; calls = []
    monkeypatch.setattr(native, 'OUTPUT', output)
    monkeypatch.setattr(native, 'validate_root', lambda *a, **kw:None)
    monkeypatch.setattr(native.inputs, 'prepared_sources', lambda *a:{native.PROTOCOL:'a'*64})
    monkeypatch.setattr(native.inputs, 'admit', lambda *a:calls.append('admitted') or {})
    keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment')
    monkeypatch.setattr(native, 'read_json', lambda *a:{k:{} for k in keys})
    monkeypatch.setattr(native, 'verify_inputs', lambda *a:calls.append('verified'))
    monkeypatch.setattr(native, 'assigned_model', lambda *a:calls.append('model'))
    monkeypatch.setattr(native, 'hardware', resources)
    def busy():
        calls.append('native gate'); raise ValueError('synthetic live native owner')
    monkeypatch.setattr(native, 'require_native_idle', busy)
    monkeypatch.setattr(native, 'create_output', lambda *a:pytest.fail('must not create an episode'))
    args = ['launcher']
    for key in ('adapter-batch', 'frontier-wait', 'hold-wait', 'contact-wait', 'tracking-wait'):
        args.extend(['--'+key+'-result-sha256', 'a'*64])
    if mode != 'execute': args.append(mode)
    monkeypatch.setattr(sys, 'argv', args)
    if mode == 'execute':
        with pytest.raises(ValueError, match='live native'): native.main()
        assert calls == ['admitted', 'verified', 'model', 'native gate']
    else:
        native.main(); assert calls == ['admitted', 'verified', 'model']
    assert not output.exists()


@pytest.mark.parametrize('key,value', [('navigation_ticks', 3000), ('navigation_ticks', 4000.),
    ('model_state_sha256', 'b'*64), ('queued_controller_changes_adopted', True),
    ('native_scene_workers', 2), ('collection_allowance_bytes', 10*1024**3)])
def test_launch_definition_rejects_undeclared_changes(monkeypatch, key, value):
    launch = native.definition() | dict(source_sha256={}, input_admission={})
    launch[key] = value
    monkeypatch.setattr(native, 'verify_ordered_launch', lambda *a:None)
    monkeypatch.setattr(native.inputs, 'verify_bound', lambda *a:pytest.fail('invalid definition must stop first'))
    with pytest.raises(ValueError, match='exact prospective'): native.verify_inputs(launch)
