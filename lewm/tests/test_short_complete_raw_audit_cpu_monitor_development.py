"""Negative evidence retention and exact-report checks, with synthetic scopes."""
from copy import deepcopy
import json

import pytest
import torch
from scripts import replay_go2_short_complete_raw_audit_cpu_monitor_v1 as probe


def episode():
    collection = dict(layout_index=2, schedule_terminal='SENSOR_OR_MODEL_FAILURE',
        command_ticks=13, completed_ticks=13, physics_samples=1400, rgbd_frames=14,
        auxiliary_frames=14, decisions=14, terminal_zero_ticks=10)
    expected = dict(verified_round_trip=False, selected_actions={},
        observation_and_control_wall_ms=[1.]*14, raw_model_command_replay_pass=True,
        private_physics={'contact':False})
    return collection, expected


@pytest.mark.parametrize('key,value', [('decisions',13),('physics_samples',1399),
    ('command_ticks',14),('schedule_terminal','BUDGET_EXHAUSTED'),('layout_index',True)])
def test_incomplete_or_substituted_episode_rejected(key, value):
    collection, expected = episode(); collection[key] = value
    with pytest.raises(ValueError): probe.require_episode(collection, expected)


@pytest.mark.parametrize('key,value', [('verified_round_trip',True),('selected_actions',{'forward':1}),
    ('observation_and_control_wall_ms',[1.]*13),('raw_model_command_replay_pass',False)])
def test_changed_reference_scope_rejected(key, value):
    collection, expected = episode(); expected[key] = value
    with pytest.raises(ValueError): probe.require_episode(collection, expected)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'OUTPUT', tmp_path)
    collection, expected = episode()
    def install(mode='exact'):
        def audit(launch, actual):
            assert actual == collection
            if mode == 'device': torch.empty(1, device='meta')
            if mode == 'error': raise RuntimeError('synthetic original audit failure')
            torch.ones(1) + 1
            report = deepcopy(expected)
            if mode == 'changed': report['private_physics']['contact'] = True
            return report
        monkeypatch.setattr(probe, 'run_original', audit)
    install()
    return collection, expected, install


def test_complete_report_exact_and_scope_remains_bounded(synthetic):
    collection, expected, _ = synthetic
    result = probe.monitored_audit({}, collection, expected)
    assert result['complete_original_audit_report_exact'] and result['original_failed_navigation_preserved']
    assert not result['independent_multiarm_raw_audit_qualified'] and not result['overlap_execution_permitted']
    assert json.loads((probe.OUTPUT/'audit.json').read_text()) == expected


@pytest.mark.parametrize('mode', ['changed','device','error'])
def test_failure_preserves_monitor_and_does_not_pass(synthetic, mode):
    collection, expected, install = synthetic; install(mode)
    with pytest.raises((ValueError, RuntimeError)): probe.monitored_audit({}, collection, expected)
    measured = json.loads((probe.OUTPUT/'cpu_monitor.json').read_text())
    if mode == 'changed':
        assert measured['scope_returned_without_error']
        assert json.loads((probe.OUTPUT/'audit.json').read_text()) != expected
    else: assert not measured['scope_returned_without_error']
    if mode == 'device': assert measured['violations']


def test_direct_original_auditor_and_exact_arguments(monkeypatch):
    model = object(); geometry = object(); calls = []
    monkeypatch.setattr(probe.original, 'assigned_model', lambda launch,case:model if case == probe.CASE else None)
    monkeypatch.setattr(probe.original, 'ArticulatedCollisionGeometry', lambda urdf:geometry)
    monkeypatch.setattr(probe.original, 'audit', lambda *args,**kwargs:calls.append((args,kwargs)) or {'returned':True})
    collection, _ = episode(); launch = {'source_sha256':{probe.original.PROTOCOL:'a'*64}}
    assert probe.run_original(launch, collection) == {'returned':True}
    assert calls == [((2, collection, 'a'*64), dict(input_root=probe.original.OUTPUT, model=model,
        robot_geometry=geometry, episode_name=probe.CASE[0], condition='jepa', variant='full'))]
