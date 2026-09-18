"""Actual worker orchestration with synthetic collection and failure injection."""
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import run_go2_extended_return_budget_maze02_v1 as job


def setup(tmp_path, monkeypatch, fault=None):
    name = job.CASE[0]; events = []; verifications = []; snapshots = []
    monkeypatch.setattr(job, 'OUTPUT', tmp_path)
    monkeypatch.setattr(job.worker_resources.limits, 'validate_root', lambda root: root)
    def snapshot(root):
        index = len(snapshots); snapshots.append(index)
        return dict(monotonic_s=float(index), rss_bytes=1024**3,
            peak_rss_bytes=(49 if fault == 'late_peak' and index == 9 else 1)*1024**3,
            memory_available_bytes=70*1024**3, artifact_free_bytes=100*1024**3)
    monkeypatch.setattr(job.worker_resources, 'snapshot', snapshot)
    monkeypatch.setattr(job.run, 'digest', lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(job.run, 'read_json', lambda root, name: json.loads((root/name).read_text()))
    def verify(root, ids):
        assert all(job.run.digest(root/leaf) == sha for leaf, sha in ids.items())
        verifications.append(dict(ids))
        if fault == 'final_artifacts' and len(verifications) == 3: raise ValueError('closed artifact changed')
    monkeypatch.setattr(job.run, 'verify_artifacts', verify)
    def inputs(launch):
        events.append('inputs')
        assert launch['source_sha256'][job.PROTOCOL] == 'a'*64
        if fault == 'reauthentication' and events.count('inputs') == 2: raise ValueError('predecessor changed')
    monkeypatch.setattr(job, 'verify_inputs', inputs)
    monkeypatch.setattr(job, 'assigned_model', lambda: torch.nn.Linear(1, 1))
    monkeypatch.setattr(job, 'state_digest', lambda state: job.inputs.replay.pair.MODEL_SHA)
    monkeypatch.setattr(job, 'geometry_factory', lambda urdf: 'synthetic_geometry')
    def collect(index, protocol, **kwargs):
        events.append('collect'); assert kwargs['output'] == tmp_path and protocol == 'a'*64
        directory = tmp_path/name; directory.mkdir()
        np.savez(directory/'physics_trace.npz', physics_contact=np.zeros(3, np.uint8))
        for leaf in job.pipeline.resources.names(name, 'collection'): (tmp_path/leaf).write_text('synthetic resource evidence')
        if fault == 'collection': raise ValueError('collection failed after persisted evidence')
        return dict(navigation_ticks=8000, decisions=3, command_ticks=2, completed_ticks=2, physics_samples=850)
    monkeypatch.setattr(job.pipeline, 'collect', collect)
    monkeypatch.setattr(job.pipeline, 'artifacts', lambda index, collection: ('physics_trace.npz',))
    audit = dict(verified_round_trip=False, native_evaluation={'native_round_trip_candidate_pass': False},
        strict_physical_visibility_pass=True, hard_measurement_failed_frames=[], renderer_capture_audit={})
    def audit_run(index, collection, protocol, **kwargs):
        events.append('audit'); assert kwargs['input_root'] == tmp_path
        for leaf in job.pipeline.resources.names(name, 'audit'): (tmp_path/leaf).write_text('synthetic resource evidence')
        if fault == 'audit': raise ValueError('raw audit failed')
        return dict(audit)
    monkeypatch.setattr(job.pipeline, 'audit', audit_run)
    def readout(actual_audit, collection, contacts):
        assert actual_audit == audit and np.array_equal(contacts, np.zeros(3, np.uint8))
        return {'synthetic_readout': True}
    monkeypatch.setattr(job.original, 'case_readout', readout)
    monkeypatch.setattr(job.results.resource_audit, 'check', lambda *args: {'synthetic_resources': True})
    receipt = dict(actual_paired_execution_compared=False, full_raw_audit_retained=True)
    def prefix(collection, admission, **kwargs):
        events.append('prefix')
        assert len(kwargs['current_bindings']) == 8
        if fault == 'prefix': raise ValueError('actual prefix failed')
        return dict(receipt)
    monkeypatch.setattr(job.results, 'prefix_result', prefix)
    def require(record, actual_audit, admission, **kwargs):
        events.append('require_worker')
        assert record['status'] == job.WORKER_STATUS and kwargs['prefix_receipt'] == receipt
        assert actual_audit == audit and len(record['artifact_sha256']) == 9
        if fault == 'worker_validation': raise ValueError('worker rejected')
    monkeypatch.setattr(job.results, 'require_worker', require)
    original_write = job.bounded_json
    def write(path, value):
        if fault == 'terminal_write' and path.name.endswith('_worker_terminal.json'):
            raise OSError('terminal write failed')
        original_write(path, value)
    monkeypatch.setattr(job, 'bounded_json', write)
    original_write(tmp_path/'launch.json', {'source_sha256': {job.PROTOCOL: 'a'*64}, 'input_admission': {}})
    return events, verifications, job.run.digest(tmp_path/'launch.json')


def test_worker_closes_and_rehashes_evidence_through_terminal_write(tmp_path, monkeypatch):
    events, verifications, sha = setup(tmp_path, monkeypatch)
    record = job.worker(sha)
    assert record['status'] == job.WORKER_STATUS and record['verified_round_trip'] is False
    assert events == ['inputs', 'collect', 'audit', 'prefix', 'require_worker', 'inputs']
    assert len(verifications) == 3 and len(verifications[-1]) == 9
    assert json.loads((tmp_path/(job.CASE[0]+'_worker_terminal.json')).read_text()) == record
    assert record['worker_log_sha256'] == job.run.digest(tmp_path/(job.CASE[0]+'_worker.log'))
    assert job.worker_resources.check(tmp_path, job.CASE[0])['worker_completed']


@pytest.mark.parametrize('fault', ['collection', 'audit', 'prefix', 'worker_validation',
    'reauthentication', 'final_artifacts'])
def test_worker_failure_preserves_partial_evidence_and_never_completes(tmp_path, monkeypatch, fault):
    events, _, sha = setup(tmp_path, monkeypatch, fault)
    record = job.worker(sha)
    assert record['status'] == job.WORKER_FAILURE and record['failure']
    assert (tmp_path/job.CASE[0]/'physics_trace.npz').is_file()
    assert json.loads((tmp_path/(job.CASE[0]+'_worker_terminal.json')).read_text()) == record
    receipt = json.loads((tmp_path/job.worker_resources.names(job.CASE[0])[1]).read_text())
    assert not receipt['worker_completed'] and receipt['error']
    assert not receipt['terminal_record_write_observed']
    with pytest.raises(ValueError): job.worker_resources.check(tmp_path, job.CASE[0])
    if fault == 'collection': assert 'audit' not in events
    if fault == 'audit': assert 'prefix' not in events


@pytest.mark.parametrize('fault', ['late_peak', 'terminal_write'])
def test_late_failure_raises_to_parent_and_cannot_be_hidden_by_worker_status(tmp_path, monkeypatch, fault):
    _, _, sha = setup(tmp_path, monkeypatch, fault)
    with pytest.raises((job.pipeline.resources.ResourceLimitError, OSError)):
        job.worker(sha)
    receipt = json.loads((tmp_path/job.worker_resources.names(job.CASE[0])[1]).read_text())
    assert not receipt['worker_completed'] and receipt['error']
    terminal = tmp_path/(job.CASE[0]+'_worker_terminal.json')
    if fault == 'late_peak':
        assert json.loads(terminal.read_text())['status'] == job.WORKER_STATUS
        with pytest.raises(job.pipeline.resources.ResourceLimitError): job.worker_resources.check(tmp_path, job.CASE[0])
    else: assert not terminal.exists()
