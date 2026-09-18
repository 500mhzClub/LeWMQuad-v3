"""Whole-worker failures, including peaks after collection and final writes."""
import json

import pytest

from scripts import extended_return_budget_worker_resources_development as worker


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.limits, 'validate_root', lambda root: root)
    state = dict(monotonic_s=0., rss_bytes=worker.limits.GIB,
        peak_rss_bytes=worker.limits.GIB, memory_available_bytes=70*worker.limits.GIB,
        artifact_free_bytes=100*worker.limits.GIB)
    def snapshot(root):
        assert root == tmp_path
        state['monotonic_s'] += 1.
        return dict(state)
    monkeypatch.setattr(worker, 'snapshot', snapshot)
    return state, worker.WorkerEnvelope(tmp_path, 'synthetic')


def test_completed_worker_includes_final_write_and_reconstructs_receipt(tmp_path, monkeypatch):
    state, monitor = fixture(tmp_path, monkeypatch)
    for stage in worker.STAGES:
        monitor.check(stage)
        state['artifact_free_bytes'] -= worker.limits.GIB
    receipt = monitor.finish()
    assert worker.check(tmp_path, 'synthetic') == receipt
    assert receipt['worker_completed'] and receipt['terminal_record_write_observed']
    assert receipt['maximum_reported_peak_rss_bytes'] == worker.limits.GIB
    assert not receipt['parent_result_serialization_included']
    assert not receipt['between_sample_availability_bounded']
    with pytest.raises(ValueError): worker.WorkerEnvelope(tmp_path, 'synthetic')
    with pytest.raises(ValueError): monitor.finish()


@pytest.mark.parametrize('stage', ['audit_written', 'prefix_accounted', 'terminal_written'])
def test_peak_breach_is_retained_when_current_rss_has_already_recovered(tmp_path, monkeypatch, stage):
    state, monitor = fixture(tmp_path, monkeypatch)
    for earlier in worker.STAGES[:worker.STAGES.index(stage)]: monitor.check(earlier)
    state['peak_rss_bytes'] = 48*worker.limits.GIB + 1
    with pytest.raises(worker.limits.ResourceLimitError): monitor.check(stage)
    state['peak_rss_bytes'] = worker.limits.GIB
    with pytest.raises(worker.limits.ResourceLimitError, match='latched'): monitor.check(stage)
    receipt = monitor.finish()
    assert not receipt['worker_completed'] and receipt['error']
    assert receipt['maximum_sampled_rss_bytes'] == worker.limits.GIB
    assert receipt['maximum_reported_peak_rss_bytes'] > 48*worker.limits.GIB
    with pytest.raises(worker.limits.ResourceLimitError): worker.check(tmp_path, 'synthetic')


@pytest.mark.parametrize('field,value', [
    ('artifact_free_bytes', 56*worker.limits.GIB-1),
    ('memory_available_bytes', 16*worker.limits.GIB-1),
    ('rss_bytes', 48*worker.limits.GIB+1)])
def test_whole_worker_cumulative_disk_and_runtime_limits(tmp_path, monkeypatch, field, value):
    state, monitor = fixture(tmp_path, monkeypatch)
    monitor.check('start'); state[field] = value
    with pytest.raises(worker.limits.ResourceLimitError): monitor.check('inputs_verified')
    assert not monitor.finish()['worker_completed']


@pytest.mark.parametrize('field,value', [
    ('artifact_free_bytes', 84*worker.limits.GIB-1),
    ('memory_available_bytes', 64*worker.limits.GIB-1)])
def test_initial_headroom_is_required_before_work(tmp_path, monkeypatch, field, value):
    state, monitor = fixture(tmp_path, monkeypatch); state[field] = value
    with pytest.raises(worker.limits.ResourceLimitError): monitor.check('start')
    assert not monitor.finish()['worker_completed']


@pytest.mark.parametrize('fault', ['missing_terminal', 'order', 'clock', 'receipt', 'external_failure'])
def test_incomplete_or_altered_whole_worker_cannot_be_accepted(tmp_path, monkeypatch, fault):
    _, monitor = fixture(tmp_path, monkeypatch)
    for stage in worker.STAGES[:-1] if fault == 'missing_terminal' else worker.STAGES:
        monitor.check(stage)
    monitor.finish(ValueError('prefix failed') if fault == 'external_failure' else None)
    stream = tmp_path/monitor.stream_name
    rows = [json.loads(line) for line in stream.read_text().splitlines()]
    if fault == 'order': rows[5]['stage'] = 'collection_persisted'
    if fault == 'clock': rows[5]['monotonic_s'] = 0.
    if fault in ('order', 'clock'): stream.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    if fault == 'receipt':
        path = tmp_path/monitor.result_name
        receipt = json.loads(path.read_text()); receipt['maximum_reported_peak_rss_bytes'] += 1
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): worker.check(tmp_path, 'synthetic')


def test_wrong_runtime_stage_latches_and_preserves_failure(tmp_path, monkeypatch):
    _, monitor = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='ordered'): monitor.check('terminal_written')
    assert not monitor.finish()['worker_completed']
