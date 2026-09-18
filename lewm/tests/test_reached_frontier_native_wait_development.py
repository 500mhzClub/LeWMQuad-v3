import pytest
from scripts import await_go2_reached_frontier_maze03_native_v1 as mod


def test_waits_for_both_original_processes_without_reading_live_results(monkeypatch):
    seconds = [0]; events = []; reads = []
    monkeypatch.setattr(mod, 'owner_live', lambda owner: seconds[0] < (30 if owner == mod.PREFIX_OWNER else 60))
    def identity(root, *args):
        if root == mod.inputs.replay.OUTPUT:
            assert seconds[0] >= 30; answer = 'prefix'
        else:
            assert seconds[0] >= 60; answer = 'batch'
        reads.append((seconds[0], answer)); return answer
    monkeypatch.setattr(mod, 'completed_identity', identity)
    monkeypatch.setattr(mod, 'verify', lambda sources: None)
    def sleep(n): seconds[0] += n
    result = mod.wait_for_inputs({}, lambda *a, **k: events.append(k), sleep=sleep, clock=lambda: seconds[0])
    assert result == dict(prefix_result_sha256='prefix', adapter_batch_result_sha256='batch')
    assert len(events) == 2 and reads == [(30, 'prefix'), (60, 'prefix'), (60, 'batch')]


def test_ended_owner_without_complete_original_result_is_terminal(tmp_path):
    with pytest.raises(ValueError, match='ended without a complete result'):
        mod.completed_identity(tmp_path, 'sha', 'status', {})


def test_original_failure_blocks_new_native_child(tmp_path):
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError, match='original prerequisite failed'):
        mod.completed_identity(tmp_path, 'sha', 'status', {})


def test_live_original_is_not_restarted_when_wait_bound_expires(monkeypatch):
    seconds = [0]
    monkeypatch.setattr(mod, 'WAIT_SECONDS', 30)
    monkeypatch.setattr(mod, 'owner_live', lambda owner: True)
    monkeypatch.setattr(mod, 'completed_identity', lambda *a: pytest.fail('read a live original result'))
    def sleep(n): seconds[0] += n
    with pytest.raises(ValueError, match='wait expired'):
        mod.wait_for_inputs({}, lambda *a, **k: None, sleep=sleep, clock=lambda: seconds[0])


def test_changed_process_identity_is_not_reinterpreted_as_completion(monkeypatch):
    def identity(owner): raise ValueError('process identity changed')
    monkeypatch.setattr(mod, 'owner_live', identity)
    monkeypatch.setattr(mod, 'completed_identity', lambda *a: pytest.fail('used a replacement process result'))
    with pytest.raises(ValueError, match='process identity changed'):
        mod.wait_for_inputs({}, lambda *a, **k: None)


def test_first_completed_result_identity_stays_pinned_while_other_job_runs(monkeypatch):
    seconds = [0]
    monkeypatch.setattr(mod, 'owner_live', lambda owner: owner == mod.BATCH_OWNER)
    monkeypatch.setattr(mod, 'completed_identity', lambda *a: str(seconds[0]))
    def sleep(n): seconds[0] += n
    with pytest.raises(ValueError, match='completed original prerequisite identity changed'):
        mod.wait_for_inputs({}, lambda *a, **k: None, sleep=sleep, clock=lambda: seconds[0])
