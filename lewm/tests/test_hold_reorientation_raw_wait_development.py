import pytest
from scripts import await_go2_hold_reorientation_raw_prefix_v1 as wait


def test_waits_for_same_worker_then_returns_only_completed_identity(monkeypatch):
    active = iter([True, True, False]); seen = []; sleeps = []
    def live(owner):
        seen.append(owner)
        return next(active) if owner is wait.WORKER else True
    monkeypatch.setattr(wait, 'owner_live', live)
    monkeypatch.setattr(wait, 'completed_worker_identity', lambda: 'complete-sha')
    events = []
    assert wait.wait_for_worker(lambda *a, **k: events.append((a,k)), sleep=sleeps.append,
        clock=lambda: 0) == 'complete-sha'
    assert sleeps == [30, 30] and len(events) == 2
    assert seen == [wait.WORKER, wait.BATCH_OWNER, wait.WORKER, wait.BATCH_OWNER, wait.WORKER]


def test_missing_terminal_after_owner_ends_is_failure_not_restart(monkeypatch):
    monkeypatch.setattr(wait, 'owner_live', lambda owner: False)
    def incomplete(): raise ValueError('no terminal')
    monkeypatch.setattr(wait, 'completed_worker_identity', incomplete)
    with pytest.raises(ValueError, match='no terminal'):
        wait.wait_for_worker(lambda *a, **k: None, sleep=lambda n: pytest.fail('no sleep'), clock=lambda: 0)


def test_parent_loss_stops_handoff(monkeypatch):
    monkeypatch.setattr(wait, 'owner_live', lambda owner: owner is wait.WORKER)
    with pytest.raises(ValueError, match='parent'):
        wait.wait_for_worker(lambda *a, **k: None, sleep=lambda n: pytest.fail('no sleep'), clock=lambda: 0)


def test_timeout_never_restarts_original_or_launches_child(monkeypatch):
    monkeypatch.setattr(wait, 'owner_live', lambda owner: True)
    times = iter([0, wait.WAIT_SECONDS])
    with pytest.raises(ValueError, match='expired'):
        wait.wait_for_worker(lambda *a, **k: None, sleep=lambda n: pytest.fail('no sleep'), clock=lambda: next(times))


def test_changed_owner_identity_is_not_treated_as_completion(monkeypatch):
    def changed(owner): raise ValueError('identity changed')
    monkeypatch.setattr(wait, 'owner_live', changed)
    monkeypatch.setattr(wait, 'completed_worker_identity', lambda: pytest.fail('cannot admit replacement'))
    with pytest.raises(ValueError, match='identity changed'):
        wait.wait_for_worker(lambda *a, **k: None, clock=lambda: 0)
