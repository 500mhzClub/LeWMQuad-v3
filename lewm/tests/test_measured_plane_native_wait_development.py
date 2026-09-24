"""Preserve the live owner and do not infer completion from elapsed time."""
import pytest
from scripts import await_go2_measured_plane_maze02_native_v1 as waiter


def test_waits_for_exact_owner_then_admits_actual_completed_result(monkeypatch):
    states=iter([True,True,False]);events=[];sleeps=[];admitted=[]
    monkeypatch.setattr(waiter.run,'owner_live',lambda owner:next(states))
    monkeypatch.setattr(waiter.run,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(waiter.run,'digest',lambda p:'actual-completed-result')
    monkeypatch.setattr(waiter.inputs,'admit_queue',lambda sha,sources:admitted.append(sha))
    monkeypatch.setattr(waiter.run,'verify',lambda sources:None)
    sha=waiter.wait_for_queue({},lambda status,**kw:events.append(status),sleep=sleeps.append)
    assert sha == 'actual-completed-result' and admitted == [sha]
    assert len(events)==2 and sleeps==[30,30]


def test_ended_owner_without_valid_completion_is_not_restarted(monkeypatch):
    monkeypatch.setattr(waiter.run,'owner_live',lambda owner:False)
    monkeypatch.setattr(waiter.run,'digest',lambda p:'unverified-result')
    def fail(*a): raise ValueError('original queue failed')
    monkeypatch.setattr(waiter.inputs,'admit_queue',fail)
    with pytest.raises(ValueError,match='original queue failed'):
        waiter.wait_for_queue({},lambda *a,**kw:None,sleep=lambda _:pytest.fail('restarted wait'))


def test_owner_observation_failure_is_not_completion(monkeypatch):
    def fail(*a): raise RuntimeError('transient observation failure')
    monkeypatch.setattr(waiter.run,'owner_live',fail)
    monkeypatch.setattr(waiter.run,'digest',lambda _:pytest.fail('read result after observation failure'))
    with pytest.raises(RuntimeError,match='observation'):
        waiter.wait_for_queue({},lambda *a,**kw:None)
