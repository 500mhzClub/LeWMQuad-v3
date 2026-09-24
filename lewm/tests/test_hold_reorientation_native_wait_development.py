import pytest
from scripts import await_go2_hold_reorientation_maze02_native_v1 as wait


def test_each_original_owner_must_end_before_handoff(monkeypatch):
    rounds = iter([True, True, False, True, False, False]); sleeps=[]; events=[]
    monkeypatch.setattr(wait, 'owner_live', lambda owner: next(rounds))
    monkeypatch.setattr(wait, 'completed_identity', lambda root, *a: str(root))
    monkeypatch.setattr(wait, 'verify', lambda sources: None)
    result=wait.wait_for_inputs({},lambda *a,**k:events.append((a,k)),sleep=sleeps.append,clock=lambda:0)
    assert sleeps==[30,30] and len(events)==2
    assert result=={key:str(root) for key,_,root,_,_ in wait.PREREQUISITES}


def test_first_completed_identity_cannot_change_while_waiting(monkeypatch):
    rounds=iter([False,True,False,False]); hashes=iter(['first','changed'])
    monkeypatch.setattr(wait,'owner_live',lambda owner:next(rounds))
    monkeypatch.setattr(wait,'completed_identity',lambda *a:next(hashes))
    with pytest.raises(ValueError,match='identity changed'):
        wait.wait_for_inputs({},lambda *a,**k:None,sleep=lambda n:None,clock=lambda:0)


def test_missing_completion_stops_without_restarting(monkeypatch):
    monkeypatch.setattr(wait,'owner_live',lambda owner:False)
    def missing(*a):raise ValueError('missing complete result')
    monkeypatch.setattr(wait,'completed_identity',missing)
    with pytest.raises(ValueError,match='missing'):
        wait.wait_for_inputs({},lambda *a,**k:None,sleep=lambda n:pytest.fail('no sleep'),clock=lambda:0)


def test_bounded_wait_does_not_launch_replacement(monkeypatch):
    times=iter([0,wait.WAIT_SECONDS])
    monkeypatch.setattr(wait,'owner_live',lambda owner:True)
    with pytest.raises(ValueError,match='expired'):
        wait.wait_for_inputs({},lambda *a,**k:None,sleep=lambda n:pytest.fail('no sleep'),clock=lambda:next(times))


def test_owner_identity_error_is_not_completion(monkeypatch):
    def changed(owner):raise ValueError('identity changed')
    monkeypatch.setattr(wait,'owner_live',changed)
    monkeypatch.setattr(wait,'completed_identity',lambda *a:pytest.fail('cannot admit changed owner'))
    with pytest.raises(ValueError,match='identity changed'):
        wait.wait_for_inputs({},lambda *a,**k:None,clock=lambda:0)
