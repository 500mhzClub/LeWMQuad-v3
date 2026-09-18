import pytest
from scripts import await_go2_all_phase_residual_maze02_native_v1 as mod


def test_waits_for_both_original_owners_without_restarting(monkeypatch):
    step=[0];reads=[];events=[]
    def live(owner):
        return step[0]<(2 if owner==mod.inputs.NATIVE_OWNER else 1)
    monkeypatch.setattr(mod.inputs.corrected_wait,'owner_live',live)
    monkeypatch.setattr(mod,'verify',lambda _:None)
    def completed(root,*args):
        reads.append((step[0],root));return 'native sha' if root==mod.inputs.native.OUTPUT else 'correction sha'
    monkeypatch.setattr(mod,'completed_identity',completed)
    result=mod.wait_for_inputs({},lambda *a,**k:events.append((a,k)),
        sleep=lambda _:step.__setitem__(0,step[0]+1),clock=lambda:float(step[0]))
    assert result==dict(native_result_sha256='native sha',correction_wait_result_sha256='correction sha')
    assert len(events)==2
    assert all(step>=2 for step,root in reads if root==mod.inputs.native.OUTPUT)


def test_observation_wait_expiry_never_restarts_live_owners(monkeypatch):
    monkeypatch.setattr(mod.inputs.corrected_wait,'owner_live',lambda _:True)
    monkeypatch.setattr(mod,'completed_identity',lambda *a:pytest.fail('read live owner result'))
    ticks=iter((0.,float(mod.WAIT_SECONDS)))
    with pytest.raises(ValueError,match='expired'):
        mod.wait_for_inputs({},lambda *a,**k:None,sleep=lambda _:pytest.fail('slept after expiry'),clock=lambda:next(ticks))


def test_missing_terminal_stops_even_if_other_owner_is_live(monkeypatch):
    monkeypatch.setattr(mod.inputs.corrected_wait,'owner_live',lambda owner:owner==mod.inputs.NATIVE_OWNER)
    monkeypatch.setattr(mod,'completed_identity',lambda *a:(_ for _ in ()).throw(ValueError('missing result')))
    with pytest.raises(ValueError,match='missing'):
        mod.wait_for_inputs({},lambda *a,**k:None,sleep=lambda _:pytest.fail('ignored missing terminal'))


@pytest.mark.parametrize('failure',[True,False])
def test_failed_or_absent_original_result_never_admitted(tmp_path,failure):
    if failure:(tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='failed|without complete'):
        mod.completed_identity(tmp_path,'launch','status',{})
