import pytest
from lewm import auxiliary_depth_reobserve_goal_probe_development as module


def controller():
    return module.AuxiliaryDepthReobserveGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)


def feed(monkeypatch,c,*,terminal=module.NO_FEASIBLE,action=None,failure=None):
    def advance(self,*args,**kwargs):
        if self.terminal is not None:return self._result([0.,0.,0.],None,None)
        self.tick+=1;self.terminal=terminal;self.failure=failure
        return self._result([0.,0.,0.] if action is None else [.2,0.,0.],
            dict(prediction=[],action=action,view_budget_exhausted=terminal=='VIEW_BUDGET_EXHAUSTED'),1.)
    monkeypatch.setattr(module.AuxiliaryDepthGoalProbe,'advance',advance)
    return c.advance({},None,now_ns=1)


def test_ten_zero_waits_then_latched_failure(monkeypatch):
    c=controller()
    for i in range(10):
        r=feed(monkeypatch,c)
        assert r['terminal'] is None and r['requested_command']==[0.,0.,0.]
        assert r['infeasible_wait_active'] and r['consecutive_infeasible_observations']==i+1
    r=feed(monkeypatch,c)
    assert r['terminal']==module.NO_FEASIBLE and not r['infeasible_wait_active']
    r=feed(monkeypatch,c,terminal=None,action='forward')
    assert r['terminal']==module.NO_FEASIBLE and r['requested_command']==[0.,0.,0.]


def test_feasible_action_recovery_resets_wait_counter(monkeypatch):
    c=controller();feed(monkeypatch,c)
    r=feed(monkeypatch,c,terminal=None,action='forward')
    assert r['requested_command']==[.2,0.,0.] and r['terminal'] is None
    assert r['consecutive_infeasible_observations']==0 and r['feasible_action_recoveries']==1
    assert feed(monkeypatch,c)['consecutive_infeasible_observations']==1


@pytest.mark.parametrize('terminal',['SENSOR_OR_MODEL_FAILURE','VIEW_BUDGET_EXHAUSTED',
    'MISSION_TICK_BUDGET_EXHAUSTED','OBSERVED_GOAL_CANDIDATE'])
def test_other_terminal_contracts_remain_latched(monkeypatch,terminal):
    c=controller();r=feed(monkeypatch,c,terminal=terminal)
    assert r['terminal']==terminal and not r['infeasible_wait_active']
    assert feed(monkeypatch,c,terminal=None,action='forward')['terminal']==terminal
