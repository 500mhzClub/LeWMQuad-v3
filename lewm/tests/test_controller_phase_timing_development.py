from types import SimpleNamespace
import pytest
import torch
from lewm.controller_phase_timing_development import PhaseTiming, PhaseProxy, model_forward_timing


def test_nested_durations_partition_root_without_double_counting():
    clock=iter([0,1,5,9]);t=PhaseTiming(clock=lambda:next(clock))
    with t.scope('root'):
        with t.scope('child'):pass
    s=t.snapshot()
    assert s['root']==dict(calls=1,inclusive_ns=9,exclusive_ns=5)
    assert s['child']==dict(calls=1,inclusive_ns=4,exclusive_ns=4)
    assert sum(v['exclusive_ns'] for v in s.values())==9
    s['root']['calls']=99;assert t.snapshot()['root']['calls']==1


def test_failure_unwinds_scopes_and_active_reset_rejected():
    t=PhaseTiming()
    with pytest.raises(RuntimeError):
        with t.scope('outer'):
            with pytest.raises(ValueError):t.reset()
            with pytest.raises(ValueError):t.snapshot()
            with t.scope('inner'):raise RuntimeError('original failure')
    assert not t.stack and set(t.snapshot())=={'outer','inner'}
    t.reset();assert t.snapshot()=={}


def test_proxy_preserves_return_identity_side_effects_and_exception():
    t=PhaseTiming();value=[]
    class Original:
        def __init__(self):self.count=0
        def call(self,x,*,flag):
            self.count+=1;value.append(x)
            if flag:raise RuntimeError('original exception')
            return value
    original=Original();proxy=PhaseProxy(original,t,{'call':'target.call'})
    assert proxy.call(3,flag=False) is value and proxy.count==1
    proxy.count=7;assert original.count==7
    with pytest.raises(RuntimeError,match='original exception'):proxy.call(4,flag=True)
    assert value==[3,4] and t.snapshot()['target.call']['calls']==2 and not t.stack


@pytest.mark.parametrize('fails',[False,True])
def test_model_hooks_preserve_outputs_and_cleanup_after_success_or_failure(fails):
    class Model(torch.nn.Module):
        def forward(self,x):
            if fails:raise RuntimeError('model error')
            return x
    model=Model();x=torch.ones(2);t=PhaseTiming()
    if fails:
        with pytest.raises(RuntimeError,match='model error'):
            with model_forward_timing(model,t):model(x)
    else:
        with model_forward_timing(model,t):assert model(x) is x
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert t.snapshot()['model.forward']['calls']==1 and not t.stack


def test_phase_controller_preserves_complete_public_packet_decisions(monkeypatch):
    from copy import deepcopy
    from functools import partial
    from lewm.controller_phase_timing_development import PhaseTimedLaterFloorController
    from lewm.later_floor_resolution_controller_development import LaterFloorResolutionRoundTripController
    from lewm.tests.test_frame_floor_cache_development import equal
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    monkeypatch.setattr(fixture,'visual',partial(visual,origin=1_500_000_000))
    kwargs=dict(public_mission=dict(goal_initial_body_xy_m=[1.,0.],return_initial_body_xy_m=[0.,0.],
        require_return_after_goal=True),navigation_ticks=100,condition='jepa',variant='full',persistent=True)
    t=PhaseTiming();old=LaterFloorResolutionRoundTripController(None,None,**kwargs)
    new=PhaseTimedLaterFloorController(None,None,timing=t,**kwargs);previous=None
    for frame in range(2):
        p,d,a,raw,now=packets(frame,previous)
        old.motion=SimpleNamespace(observe=lambda *args,**kw:deepcopy(raw))
        new.motion=PhaseProxy(SimpleNamespace(observe=lambda *args,**kw:deepcopy(raw)),t,{'observe':'motion.observe'})
        t.reset();x=old.observe(p,d,None,now_ns=now,auxiliary_depth=a)
        y=new.observe(p,d,None,now_ns=now,auxiliary_depth=a)
        assert y['terminal'] is None,y.get('failure');equal(x,y)
        values=t.snapshot()
        assert values['map.observe']['calls']==values['motion.observe']['calls']==1
        assert sum(v['exclusive_ns'] for v in values.values())==values['controller.observe']['inclusive_ns']
        previous=raw
