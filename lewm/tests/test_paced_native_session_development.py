from types import SimpleNamespace
import numpy as np
import pytest

from lewm_genesis.lewm_contract import SafetyLimits,apply_safety_limits_batch
from scripts.paced_native_session_development import command_policy_step
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop


def test_policy_service_retains_100ms_slew_and_accepts_midwindow_zero():
    limits=SafetyLimits(-.3,.3,0.,0.,.5,.25,0.,.35)
    rows=[];steps=[];clocks=[]
    runner=SimpleNamespace(_policy_dt_ns=20_000_000,_physics_steps_per_policy=10,
        _sim_time_ns=1_500_000_000,_last_executed=np.zeros((1,3),np.float32),safety=limits,
        _build_observation=lambda x:x,_apply_joint_targets=lambda x:None)
    session=SimpleNamespace(ctx=SimpleNamespace(runner=runner,
        policy=SimpleNamespace(act=lambda x:x),
        build=SimpleNamespace(scene=SimpleNamespace(step=lambda:steps.append(1)))),
        _sample=lambda requested,applied,stamp:rows.append((applied.copy(),stamp)),
        physics_clock_callback=clocks.append)
    expected,_=apply_safety_limits_batch(np.array([[[.2,0.,.45]]],np.float32),
        np.zeros((1,3),np.float32),limits)
    for _ in range(3):
        np.testing.assert_array_equal(command_policy_step(session,[.2,0.,.45]),expected[0,0])
    # A new request is actually serviced at 60ms, not deferred to 100ms.
    assert command_policy_step(session,[0.,0.,0.])==[0.,0.,0.]
    assert command_policy_step(session,[0.,0.,0.])==[0.,0.,0.]
    assert runner._sim_time_ns==1_600_000_000 and len(steps)==len(rows)==50
    assert clocks==list(range(1_502_000_000,1_600_000_001,2_000_000))
    assert all(np.array_equal(r[0],expected[0,0]) for r in rows[:30])
    assert all(np.array_equal(r[0],[0.,0.,0.]) for r in rows[30:])


def test_physical_stop_keeps_actual_partial_tick_time_and_command():
    clocks=[];steps=[]
    runner=SimpleNamespace(_policy_dt_ns=20_000_000,_physics_steps_per_policy=10,
        _sim_time_ns=1_500_000_000,_last_executed=np.zeros((1,3),np.float32),
        safety=SafetyLimits(-.3,.3,0.,0.,.5,.25,0.,.35),
        _build_observation=lambda x:x,_apply_joint_targets=lambda x:None)
    def sample(requested,applied,stamp):
        if len(steps)==6:raise PhysicalStop('speed stop')
    session=SimpleNamespace(ctx=SimpleNamespace(runner=runner,
        policy=SimpleNamespace(act=lambda x:x),
        build=SimpleNamespace(scene=SimpleNamespace(step=lambda:steps.append(1)))),
        _sample=sample,physics_clock_callback=clocks.append)
    with pytest.raises(PhysicalStop,match='speed stop'):
        command_policy_step(session,[.2,0.,0.])
    assert len(steps)==6
    assert runner._sim_time_ns==1_512_000_000
    assert clocks==list(range(1_502_000_000,1_510_000_001,2_000_000))
    np.testing.assert_array_equal(runner._last_executed,np.array([[.2,0.,0.]],np.float32))
