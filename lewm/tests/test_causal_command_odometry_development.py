import copy
import math

import numpy as np
import pytest

from lewm.causal_command_odometry_development import CausalCommandOdometry
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_relative_gyro_turn_development import initialized,append,packet


def stream(commands,gyro=(0,0,0)):
    buffer=initialized(); rows=[packet(buffer,80)]
    for tick,command in enumerate(commands,1):
        for step in range(80+5*(tick-1)+1,80+5*tick+1):
            # The shared helper generates the regular zero-command clock.
            append(buffer,step,gyro)
        p=packet(buffer,80+5*tick)
        # Supply a consistent synthetic applied-command history in each new
        # packet; all overlaps keep the same previously observed values.
        for j,previous in enumerate(commands[:tick]):
            ns=(85+5*j)*20_000_000; matches=np.flatnonzero(p['sensor_state']['control']['applied_command']['measured_ns']==ns)
            for index in matches: p['sensor_state']['control']['applied_command']['values'][index]=previous
        rows.append(p)
    return rows


def test_interval_ending_command_is_integrated_without_one_tick_lag():
    rows=stream([[.3,0,0],[-.2,0,0],[0,0,0]])
    o=CausalCommandOdometry(); o.begin(rows[0],now_ns=1_600_000_000)
    for i,p in enumerate(rows[1:],1): o.step(p,now_ns=1_600_000_000+i*100_000_000)
    assert o.position==pytest.approx([.01,0,0],abs=1e-9)
    assert o.relative_point([.8,0,0],now_ns=1_900_000_000)==pytest.approx([.79,0,0],abs=1e-9)
    assert o.snapshot(now_ns=1_900_000_000)['metric_translation_qualified'] is False


def test_turning_rotates_translation_and_remaining_goal():
    rows=stream([[.2,0,0]]*10,gyro=(0,0,.5)); o=CausalCommandOdometry()
    o.begin(rows[0],now_ns=1_600_000_000)
    for p in rows[1:]: o.step(p,now_ns=p['image']['measured_ns'])
    assert o.position[0]>.18 and o.position[1]>.04
    assert o.rotation[0,0]==pytest.approx(math.cos(.495),abs=1e-10)
    assert not np.allclose(o.relative_point([.8,0,0],now_ns=2_600_000_000),[.8,0,0])


@pytest.mark.parametrize('fault',['clock_gap','identity','rewritten_command','invalid_command','privilege','stale'])
def test_input_fault_latches_and_cannot_expose_stale_estimate(fault):
    rows=stream([[.2,0,0],[.2,0,0]]); o=CausalCommandOdometry()
    o.begin(rows[0],now_ns=1_600_000_000); p=copy.deepcopy(rows[1]); now=1_700_000_000
    if fault=='clock_gap': p=rows[2]; now=1_800_000_000
    if fault=='identity': p['sensor_state']['identity']=(0,0,1)
    if fault=='rewritten_command': p['sensor_state']['control']['applied_command']['values'][-2,0]=.1
    if fault=='invalid_command': p['sensor_state']['control']['applied_command']['valid'][-1]=False
    if fault=='privilege': p['base_pose_world']=[0]*7
    if fault=='stale': now=1_800_000_000
    with pytest.raises(SensorContractError): o.step(p,now_ns=now)
    assert o.status=='FAILED_SENSOR'
    with pytest.raises(SensorContractError): o.snapshot(now_ns=1_600_000_000)


def test_begin_and_query_require_fresh_clock_and_finite_point():
    p=stream([])[0]; o=CausalCommandOdometry(); o.begin(p,now_ns=1_600_000_000)
    with pytest.raises(SensorContractError): o.begin(p,now_ns=1_600_000_000)
    with pytest.raises(SensorContractError): o.relative_point([1,2,float('nan')],now_ns=1_600_000_000)
    with pytest.raises(SensorContractError): o.snapshot(now_ns=1_700_000_000)
    result=o.snapshot(now_ns=1_600_000_000); result['command_integrated_position_initial_body_m'][0]=99
    assert o.position[0]==0
