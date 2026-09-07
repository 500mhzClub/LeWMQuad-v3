import copy
import math

import numpy as np
import pytest

from lewm.causal_relative_orientation_development import CausalRelativeOrientation
from lewm.causal_sensor_state import SensorContractError
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_relative_gyro_turn_development import initialized,append,packet


def test_twenty_second_nonplanar_integration_has_no_turn_deadline_or_double_counting():
    rate=np.array([.1,.2,.3]); buffer=initialized(rate); tracker=CausalRelativeOrientation()
    tracker.begin(packet(buffer,80),now_ns=1_600_000_000)
    for tick in range(1,201):
        for step in range(80+(tick-1)*5+1,80+tick*5+1): append(buffer,step,rate)
        result=tracker.step(packet(buffer,80+tick*5),now_ns=1_600_000_000+tick*100_000_000)
    assert result['samples_integrated']==1000
    assert np.asarray(result['rotation_initial_body_from_current_body'])==pytest.approx(rotation_increment(rate*20),abs=1e-12)


def test_initial_direction_is_rotated_into_current_body_not_rotated_with_it():
    buffer=initialized([0,0,math.pi/2]); tracker=CausalRelativeOrientation()
    tracker.begin(packet(buffer,80),now_ns=1_600_000_000)
    for tick in range(1,11):
        for step in range(80+(tick-1)*5+1,80+tick*5+1): append(buffer,step,[0,0,math.pi/2])
        tracker.step(packet(buffer,80+tick*5),now_ns=1_600_000_000+tick*100_000_000)
    assert tracker.transport_xy([0,.8],now_ns=2_600_000_000)==pytest.approx([.8,0],abs=1e-12)
    assert tracker.transport_xy([.8,0],now_ns=2_600_000_000)==pytest.approx([0,-.8],abs=1e-12)
    with pytest.raises(SensorContractError): tracker.transport_xy([0,.8],now_ns=2_700_000_000)


@pytest.mark.parametrize('fault',['gap','identity','invalid','rewrite_old','rewrite_boundary','privilege'])
def test_fault_is_transactional_and_latched(fault):
    buffer=initialized(); tracker=CausalRelativeOrientation(); tracker.begin(packet(buffer,80),now_ns=1_600_000_000)
    for step in range(81,86): append(buffer,step,[0,0,.2])
    value=packet(buffer,85); ns=1_700_000_000
    if fault=='gap': ns+=100_000_000
    if fault=='identity': value['sensor_state']['identity']=(0,0,1)
    if fault=='invalid':
        value['sensor_state']['sensed']['gyro']['valid'][-2]=False; value['sensor_state']['sensed']['gyro']['values'][-2]=0
    if fault=='rewrite_old': value['sensor_state']['sensed']['gyro']['values'][0,0]=1
    if fault=='rewrite_boundary': value['sensor_state']['sensed']['gyro']['values'][-6,0]=1
    if fault=='privilege': value['world_orientation']=np.eye(3)
    with pytest.raises(SensorContractError): tracker.step(value,now_ns=ns)
    assert tracker.status=='FAILED_SENSOR' and tracker.samples_integrated==0
    with pytest.raises(SensorContractError): tracker.snapshot(now_ns=1_600_000_000)
    with pytest.raises(SensorContractError): tracker.step(packet(buffer,85),now_ns=1_700_000_000)


def test_snapshot_and_input_mutation_cannot_rewrite_integrated_state():
    buffer=initialized(); value=packet(buffer,80); tracker=CausalRelativeOrientation()
    snapshot=tracker.begin(value,now_ns=1_600_000_000)
    snapshot['rotation_initial_body_from_current_body'][0][0]=99
    value['sensor_state']['sensed']['gyro']['values'][:]=100
    for step in range(81,86): append(buffer,step,[0,0,0])
    result=tracker.step(packet(buffer,85),now_ns=1_700_000_000)
    assert np.asarray(result['rotation_initial_body_from_current_body'])==pytest.approx(np.eye(3))
    with pytest.raises(SensorContractError): tracker.begin(packet(buffer,85),now_ns=1_700_000_000)
