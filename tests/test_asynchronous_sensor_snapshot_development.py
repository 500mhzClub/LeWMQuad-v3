import numpy as np
import pytest

from lewm.simulated_body_observation_development import BodyObservationBuffer, SCHEMAS
from lewm.fast_gyro_development import FastGyroBuffer
from scripts.asynchronous_camera_session_development import freeze_sensor_owner


def assert_equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_equal(a[k],b[k])
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    else:assert a==b


@pytest.mark.parametrize('count',[1,6,300])
def test_history_is_exact_and_independent_of_later_sensor_updates(count):
    body=BodyObservationBuffer((0,0,0)); fast=FastGyroBuffer((0,0,0))
    for i in range(count):
        now=i*100_000_000
        for s in SCHEMAS:
            body.buffer.append(s.name,np.full(len(s.channels),i),np.ones(len(s.channels),bool),
                measured_ns=now,available_ns=now,identity=(0,0,0),calibration_id=s.calibration_id)
        fast.append([i,i,i],np.ones(3,bool),measured_ns=now,available_ns=now)
    image=np.zeros((480,640,3),np.uint8)
    expected=body.packet(image,now); expected_fast=fast.packet(now_ns=now)
    frozen=freeze_sensor_owner(body,now); frozen_fast=freeze_sensor_owner(fast,now)
    # Stress both buffer mutation and episode reset after submission.
    for owner in (body,fast):
        for rows in owner.buffer._samples.values():rows[-1][2][:]=-99
        owner.buffer.begin_episode((0,0,1))
    assert_equal(frozen.packet(image,now),expected)
    assert_equal(frozen_fast.packet(now_ns=now),expected_fast)


def test_future_availability_cannot_be_silently_removed():
    fast=FastGyroBuffer((0,0,0))
    fast.append([1,2,3],np.ones(3,bool),measured_ns=0,available_ns=10)
    with pytest.raises(ValueError,match='future'):
        freeze_sensor_owner(fast,0)
