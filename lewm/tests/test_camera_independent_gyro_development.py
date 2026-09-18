import numpy as np
import pytest

from lewm.camera_independent_gyro_development import CameraIndependentGyro, RETAINED_SNAPSHOTS
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.tests.test_fast_gyro_development import buffers, advance, packet


def test_gapped_camera_queries_preserve_every_measured_rotation_interval():
    slow, fast = buffers()
    candidate = CameraIndependentGyro(); original = FastRelativeOrientation()
    start = 1_600_000_000
    p, f = packet(slow, 80), fast.packet(now_ns=start)
    assert candidate.observe(p, f, now_ns=start) == original.begin(p, f, now_ns=start)
    for tick in range(1, 16):
        # Changing rotation axes makes skipped integration observable.
        p, f, now = advance(slow, fast, tick, (.02*tick, -.03*(tick%3), .04))
        expected = original.step(p, f, now_ns=now)
        assert candidate.observe(p, f, now_ns=now) == expected
        if tick in (1, 4, 9, 15):
            assert candidate.for_camera(measured_ns=now) == expected
            assert expected['samples_integrated'] == 50*tick
    np.testing.assert_array_equal(candidate.integrator.rotation, original.rotation)


def test_missing_gyro_packet_still_latches_failure():
    slow, fast = buffers(); candidate = CameraIndependentGyro(); start = 1_600_000_000
    candidate.observe(packet(slow, 80), fast.packet(now_ns=start), now_ns=start)
    advance(slow, fast, 1, (0., 0., .1))
    p, f, now = advance(slow, fast, 2, (0., 0., .1))
    with pytest.raises(ValueError): candidate.observe(p, f, now_ns=now)
    assert candidate.integrator.status == 'FAILED_SENSOR'
    with pytest.raises(ValueError): candidate.for_camera(measured_ns=start)
    with pytest.raises(ValueError): candidate.observe(p, f, now_ns=now)


def test_retention_has_exact_timestamps_and_independent_return_values():
    slow, fast = buffers(); candidate = CameraIndependentGyro(); start = 1_600_000_000
    value = candidate.observe(packet(slow, 80), fast.packet(now_ns=start), now_ns=start)
    value['rotation_initial_body_from_current_body'][0][0] = 999
    assert candidate.for_camera(measured_ns=start)['rotation_initial_body_from_current_body'][0][0] == 1.
    value = candidate.for_camera(measured_ns=start)
    value['samples_integrated'] = 999
    assert candidate.for_camera(measured_ns=start)['samples_integrated'] == 0
    for tick in range(1, RETAINED_SNAPSHOTS+1):
        p, f, now = advance(slow, fast, tick, (0., .1, 0.))
        candidate.observe(p, f, now_ns=now)
    assert len(candidate.snapshots) == RETAINED_SNAPSHOTS
    for stamp in (start, now+100_000_000, now-1):
        with pytest.raises(ValueError): candidate.for_camera(measured_ns=stamp)
    assert candidate.for_camera(measured_ns=now-100_000_000)['decision_ns'] == now-100_000_000
