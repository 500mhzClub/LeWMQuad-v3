import numpy as np
import pytest

from lewm.batched_gyro_consensus_development import register
from lewm.full_consensus_early_exit_development import register as original
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_rgbd_correspondence_motion_development import point_fixture


@pytest.mark.parametrize('mode', ['gyro', 'joint'])
@pytest.mark.parametrize('seed', range(12))
def test_same_pose_mask_quality_and_rejection(mode, seed):
    a, b, ua, ub, R, _ = point_fixture()
    rng = np.random.default_rng(seed)
    b = b.copy()
    n = seed % 10
    b[:n] += rng.normal(0, .12, (n, 3))
    results = []
    for function in (original, register):
        try:
            results.append(function(a, b, ua, ub, gyro_rotation=R, mode=mode, frame=seed))
        except SensorContractError as error:
            results.append(str(error))
    if isinstance(results[0], str):
        assert results[0] == results[1]
    else:
        assert not isinstance(results[1], str)
        for x, y in zip(results[0][:3], results[1][:3]):
            np.testing.assert_array_equal(x, y)
        assert results[0][3] == results[1][3]


def test_degenerate_geometry_and_invalid_gyro_still_rejected():
    a, b, ua, ub, R, _ = point_fixture()
    for aa, bb, gyro in ((a*0, b*0, R), (a, b, R*2)):
        for function in (original, register):
            with pytest.raises(SensorContractError):
                function(aa, bb, ua, ub, gyro_rotation=gyro, mode='gyro', frame=1)
