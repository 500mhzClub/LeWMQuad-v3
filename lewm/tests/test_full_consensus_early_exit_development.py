import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.full_consensus_early_exit_development import register
from lewm.joint_rgbd_rigid_pose_development import register as original
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_rgbd_correspondence_motion_development import point_fixture


@pytest.mark.parametrize('mode', ['joint', 'gyro'])
@pytest.mark.parametrize('outliers', [False, True])
def test_identical_pose_mask_and_quality_with_or_without_full_consensus(mode, outliers):
    a, b, ua, ub, R, _ = point_fixture()
    if outliers: b = b.copy(); b[:3] += [.1, .2, .3]
    expected = original(a, b, ua, ub, gyro_rotation=R, mode=mode, frame=1)
    actual = register(a, b, ua, ub, gyro_rotation=R, mode=mode, frame=1)
    for x, y in zip(expected[:3], actual[:3], strict=True): np.testing.assert_array_equal(x, y)
    old_count = expected[3].pop('valid_proposals'); new_count = actual[3].pop('valid_proposals')
    assert expected[3] == actual[3]
    if outliers: assert new_count == old_count
    else: assert new_count == 1 and old_count > 1


def test_full_consensus_does_not_skip_final_gyro_disagreement_check():
    a, b, ua, ub, R, _ = point_fixture()
    for function in (original, register):
        with pytest.raises(SensorContractError, match='disagree'):
            function(a, b, ua, ub, gyro_rotation=R@rotation_increment([0, 0, .2]), mode='joint', frame=1)
