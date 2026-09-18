from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_gyro_conditioned_pair_pose_development import example
from lewm.gyro_consensus_pair_pose_development import consensus_refit
from lewm.deferred_registration_copy_development import deferred_consensus_refit


@pytest.mark.parametrize('camera', ['primary', 'auxiliary', 'joint'])
def test_same_pruned_pose_and_fully_independent_output(camera):
    candidate, R, _ = example(camera)
    original = candidate['registration']
    original.update(lifted_matches=40, inliers=40, extra_evidence={'nested': [{'value': 7}]})
    original['reference_inlier_points_body_m'][0][0] += .08
    before = deepcopy(original)
    kw = dict(camera=camera, last_p=np.zeros(3), last_R=np.eye(3))
    expected = consensus_refit(candidate, R, **kw)
    actual = deferred_consensus_refit(candidate, R, **kw)
    for key in ('p', 'R', 't', 'local_R'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['registration'] == expected['registration']
    assert original == before
    actual['registration']['extra_evidence']['nested'][0]['value'] = 99
    actual['registration']['reference_inlier_points_body_m'][0][0] = 99
    assert original == before
    assert expected['registration']['extra_evidence']['nested'][0]['value'] == 7


def test_failed_consensus_preserves_input_and_original_rejection():
    candidate, R, _ = example('primary')
    candidate['registration'].update(lifted_matches=78, inliers=40)
    candidate['registration']['reference_inlier_points_body_m'][0][0] += .08
    before = deepcopy(candidate['registration'])
    for fit in (consensus_refit, deferred_consensus_refit):
        with pytest.raises(ValueError, match='strict majority'):
            fit(candidate, R, camera='primary', last_p=np.zeros(3), last_R=np.eye(3))
        assert candidate['registration'] == before
