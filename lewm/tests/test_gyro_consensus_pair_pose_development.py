"""Robust image support must survive gyro refitting, including pooled views."""
import numpy as np
import pytest
from lewm.tests.test_gyro_conditioned_pair_pose_development import example
from lewm.gyro_consensus_pair_pose_development import consensus_refit


@pytest.mark.parametrize('camera', ['primary', 'auxiliary', 'joint'])
def test_prunes_bad_correspondence_and_recovers_motion(camera):
    candidate, R, t = example(camera)
    reg = candidate['registration']; reg.update(lifted_matches=40, inliers=40)
    reg['reference_inlier_points_body_m'][0][0] += .08
    result = consensus_refit(candidate, R, camera=camera, last_p=np.zeros(3), last_R=np.eye(3))
    np.testing.assert_allclose(result['p'], t, rtol=0, atol=1e-12)
    receipt = result['registration']['gyro_conditioned_refinement']
    assert receipt['rejected_original_indices'] == [0]
    assert result['registration']['inliers'] == 39
    assert receipt['strict_majority_of_original_matches']


def test_pruning_cannot_destroy_original_strict_majority():
    candidate, R, _ = example('primary')
    candidate['registration'].update(lifted_matches=78, inliers=40)
    candidate['registration']['reference_inlier_points_body_m'][0][0] += .08
    with pytest.raises(ValueError, match='strict majority'):
        consensus_refit(candidate, R, camera='primary', last_p=np.zeros(3), last_R=np.eye(3))
