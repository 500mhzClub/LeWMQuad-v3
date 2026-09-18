"""Known camera geometry tests for the experimental gyro-conditioned refit."""
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.gyro_conditioned_pair_pose_development import refit
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.rgbd_correspondence_motion_development import project
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference
from lewm.joint_camera_registration_development import project_views


def example(camera):
    rng = np.random.default_rng(103)
    optical = rng.uniform([-.25, -.2, 1.], [.25, .2, 1.5], (40, 3))
    T = np.asarray(BODY_FROM_OPTICAL)
    b = optical@T[:3, :3].T+T[:3, 3]
    A, offset = body_from_reference()
    if camera == 'auxiliary': b = b@A.T+offset
    if camera == 'joint': b = np.concatenate((b[:20], b[20:]@A.T+offset))
    theta = .025
    R = np.array([[np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    t = np.array([.012, -.003, .001]); a = b@R.T+t
    projection = (lambda x: project_views(x, 20)) if camera == 'joint' else (
        (lambda x: project((x-offset)@A)) if camera == 'auxiliary' else project)
    ua, va = projection(a); ub, vb = projection(b)
    assert va.all() and vb.all()
    ref = SimpleNamespace(position=np.zeros(3), rotation=np.eye(3))
    reg = dict(reference_inlier_points_body_m=a.tolist(), current_inlier_points_body_m=b.tolist(),
        reference_inlier_pixels=ua.tolist(), current_inlier_pixels=ub.tolist())
    if camera == 'joint': reg.update(calibrated_body_frame_fit=True, camera_inliers=[20, 20])
    return dict(reference=ref, registration=reg, local_R=R, R=R, t=t, p=t), R, t


@pytest.mark.parametrize('camera', ['primary', 'auxiliary', 'joint'])
def test_known_motion_recovers_body_translation_for_each_camera(camera):
    candidate, R, t = example(camera)
    result = refit(candidate, R, camera=camera, last_p=np.zeros(3), last_R=np.eye(3))
    np.testing.assert_allclose(result['p'], t, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result['R'], R, rtol=0, atol=1e-12)
    assert result['registration']['mode'] == 'gyro_rgbd_refit'
    assert result['registration']['gyro_conditioned_refinement']['all_accepted_image_points_retained']


def test_incorrect_gyro_is_rejected_by_image_support():
    candidate, _, _ = example('primary')
    theta = .3
    wrong = np.array([[np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    with pytest.raises(ValueError, match='loses an accepted RGB-D correspondence'):
        refit(candidate, wrong, camera='primary', last_p=np.zeros(3), last_R=np.eye(3))
