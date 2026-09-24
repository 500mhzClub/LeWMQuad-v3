import numpy as np
import pytest
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference
from lewm.rgbd_correspondence_motion_development import project
from lewm.joint_camera_registration_development import register_views
from lewm.causal_sensor_state import SensorContractError


def test_joint_camera_known_motion_and_wrong_view_pixels():
    # Each camera alone has fewer than twelve pairs. Their fixed extrinsics
    # differ; the known physical motion must survive both reprojection models.
    optical = np.array([[-.3,-.2,1.2], [.3,-.2,1.3], [-.3,.2,1.4],
        [.3,.2,1.5], [0.,-.1,1.6], [.1,.1,1.7]])
    E = np.asarray(BODY_FROM_OPTICAL)
    b = optical@E[:3,:3].T+E[:3,3]
    A, offset = body_from_reference()
    theta = .02
    R = np.array([[np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    t = np.array([.015, -.005, .002])
    a = b@R.T+t
    aux_body = b@A.T+offset
    aux_a = (aux_body@R.T+t-offset)@A
    primary = (a, b, project(a)[0], project(b)[0])
    auxiliary = (aux_a, b, project(aux_a)[0], project(b)[0])
    fitted_R, fitted_t, mask, receipt, _ = register_views(primary, auxiliary, gyro_rotation=R, frame=71)
    np.testing.assert_allclose(fitted_R, R, atol=1e-12)
    np.testing.assert_allclose(fitted_t, t, atol=1e-12)
    assert mask.all() and receipt['camera_inliers'] == [6, 6]
    corrupted = (*auxiliary[:3], auxiliary[3]+50.)
    with pytest.raises(SensorContractError):
        register_views(primary, corrupted, gyro_rotation=R, frame=71)
