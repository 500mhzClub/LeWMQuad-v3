import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference,gyro_in_reference,pose_in_body
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL


def test_reference_lift_maps_to_actual_auxiliary_mount_without_native_pose():
    optical=np.array([[.1,.2,.5],[-.3,.4,2.],[.4,-.1,4.]])
    T=np.asarray(BODY_FROM_OPTICAL);E=body_from_optical();R,t=body_from_reference()
    reference=optical@T[:3,:3].T+T[:3,3]
    np.testing.assert_allclose(reference@R.T+t,optical@E[:3,:3].T+E[:3,3],atol=1e-14,rtol=0)


def test_rigid_pose_and_gyro_conjugation_include_the_camera_lever_arm():
    A,b=body_from_reference();body_R=Rotation.from_rotvec([.07,-.13,.19]).as_matrix()
    body_t=np.array([.08,-.02,.03]);Q=gyro_in_reference(body_R)
    ref_t=A.T@(body_t-b+body_R@b)
    R,t=pose_in_body(Q,ref_t)
    np.testing.assert_allclose(R,body_R,atol=1e-14,rtol=0)
    np.testing.assert_allclose(t,body_t,atol=1e-14,rtol=0)
    points=np.array([[.2,-.1,.5],[.7,.2,.3]])
    by_reference=(points@Q.T+ref_t)@A.T+b
    by_body=(points@A.T+b)@R.T+t
    np.testing.assert_allclose(by_reference,by_body,atol=1e-14,rtol=0)
    assert np.linalg.norm(A@ref_t-body_t)>1e-4


@pytest.mark.parametrize('translation',[[1.,2.],[float('nan'),0.,0.]])
def test_invalid_reference_translation_is_rejected(translation):
    with pytest.raises(ValueError):pose_in_body(np.eye(3),translation)


def test_nonrotation_gyro_input_is_rejected():
    with pytest.raises(ValueError):gyro_in_reference(np.eye(3)*2)
