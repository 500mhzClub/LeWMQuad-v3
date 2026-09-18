import numpy as np
from lewm.orthonormal_gyro_visual_motion_development import orthonormalize
from lewm.relative_gyro_turn_development import rotation_increment


def test_remove_roundoff_stretch_without_changing_rotation_direction():
    expected = rotation_increment(np.array([.12, -.08, .51]))
    stretched = expected @ np.diag([1.+1e-12, 1.-2e-12, 1.+3e-12])
    result = orthonormalize(stretched)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result.T@result, np.eye(3), rtol=0, atol=2e-15)
    assert np.linalg.det(result) > 0


def test_long_reference_composition_stays_inside_existing_rotation_tolerance():
    gyro = np.eye(3); pose = np.eye(3)
    step = rotation_increment(np.array([.0001, -.0002, .003]))
    for _ in range(10000):
        previous = gyro
        gyro = orthonormalize(gyro @ step)
        pose = pose @ (previous.T @ gyro)
    np.testing.assert_allclose(pose, gyro, rtol=0, atol=1e-10)
    np.testing.assert_allclose(gyro.T@gyro, np.eye(3), rtol=0, atol=2e-15)
