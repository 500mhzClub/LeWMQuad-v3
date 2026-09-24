import numpy as np
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical,reference_pose,project_square
from lewm.auxiliary_tilted_depth_geometry_development import project_square as prior_projection
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL


def test_fixed_downward_optical_axis_and_same_mount():
    E=body_from_optical()
    np.testing.assert_allclose(E[:3,2],[np.sqrt(.5),0.,-np.sqrt(.5)],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(E[:3,3],[.35,0.,.08])
    np.testing.assert_allclose(E[:3,:3].T@E[:3,:3],np.eye(3),rtol=0,atol=1e-15)


def test_reference_adapter_preserves_full_optical_pose():
    T=np.asarray(BODY_FROM_OPTICAL);E=body_from_optical()
    R=np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
    p=np.array([.3,-.8,.04]);Q,q=reference_pose(R,p)
    for x in (np.array([0.,0.,1.]),np.array([.2,-.3,2.])):
        np.testing.assert_allclose(p+R@(E[:3,:3]@x+E[:3,3]),q+Q@(T[:3,:3]@x+T[:3,3]),rtol=0,atol=1e-15)


def test_steeper_camera_covers_nearer_floor_without_claiming_pixels_or_occlusion():
    args=([.52,0.],-.34,np.eye(3),np.zeros(3))
    assert not prior_projection(*args)['entire_square_in_frustum']
    new=project_square(*args)
    assert new['entire_square_in_frustum']
    assert not new['occlusion_checked'] and not new['measured_floor_coverage'] and not new['hardware_mount_validated']
    assert not project_square([0.,0.],-.34,np.eye(3),np.zeros(3))['entire_square_in_frustum']
