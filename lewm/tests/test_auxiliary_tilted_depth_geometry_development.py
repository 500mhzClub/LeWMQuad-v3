import math
import numpy as np
from lewm.auxiliary_tilted_depth_geometry_development import body_from_optical,reference_pose,project_square
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL


def test_auxiliary_optical_pose_adapter_preserves_full_extrinsics():
    E=body_from_optical();T=np.asarray(BODY_FROM_OPTICAL)
    assert np.allclose(E[:3,:3].T@E[:3,:3],np.eye(3),atol=1e-15)
    assert E[2,2]<0 and np.linalg.det(E[:3,:3])>0
    for yaw in (-2.,0.,1.3):
        c,s=math.cos(yaw),math.sin(yaw);R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        p=np.array([.3,-.8,.04]);Q,q=reference_pose(R,p)
        for point in (np.array([0.,0.,1.]),np.array([.2,-.3,2.])):
            direct=p+R@(E[:3,:3]@point+E[:3,3])
            adapted=q+Q@(T[:3,:3]@point+T[:3,3])
            assert np.allclose(direct,adapted,atol=1e-15,rtol=0)


def test_floor_projection_distinguishes_near_blind_region_without_coverage_claim():
    visible=project_square([.8,0.],-.34,np.eye(3),np.zeros(3))
    unseen=project_square([0.,0.],-.34,np.eye(3),np.zeros(3))
    assert visible['entire_square_in_frustum'] and not unseen['entire_square_in_frustum']
    assert not visible['measured_floor_coverage'] and not visible['occlusion_checked']
    assert not visible['hardware_mount_validated'] and not visible['navigation_qualified']
