"""Fixed auxiliary-camera geometry candidate; no rendered or physical evidence."""
import math
import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL,FOCAL
from lewm.joint_rgbd_rigid_pose_development import proper

CALIBRATION_ID='auxiliary_downward45_depth_geometry_candidate_v1'
MOUNT_BODY_M=(.35,0.,.08)
DOWN_PITCH_RAD=math.pi/4


def body_from_optical():
    c,s=math.cos(DOWN_PITCH_RAD),math.sin(DOWN_PITCH_RAD)
    pitch=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    E=np.eye(4);E[:3,:3]=pitch@np.asarray(BODY_FROM_OPTICAL)[:3,:3]
    E[:3,3]=MOUNT_BODY_M
    return E


def reference_pose(rotation_map_from_body,position_map):
    """Represent the auxiliary optical pose in the existing depth helper frame.

    This is a coordinate adapter, never a second estimate of the robot pose.
    It preserves the optical-to-map transform exactly in real arithmetic.
    """
    R=proper(rotation_map_from_body);p=np.asarray(position_map,float)
    if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('finite observed body position required')
    E=body_from_optical();T=np.asarray(BODY_FROM_OPTICAL)
    reference_R=R@E[:3,:3]@T[:3,:3].T
    reference_p=p+R@E[:3,3]-reference_R@T[:3,3]
    return reference_R,reference_p


def project_square(centre_map_xy,floor_height,rotation_map_from_body,position_map,*,radius=.022):
    xy=np.asarray(centre_map_xy,float);p=np.asarray(position_map,float);R=proper(rotation_map_from_body)
    if (xy.shape!=(2,) or p.shape!=(3,) or not np.isfinite([*xy,*p,floor_height]).all()
            or not np.isfinite(radius) or not 0<radius<=.1):raise ValueError('finite nominal floor square required')
    corners=xy+radius*np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    world=np.column_stack((corners,np.full(4,floor_height)));E=body_from_optical()
    optical=((world-p)@R-E[:3,3])@E[:3,:3];z=optical[:,2]
    uv=optical[:,:2]/np.maximum(z[:,None],1e-12)*FOCAL+[319.5,239.5]
    lo=uv.min(0);hi=uv.max(0);margin=1e-9+64*np.finfo(float).eps*np.maximum(np.abs(lo),np.abs(hi))
    lo-=margin;hi+=margin
    return dict(entire_square_in_frustum=bool((z>=.2).all() and (z<=5.).all()
        and (lo>=0).all() and (hi<[639,479]).all()),optical_depth_range_m=[float(z.min()),float(z.max())],
        pixel_lower_xy=lo.tolist(),pixel_upper_xy=hi.tolist(),calibration_id=CALIBRATION_ID,
        hypothetical_geometry_only=True,occlusion_checked=False,measured_floor_coverage=False,
        hardware_mount_validated=False,navigation_qualified=False)

