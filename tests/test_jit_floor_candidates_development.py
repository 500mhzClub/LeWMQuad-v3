import math
import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.sampled_plane_candidates_development import measured_candidates as original
from lewm.jit_floor_candidates_development import measured_candidates as compiled


def plane_depth(height, angle):
    u=(np.arange(640)+.5-320)/FOCAL
    w=(np.arange(480)+.5-240)/FOCAL
    normal=np.array([math.sin(angle),0.,math.cos(angle)])
    # Optical rays transformed by the fixed primary adapter.
    denominator=normal[0]-normal[2]*w[:,None]+np.zeros((480,640))
    with np.errstate(divide='ignore',invalid='ignore'):
        depth=-(height+normal@np.asarray(BODY_FROM_OPTICAL)[:3,3])/denominator
    valid=np.isfinite(depth)&(depth>=.2)&(depth<=5.)
    return np.where(valid,depth,0.),valid


def test_height_and_normal_boundaries_match_original_candidates():
    for height,angle in [(h,a) for h in (.15,.3) for a in (0.,math.acos(.97)-1e-14,
            math.acos(.97),math.acos(.97)+1e-14)]:
        depth,valid=plane_depth(height,angle)
        a=original(depth,valid,BODY_FROM_OPTICAL,np.array([0.,0.,1.]))
        b=compiled(depth,valid,BODY_FROM_OPTICAL,np.array([0.,0.,1.]))
        assert all(x.shape==y.shape and x.tobytes()==y.tobytes() for x,y in zip(a,b))


def test_missing_neighbour_pixels_preserve_candidate_exclusion():
    depth,valid=plane_depth(.3,0.)
    valid[201::17,101::19]=False;depth[~valid]=0.
    a=original(depth,valid,BODY_FROM_OPTICAL,np.array([0.,0.,1.]))
    b=compiled(depth,valid,BODY_FROM_OPTICAL,np.array([0.,0.,1.]))
    assert all(x.shape==y.shape and x.tobytes()==y.tobytes() for x,y in zip(a,b))
