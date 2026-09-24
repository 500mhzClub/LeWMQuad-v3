from copy import deepcopy
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_causal_depth_observation_development import frame
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth,validate_depth,body_points,FOCAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical


def packet():
    policy,_,now=frame();native=np.full((480,640),2.,np.float32)
    return policy,from_native_depth(native,policy,measured_ns=now,available_ns=now,now_ns=now),now


def test_auxiliary_mount_pixel_centres_unknowns_and_copies():
    policy,d,now=packet();original=deepcopy(policy)
    cloud=body_points(d,policy,now_ns=now);T=body_from_optical()
    expected=T[:3,:3]@np.array([2*(2.5-320)/FOCAL,2*(2.5-240)/FOCAL,2])+T[:3,3]
    np.testing.assert_allclose(cloud['points_body_m'][0,0],expected,atol=1e-12)
    native=np.full((480,640),2.,np.float32);native[2,2:8]=[np.nan,np.inf,-.1,0.,.19,200.]
    d=from_native_depth(native,policy,measured_ns=now,available_ns=now,now_ns=now);native[:]=3.
    assert not d['valid'][2,2:8].any() and np.all(d['depth_m'][2,2:8]==0.) and d['depth_m'][3,3]==2.
    assert np.isnan(body_points(d,policy,now_ns=now)['points_body_m'][0,0]).all()
    np.testing.assert_array_equal(policy['image']['rgb'],original['image']['rgb'])


@pytest.mark.parametrize('fault',('world_pose','segmentation','future','stale','early','episode','rgb','calibration','hardware','mask','depth'))
def test_auxiliary_rejects_privilege_calibration_clock_identity_and_mask_faults(fault):
    p,d,now=packet()
    if fault in ('world_pose','segmentation'):d[fault]=[0.,0.,0.]
    elif fault=='future':d['available_ns']=now+1
    elif fault=='stale':d['measured_ns']-=100_000_000
    elif fault=='early':d['available_ns']-=1
    elif fault=='episode':d['identity']=(0,0,1)
    elif fault=='rgb':d['primary_rgb_sha256']='0'*64
    elif fault=='calibration':d['calibration_id']=p['image']['calibration_id']
    elif fault=='hardware':d['hardware_calibrated']=True
    elif fault=='mask':d['valid'][0,0]=False
    elif fault=='depth':d['depth_m'][0,0]=np.nan
    with pytest.raises(SensorContractError):validate_depth(d,p,now_ns=now)

