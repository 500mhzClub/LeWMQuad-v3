import numpy as np
import pytest
from lewm.conditioned_support_tracker_development import register
from lewm.full_consensus_early_exit_development import register as original
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL,FOCAL
from lewm.causal_sensor_state import SensorContractError


def cloud(pixels):
    optical=np.column_stack((2*(pixels[:,0]+.5-320)/FOCAL,2*(pixels[:,1]+.5-240)/FOCAL,np.full(len(pixels),2.)))
    T=np.asarray(BODY_FROM_OPTICAL)
    return optical@T[:3,:3].T+T[:3,3]


def test_conditioned_patch_does_not_require_crossing_image_bins():
    pixels=np.array([(x,y) for x in [340.,355.,370.,385.] for y in [180.,200.,220.,240.]])
    points=cloud(pixels)
    kwargs=dict(gyro_rotation=np.eye(3),mode='joint',frame=1)
    with pytest.raises(SensorContractError,match='grid support'):
        original(points,points,pixels,pixels,**kwargs)
    R,t,mask,receipt=register(points,points,pixels,pixels,**kwargs)
    assert mask.all()
    assert np.allclose(R,np.eye(3)) and np.allclose(t,0.)
    assert receipt['current_grid_cells']==1
    assert receipt['original_measured_3d_conditioning_required']
    # A line of observations cannot constrain the same rigid pose.
    pixels[:,1]=200.
    points=cloud(pixels)
    with pytest.raises(SensorContractError,match='conditioned'):
        register(points,points,pixels,pixels,**kwargs)
