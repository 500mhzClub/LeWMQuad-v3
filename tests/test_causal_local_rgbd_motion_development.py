import numpy as np
from lewm.causal_local_rgbd_motion_development import local_history_features


def test_constant_forward_and_missing_pair():
    pair=dict(rotation=np.eye(3).tolist(),translation=[.02,0.,0.])
    values=np.asarray(local_history_features([pair]*3)).reshape(3,4)
    np.testing.assert_allclose(values[:,0],[-.06,-.04,-.02],atol=1e-12)
    np.testing.assert_allclose(values[:,1:],0.,atol=1e-12)
    assert local_history_features([pair,None,pair]) is None
    assert local_history_features([pair,pair]) is None


def test_coordinates_are_current_body_not_initial_body():
    quarter=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    move=dict(rotation=np.eye(3).tolist(),translation=[1.,0.,0.])
    turn=dict(rotation=quarter.tolist(),translation=[0.,0.,0.])
    values=np.asarray(local_history_features([move,turn,move])).reshape(3,4)
    np.testing.assert_allclose(values[:,:2],[[-1.,1.],[-1.,0.],[-1.,0.]],atol=1e-12)
    np.testing.assert_allclose(values[:,2:], [[-1.,-1.],[-1.,-1.],[0.,0.]],atol=1e-12)
