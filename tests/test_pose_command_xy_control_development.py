import numpy as np
from lewm.pose_command_xy_control_development import choose_xy


def test_control_is_invariant_to_neural_xy_and_preserves_yaw_contact():
    learned=np.arange(6*8*5,dtype=float).reshape(6,8,5)/1000
    xy=np.full((6,8,2),.04)
    original=learned.copy();perturbed=learned.copy();perturbed[:,:,:2]+=100
    a=choose_xy(learned,xy,'pose_command');b=choose_xy(perturbed,xy,'pose_command')
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[:,:,2:],learned[:,:,2:])
    np.testing.assert_array_equal(learned,original)
    np.testing.assert_array_equal(choose_xy(learned,xy,'learned'),learned)
