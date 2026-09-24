import numpy as np
from lewm.commanded_planar_motion_development import forecast
from lewm.geometry_progress_pilot_development import ACTIONS


def test_forward_pulse_and_settling_tail():
    i=ACTIONS.index('forward');prefix=np.zeros((3,3))
    full=forecast(prefix,pulse=False);short=forecast(prefix,pulse=True)
    np.testing.assert_allclose(full[i,6:,:2],[[.08,0],[.08,0]],atol=1e-14)
    np.testing.assert_allclose(short[i,3:,:2],np.tile([.02,0],(5,1)),atol=1e-14)


def test_exact_arc_and_shared_rotating_prefix():
    prefix=np.tile([0.,0.,.45],(3,1));p=forecast(prefix,pulse=False)
    np.testing.assert_allclose(p[:,:3],np.repeat(p[:1,:3],6,axis=0),atol=1e-14)
    i=ACTIONS.index('left_arc');angle=.45*.4;r=.16/.45
    local=np.array([r*np.sin(angle),r*(1-np.cos(angle))]);yaw=.45*.3
    R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    np.testing.assert_allclose(p[i,6,:2],R@local,atol=1e-14)
    np.testing.assert_allclose(p[i,6,2:],[np.sin(yaw+angle),np.cos(yaw+angle)],atol=1e-14)
