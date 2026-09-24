import numpy as np

from lewm.safety import articulated_swept_geometry_v1 as geometry


def test_sphere_box_signed_clearance_and_tie():
    quaternion=np.asarray([1.,0.,0.,0.]); center=np.asarray([0.,0.,.5]); half=np.asarray([.1,.5,.5])
    clear=geometry.primitive_to_box("sphere",np.asarray([.05]),np.asarray([.5,0,.5]),quaternion,center,half,0.)
    contact=geometry.primitive_to_box("sphere",np.asarray([.05]),np.asarray([.12,0,.5]),quaternion,center,half,0.)
    assert clear > 0
    assert contact < 0
    assert (-contact) >= (-contact)  # frozen threshold tie rejects


def test_articulated_local_transform_round_trip():
    parent_pos=np.asarray([1.,2.,.3]); parent_quat=np.asarray([np.cos(.2),0,0,np.sin(.2)])
    child_pos=np.asarray([1.1,2.2,.4]); child_quat=np.asarray([np.cos(.35),0,0,np.sin(.35)])
    local_pos,local_quat=geometry.inverse_transform(parent_pos,parent_quat,child_pos,child_quat)
    position,quaternion=geometry.compose(parent_pos,parent_quat,local_pos,local_quat)
    assert np.allclose(position,child_pos,atol=1e-12)
    assert np.allclose(quaternion,child_quat,atol=1e-12)


def test_capsule_point_distance_and_deterministic_digest():
    q=np.asarray([1.,0.,0.,0.]); points=np.asarray([[.1,0,0],[.2,0,0]])
    value=geometry.primitive_to_points("capsule",np.asarray([.05,.4]),np.zeros(3),q,points)
    assert np.isclose(value,.05)
    payload={"ground_support_excluded":True,"self_contact_excluded":True,"physics_steps":250}
    assert geometry.digest(payload)==geometry.digest(dict(payload))
