import numpy as np
from lewm.course_aware_visual_servo_scene_development import specification,pack


def test_new_course_trial_uses_changed_native_spawn_and_declared_pair():
    a,b=[specification(c) for c in ('nominal','lower_friction')]
    pa,pb=pack(a),pack(b)
    assert pa.robot==pb.robot and pa.robot.spawn_xyz_m==(-.55,-.20,.375)
    np.testing.assert_allclose(pa.robot.spawn_quat_wxyz,[np.cos(-.04),0,0,np.sin(-.04)],rtol=0,atol=1e-15)
    assert a['geometry']==b['geometry'] and a['procedural_seed']==2026090649
    assert a['appearance_seed']==2026090651
    assert pa.physics_randomization.floor_friction_mu==1. and pb.physics_randomization.floor_friction_mu==.15
