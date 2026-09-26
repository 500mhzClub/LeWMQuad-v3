from types import SimpleNamespace
import numpy as np
import pytest
from lewm.navigation_capability_target_reference_development import settled_task_cues, cue_world_xy, install_task_cues
from lewm.stop_conditioned_settling_development import StopConditionedSettlingMission
from lewm.physical_execution_development import rotation_xyzw


def test_both_targets_share_fixed_world_points_and_install_once():
    episode=dict(home_se2_world=[0.,2.6,-2.9709853500113037],beacon_xy_world=[3.,4.])
    p=np.array([.002908332971855998,2.5850837230682373,.31929126381874084])
    R=rotation_xyzw([.007542120758444071,.01722867786884308,-.9959359765052795,.08807943016290665])
    cues=settled_task_cues(episode,p,R)
    assert set(cues)=={'goal_initial_body_xy_m','return_initial_body_xy_m','require_return_after_goal'}
    assert np.linalg.norm(cues['return_initial_body_xy_m'])>.01
    for key,point in [('goal_initial_body_xy_m',[3.,4.]),('return_initial_body_xy_m',[0.,2.6])]:
        np.testing.assert_allclose(cue_world_xy(cues[key],p,R),point,atol=1e-9,rtol=0)
    old=StopConditionedSettlingMission(cues|dict(return_initial_body_xy_m=[0.,0.]),navigation_ticks=4800,arrival_radius_m=.02)
    controller=SimpleNamespace(mission=old,mission_rows=[],goal=np.array(cues['goal_initial_body_xy_m']))
    install_task_cues(controller,cues)
    assert type(controller.mission) is type(old)
    np.testing.assert_array_equal(controller.mission.home,cues['return_initial_body_xy_m'])
    assert controller.mission.navigation_ticks==4800
    with pytest.raises(ValueError,match='one-time'):install_task_cues(controller,cues)
