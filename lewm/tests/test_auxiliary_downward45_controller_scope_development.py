import numpy as np
import pytest
from lewm import auxiliary_downward45_depth_observation_development as sensor
from lewm import auxiliary_tilted_depth_observation_development as prior_sensor
from lewm.auxiliary_downward45_floor_map_development import AuxiliaryDownward45FloorMap,AuxiliaryDownward45SurfaceMemory
from lewm.auxiliary_depth_floor_map_development import AuxiliaryDepthFloorMap,AuxiliaryDepthSurfaceMemory
from lewm.auxiliary_downward45_goal_probe_development import AuxiliaryDownward45GoalProbe
from lewm.auxiliary_depth_reobserve_goal_probe_development import AuxiliaryDepthReobserveGoalProbe
from lewm.tests.test_causal_depth_observation_development import frame


def test_calibrations_cannot_be_interchanged():
    p,_,now=frame();native=np.ones((480,640),np.float32)
    for source,target in ((sensor,prior_sensor),(prior_sensor,sensor)):
        d=source.from_native_depth(native,p,measured_ns=now,available_ns=now,now_ns=now)
        with pytest.raises(ValueError,match='calibration'):target.validate_depth(d,p,now_ns=now)
    assert sensor.CALIBRATION_ID!=prior_sensor.CALIBRATION_ID
    assert sensor.calibration_metadata()['body_from_optical']!=prior_sensor.calibration_metadata()['body_from_optical']


def test_collision_checks_and_controller_execution_remain_inherited():
    assert AuxiliaryDownward45SurfaceMemory.footprint is AuxiliaryDepthSurfaceMemory.footprint
    assert AuxiliaryDownward45FloorMap.observe is AuxiliaryDepthFloorMap.observe
    assert AuxiliaryDownward45GoalProbe.advance is AuxiliaryDepthReobserveGoalProbe.advance
    assert AuxiliaryDownward45GoalProbe.observe is AuxiliaryDepthReobserveGoalProbe.observe
    a=AuxiliaryDownward45GoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    b=AuxiliaryDepthReobserveGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    assert type(a.selector) is type(b.selector) and type(a.motion) is type(b.motion)
    assert type(a.mapper) is AuxiliaryDownward45FloorMap and a.memory is a.mapper.surface
    assert a.observe({}, {}, {},now_ns=1,auxiliary_depth={})['terminal']=='SENSOR_OR_MODEL_FAILURE'
