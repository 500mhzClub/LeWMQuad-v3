from dataclasses import replace
import numpy as np
import pytest
from lewm.fine_obstacle_round_trip_development import connector,dispatch_request,FRAME,FineObstacleRoundTripRuntime,_FineDispatch
from lewm.stopping_margin_dispatch_development import _StoppingDispatch
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.tests.test_fresh_obstacle_dispatch_development import observation
from lewm.tests.test_stopping_margin_dispatch_development import plan


def test_finer_observed_cells_remove_quantization_veto_without_shrinking_footprint():
    point=np.array([.435,.249])
    coarse=[list(np.floor(point/.05).astype(int))];fine=[list(np.floor(point/.01).astype(int))]
    assert not nominal_connector([0,0],[0,0],coarse,radius_m=.45)['nominal_disk_connector_clear']
    result=connector([0,0],[0,0],fine,radius_m=.45)
    assert result['nominal_disk_connector_clear'] and result['radius_m']==.45
    obs=replace(observation(200_000_000,[tuple(fine[0])]),coordinate_frame=FRAME)
    assert dispatch_request(plan('left_turn'),obs,now_ns=300_000_000)['requested_command']==[0.,0.,.45]
    near=replace(obs,occupied=frozenset({(42,10)}))
    assert dispatch_request(plan('left_turn'),near,now_ns=300_000_000)['reason']=='CURRENT_OBSERVED_OBSTACLE_VETO'
    with pytest.raises(ValueError):connector([0,0],[.6,0],fine,radius_m=.45)
    with pytest.raises(ValueError):dispatch_request(plan('left_turn'),observation(200_000_000),now_ns=300_000_000)
    mro=FineObstacleRoundTripRuntime.__mro__
    assert mro.index(_FineDispatch)<mro.index(_StoppingDispatch)
