import numpy as np
import pytest
from lewm.observed_geometry_refinement_development import segment_cell_distances,nominal_connector,sampled_floor_patch
from lewm.observed_floor_waypoint_development import inflated_cells,segment_cells
from lewm.causal_depth_observation_development import FOCAL


def test_continuous_disk_retains_radius_without_cell_start_overapproximation():
    cells=[(11,0)];a=[.107899,.181206];b=[.1,.8]
    assert segment_cells(a,b)&inflated_cells(cells,.45)
    result=nominal_connector(a,b,cells,radius_m=.45)
    assert result['nominal_disk_connector_clear'] and result['minimum_observed_cell_distance_m']>.45
    assert not result['articulated_motion_certified']


@pytest.mark.parametrize('a,b,expected',[
    ([-.1,.025],[.1,.025],0.),([-.1,.05],[.1,.05],0.),
    ([.025,.15],[.025,.15],.1),([.1,.1],[.2,.2],np.sqrt(.005)),
    ([-.1,.15],[.15,.15],.1)])
def test_closed_segment_square_distance(a,b,expected):
    assert segment_cell_distances(a,b,[(0,0)])[0]==pytest.approx(expected,abs=1e-14)


def test_tangent_radius_is_blocked_and_small_positive_margin_is_retained():
    assert not nominal_connector([-.45,.025],[-.45,.025],[(0,0)])['nominal_disk_connector_clear']
    assert nominal_connector([-.450001,.025],[-.450001,.025],[(0,0)])['nominal_disk_connector_clear']


def floor():
    ray=(np.arange(480)+.5-240)/FOCAL
    z=np.broadcast_to(np.divide(.363,ray,out=np.zeros(480),where=ray>0)[:,None],(480,640)).copy()
    valid=(z>=.2)&(z<=5.)
    return np.where(valid,z,0).astype(np.float32),valid


def classify(d,v,height=-.32):
    return sampled_floor_patch(d,v,np.eye(3),np.zeros(3),height,np.array([350]),np.array([320]))


def test_floor_patch_requires_entire_measured_neighbourhood():
    d,v=floor();assert classify(d,v)['measured_floor_patch'][0,0]
    d[349,319]=0;v[349,319]=False
    assert not classify(d,v)['measured_floor_patch'][0,0]


def test_wall_and_wrong_floor_height_do_not_become_ground():
    assert not classify(np.ones((480,640),np.float32),np.ones((480,640),bool))['measured_floor_patch'][0,0]
    d,v=floor();assert not classify(d,v,height=-.25)['measured_floor_patch'][0,0]
