import numpy as np
import pytest
from lewm.measured_floor_partition_development import MeasuredFloorPartition,foot_projection_coverage


def test_floor_and_later_unknown_same_voxel_both_remain():
    p=MeasuredFloorPartition();p.insert([[.001,.001,.001]],np.array([True]),dict(frame=0))
    p.insert([[.002,.002,.002]],np.array([False]),dict(frame=1))
    assert p.total_returns==2 and p.floor_returns==p.other_returns==1
    assert p.floor.intersect_sphere([0,0,0],.02)['intersecting_voxels']==1
    assert p.other.intersect_sphere([0,0,0],.02)['intersecting_voxels']==1
    assert p.floor.cells[(0,0,0)]['frame']==0 and p.other.cells[(0,0,0)]['frame']==1


def test_unknown_is_not_inferred_from_first_ground_witness():
    p=MeasuredFloorPartition();p.insert([[.001,0,0],[.018,0,0]],np.array([True,False]),dict(frame=0))
    assert p.other.intersect_sphere([.018,0,0],.001)['intersecting_voxels']==1
    assert p.floor.intersect_sphere([.018,0,0],.001)['intersecting_voxels']==0


def test_entire_foot_disk_requires_adjacent_measured_cells_including_boundary():
    r=foot_projection_coverage([.025,.025],.022,{(0,0)})
    assert r['entire_nominal_projection_on_measured_floor'] and not r['ground_support_approved']
    assert not foot_projection_coverage([.04,.025],.022,{(0,0)})['entire_nominal_projection_on_measured_floor']
    assert not foot_projection_coverage([.028,.025],.022,{(0,0)})['entire_nominal_projection_on_measured_floor']


def test_explicit_boolean_classifications_required_before_insertion():
    p=MeasuredFloorPartition()
    with pytest.raises(ValueError):p.insert([[0,0,0]],np.array([1]),dict(frame=0))
    assert p.total_returns==0 and not p.floor.cells and not p.other.cells
