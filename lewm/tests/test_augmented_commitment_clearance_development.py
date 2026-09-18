import pytest
from scripts.read_go2_augmented_commitment_clearance_v1 import path_witnesses


def test_interior_path_conflict_is_not_hidden_by_clear_endpoints():
    result=path_witnesses([[-.6,0],[0,0],[.7,0]],[(0,0)])
    assert result['first_sample_inside_nominal_radius']==1
    assert result['sampled_radius_conflicts']==1
    assert result['sampled_minimum_distance_to_witness_m']==0
    assert not result['all_occupied_squares_checked']


def test_empty_witness_set_cannot_establish_clearance():
    result=path_witnesses([[0,0],[.1,0]],[])
    assert result['sampled_minimum_distance_to_witness_m'] is None
    assert result['sampled_radius_conflicts']==0
    assert not result['continuous_articulated_motion_certified']
    with pytest.raises(ValueError):path_witnesses([],[(0,0)])
