"""Outcome-independent shared map-domain contract, no episode input or physics."""
import numpy as np
from lewm.navigation_capability_map_domain_development import GENERATOR_DIAGONAL_M, MAP_HALF_WIDTH_M, COARSE_HALF_CELLS
from lewm.observed_round_trip_mission_development import point
from lewm.joint_visual_floor_map_development import GRID
from lewm.observed_geometry_refinement_development import segment_cell_distances
from lewm.fine_stored_obstacle_routing_development import FineCellClearance


def check():
    assert GRID.shape==(4*COARSE_HALF_CELLS**2,2), 'whole generator domain must be allocated'
    for sign in (-1,1):
        p=np.array([sign*GENERATOR_DIAGONAL_M,0.])
        np.testing.assert_array_equal(point(p),p)
        assert len(segment_cell_distances(p,p,np.array([[sign*150,0]],dtype=int)))==1
        assert FineCellClearance([(sign*750,0)]).minimum(p,p) is not None
    assert MAP_HALF_WIDTH_M>GENERATOR_DIAGONAL_M+.1

if __name__=='__main__':
    check();print('PASS: generator-sized storage, mission points and coarse/fine clearance domains')
