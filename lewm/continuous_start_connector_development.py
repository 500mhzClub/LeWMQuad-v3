"""Use continuous disk clearance for entry into the observed floor component."""
from lewm.vectorized_connector_routing_development import propose as grid_propose
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.waypoint_alignment_planning_development import WaypointAlignmentRoundTripRuntime


def clear(start,end,occupied,radius):
    return nominal_connector(start,end,sorted(occupied),radius_m=radius)['nominal_disk_connector_clear']


def propose(floor,occupied,position,goal,*,radius_m=.45):
    result=grid_propose(floor,occupied,position,goal,radius_m=radius_m,connector_clear=clear)
    return result|dict(start_connector_geometry='continuous_disk_against_observed_cell_squares',
        routing_grid_and_occupied_cells_unchanged=True)


class ContinuousStartConnectorRuntime(WaypointAlignmentRoundTripRuntime):
    _propose=staticmethod(propose)
