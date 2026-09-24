"""Use the instructed endpoint after an observed route reaches its goal cell."""
import numpy as np
from lewm.observed_floor_waypoint_development import centre,segment_cells,CELL_M
from lewm.observed_geometry_refinement_development import nominal_connector


def terminal_target(proposal,waypoint,position,goal,floor,occupied):
    original=np.asarray(waypoint,float);p=np.asarray(position,float);g=np.asarray(goal,float)
    if any(v.shape!=(2,) or not np.isfinite(v).all() or np.max(np.abs(v))>4.9 for v in (original,p,g)):
        raise ValueError('bounded observed position, waypoint and instructed mission goal required')
    evidence=dict(selected=False,original_waypoint_map_xy_m=original.tolist(),
        mission_goal_map_xy_m=g.tolist(),goal_tolerance_changed=False,
        native_state_used=False,unobserved_floor_inferred=False,ground_support_approved=False)
    path=proposal['route_cells']
    if proposal['status']!='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL' or not path:
        return original.copy(),evidence|dict(reason='ROUTE_DOES_NOT_REACH_GOAL_CELL')
    goal_cell=tuple(map(int,np.floor(g/CELL_M)))
    if tuple(path[-1])!=goal_cell:raise ValueError('goal-route endpoint must match instructed goal cell')
    if not np.array_equal(original,centre(path[-1])):
        return original.copy(),evidence|dict(reason='INTERMEDIATE_WAYPOINT_STILL_REQUIRED')
    cells=segment_cells(p,g);missing=sorted(cells-set(floor))
    check=nominal_connector(p,g,sorted(occupied),radius_m=.45)
    evidence.update(connector_cells=[list(c) for c in sorted(cells)],
        unknown_connector_cells=[list(c) for c in missing],nominal_connector=check)
    if missing:return original.copy(),evidence|dict(reason='EXACT_GOAL_CONNECTOR_HAS_UNOBSERVED_FLOOR')
    if not check['nominal_disk_connector_clear']:
        return original.copy(),evidence|dict(reason='EXACT_GOAL_CONNECTOR_NOT_NOMINALLY_CLEAR')
    return g.copy(),evidence|dict(selected=True,reason='OBSERVED_NOMINALLY_CLEAR_FINAL_GOAL_CONNECTOR')
