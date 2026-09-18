from types import SimpleNamespace
import numpy as np

from lewm.current_position_coverage_view_development import CurrentPositionCoverageVisits


def fixture():
    snapshot=SimpleNamespace(floor=frozenset({(20,0)}),occupied=frozenset(),
        floor_height=-.30,frame=4,measured_ns=100)
    route=dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',route_cells=[[20,0]],
        target_map_xy_m=[.525,.025])
    return snapshot,route,np.zeros(3),np.eye(3),(10,0)


def test_current_pose_view_does_not_require_translating_over_unknown_floor():
    snapshot,route,p,R,target=fixture();views=CurrentPositionCoverageVisits()
    view=views._choose(snapshot,route,p,R,target)
    assert view['current_measured_position_view'] and view['route_cells']==[]
    assert view['viewpoint_map_xy_m']==[0.,0.]
    assert (0,0) not in snapshot.floor and not view['unknown_floor_admitted']
    views.context=(p,R)
    views.visit=dict(unknown_neighbour=list(target),camera_viewpoint=view,
        heading_rad=view['view_heading_rad'],aligned_ns=None)
    status,_=views.advance(snapshot,p[:2],0,np.ones(2),200,route)
    assert status=='view' and views.visit is not None


def test_fresh_unknown_view_is_not_repeated_indefinitely_at_same_pose():
    snapshot,route,p,R,target=fixture();views=CurrentPositionCoverageVisits()
    view=views._choose(snapshot,route,p,R,target); views.context=(p,R)
    views.visit=dict(unknown_neighbour=list(target),camera_viewpoint=view,started_ns=0)
    views._finish(snapshot,100,'FRESH_VIEW_PATCH_STILL_UNKNOWN',False)
    next_view=views._choose(snapshot,route,p,R,target)
    assert next_view is not None and not next_view.get('current_measured_position_view',False)
    assert not views.events[-1]['unknown_cell_observed']


def test_observed_obstacle_between_camera_and_patch_prevents_in_place_proposal():
    snapshot,route,p,R,target=fixture();views=CurrentPositionCoverageVisits()
    snapshot.occupied=frozenset({(8,0)})
    view=views._choose(snapshot,route,p,R,target)
    assert view is None or not view.get('current_measured_position_view',False)
