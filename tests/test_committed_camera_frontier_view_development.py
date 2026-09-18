from types import SimpleNamespace
import math
import numpy as np

from lewm.committed_camera_frontier_view_development import CommittedCameraFrontierVisits
from lewm.camera_frontier_viewpoint_development import directed_rotation


def setup():
    snapshot=SimpleNamespace(floor=frozenset((x,y) for x in range(11) for y in range(-5,6)),
        occupied=frozenset(),fine_occupied=frozenset(),floor_height=-.32,measured_ns=0,frame=0)
    route=dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',route_cells=[[x,0] for x in range(11)],
        target_map_xy_m=[.525,.025],goal_map_xy_m=[2.,.025])
    visits=CommittedCameraFrontierVisits()
    c,s=math.cos(1.),math.sin(1.)
    visits.context=(np.zeros(3),np.array([[c,-s,0],[s,c,0],[0,0,1]]))
    action,_=visits.advance(snapshot,np.zeros(2),1.,np.array([2.,.025]),100,route)
    assert action=='view'
    return visits,snapshot,route


def test_turn_continues_after_body_drifts_outside_approach_radius():
    visits,snapshot,route=setup();floor=snapshot.floor
    p=np.array([.22,0.,0.]);visits.context=(p,visits.context[1])
    action,receipt=visits.advance(snapshot,p[:2],1.,np.array([2.,.025]),200,route)
    assert action=='view' and receipt['directed_view_committed']
    assert not visits.events and not visits.attempted and snapshot.floor is floor
    assert (11,0) not in snapshot.floor


def test_aligned_but_blind_view_retries_without_observing_patch():
    visits,snapshot,route=setup();floor=snapshot.floor
    original=tuple(visits.visit['camera_viewpoint']['viewpoint_cell'])
    p=np.array([.52,0.,0.]);R,heading=directed_rotation(np.eye(3),p[:2],(11,0))
    visits.context=(p,R)
    action,_=visits.advance(snapshot,p[:2],heading,np.array([2.,.025]),200,route)
    assert action=='replan' and visits.visit is None
    assert visits.events[-1]['completion_reason']=='ALIGNED_VIEW_PATCH_OUTSIDE_IMAGE'
    assert not visits.events[-1]['unknown_cell_observed']
    assert original in visits.attempted[(11,0)] and snapshot.floor is floor


def test_actual_mapped_patch_releases_committed_turn():
    visits,snapshot,route=setup();snapshot.floor=snapshot.floor|{(11,0)}
    action,_=visits.advance(snapshot,np.zeros(2),1.,np.array([2.,.025]),200,route)
    assert action=='replan' and visits.events[-1]['unknown_cell_observed']


def test_aligned_visible_view_waits_for_fresh_map_and_preserves_unknown():
    visits,snapshot,route=setup();floor=snapshot.floor
    R,heading=directed_rotation(np.eye(3),np.zeros(2),(11,0))
    visits.context=(np.zeros(3),R)
    action,_=visits.advance(snapshot,np.zeros(2),heading,np.array([2.,.025]),200,route)
    assert action=='view' and visits.visit['aligned_ns']==200
    snapshot.measured_ns=200
    action,_=visits.advance(snapshot,np.zeros(2),heading,np.array([2.,.025]),300,route)
    assert action=='replan' and snapshot.floor is floor and (11,0) not in floor
    assert visits.events[-1]['completion_reason']=='FRESH_VIEW_PATCH_STILL_UNKNOWN'
    assert not visits.events[-1]['unknown_cell_observed']


def test_new_goal_route_takes_priority_over_committed_turn():
    visits,snapshot,route=setup();route['status']='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    action,_=visits.advance(snapshot,np.zeros(2),1.,np.array([2.,.025]),200,route)
    assert action=='route' and visits.visit is None
    assert visits.events[-1]['completion_reason']=='MISSION_GOAL_ROUTE_AVAILABLE'
