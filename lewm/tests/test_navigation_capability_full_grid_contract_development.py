"""Map-origin, translation and edge/corner contracts; no maze inputs or physics."""
from types import SimpleNamespace
import itertools
import numpy as np
import pytest
from lewm.navigation_capability_map_domain_development import COARSE_HALF_CELLS
from lewm.observed_floor_waypoint_development import centre
from lewm.clearance_preferred_route_development import clearance_costs
from lewm.cached_fine_connectivity_development import search_graph, nearby_candidates
from lewm.axis_aligned_fine_connectivity_development import search_graph as axis_search
from lewm.fine_goal_route_development import fine_goal_route
from lewm.cached_fine_goal_route_development import cached_fine_goal_route
from lewm.cached_fine_connectivity_development import fine_goal_route as cached_route
from lewm.axis_aligned_fine_connectivity_development import fine_goal_route as axis_route

ANCHORS=list(itertools.product((-157,0,157),repeat=2))


@pytest.mark.parametrize('anchor',ANCHORS)
def test_every_fine_graph_reads_cost_at_its_own_spatial_cell(anchor):
    x,y=anchor;direction=-1 if x>0 else 1
    cells=[(x+direction*i,y) for i in range(4)]
    floor=frozenset(cells);obstacle=(x,y+(-3 if y>0 else 3))
    occupied=frozenset([obstacle]);fine=frozenset([(obstacle[0]*5,obstacle[1]*5)])
    costs=clearance_costs(occupied)
    expected=sum(.5*(costs[a[0]+COARSE_HALF_CELLS,a[1]+COARSE_HALF_CELLS]+
        costs[b[0]+COARSE_HALF_CELLS,b[1]+COARSE_HALF_CELLS]) for a,b in zip(cells,cells[1:]))
    for function in (search_graph,axis_search):
        found=function(floor,occupied,fine,.01,cells[0],cells[-1])
        assert found is not None
        assert list(found[0])==cells
        assert found[1]==expected, function.__module__
    snapshot=SimpleNamespace(floor=floor,occupied=occupied,fine_occupied=fine)
    for function in (fine_goal_route,cached_fine_goal_route,cached_route,axis_route):
        r=function(snapshot,centre(cells[0]),centre(cells[-1]),dict(
            status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',nominal_radius_m=.01,route_cells=[list(cells[0])]))
        assert r['fine_goal_route']['weighted_path_cost']==expected,function.__module__
    from lewm.clearance_preferred_route_development import preferred_path
    path,receipt=preferred_path(floor,occupied,cells[0],cells[-1],radius_m=.01)
    assert path==[list(c) for c in cells]
    assert receipt['weighted_path_cost']==expected
    assert nearby_candidates(floor,centre(cells[0]))==cells


@pytest.mark.parametrize('xy',list(itertools.product((-7.9,0.,7.9),repeat=2)))
def test_map_and_geometry_consumers_accept_full_domain_without_inventing_floor(xy):
    from lewm.observed_round_trip_mission_development import point
    from lewm.current_observation_planning_map_development import occupied_cells
    from lewm.later_floor_evidence_development import floor_squares
    from lewm.measured_floor_partition_development import foot_projection_coverage
    from lewm.retained_floor_patch_development import RetainedFloorPatches
    from lewm.batched_retained_floor_patch_development import BatchedRetainedFloorPatches
    from lewm.visibility_batched_retained_floor_patch_development import VisibilityBatchedRetainedFloorPatches
    from lewm.progressive_batched_retained_floor_patch_development import ProgressiveBatchedRetainedFloorPatches
    p=np.array(xy);np.testing.assert_array_equal(point(p),p)
    key=tuple(np.floor(p/.05).astype(int))
    assert occupied_cells(np.array([[*p,.2]]),0.)=={key}
    box=np.array([[*(p-.001),-.001],[*(p+.001),.001]])
    cells,reason=floor_squares(box,0.);assert reason is None and key in cells
    assert not foot_projection_coverage(p,.022,set())['entire_nominal_projection_on_measured_floor']
    for cls in (RetainedFloorPatches,BatchedRetainedFloorPatches,VisibilityBatchedRetainedFloorPatches,ProgressiveBatchedRetainedFloorPatches):
        assert not cls().coverage(p[None])[0]['complete_nominal_foot_patch']


@pytest.mark.parametrize('xy',list(itertools.product((-7.9,0.,7.9),repeat=2)))
def test_coarse_routes_segments_coverage_and_geometry_at_edges_and_corners(xy):
    from lewm import observed_floor_waypoint_development as base
    from lewm import continuous_connector_waypoint_development as continuous
    from lewm import reached_frontier_waypoint_development as reached
    from lewm import vectorized_connector_routing_development as vectorized
    from lewm.observed_geometry_refinement_development import segment_cell_distances,nominal_connector
    from lewm.fine_stored_obstacle_routing_development import FineCellClearance
    from lewm.axis_aligned_fine_connectivity_development import AxisGraphClearance
    from lewm.coverage_translation_view_development import footprint_extension
    from lewm.joint_visual_floor_map_development import floor_coverage
    from lewm.frame_cached_floor_geometry_development import FloorFrameGeometry
    from lewm.body_projected_floor_geometry_development import BodyProjectedFloorGeometry
    from lewm.local_floor_routing_map_development import LocalCoverageGeometry
    from lewm.current_plane_floor_coverage_development import CurrentPlaneCoverageGeometry
    from lewm.projected_polygon_floor_coverage_development import ProjectedPolygonCoverageGeometry
    p=np.array(xy);key=tuple(map(int,np.floor(p/.05)));floor=frozenset([key])
    for module in (base,continuous,reached,vectorized):
        proposal=module.propose(floor,set(),p,p,radius_m=.01)
        assert proposal['route_cells']==[list(key)]
    from lewm.fine_stored_obstacle_routing_development import proposer
    snapshot=SimpleNamespace(fine_occupied=frozenset(),occupied=frozenset())
    assert proposer(snapshot)(floor,set(),p,p,radius_m=.01)['route_cells']==[list(key)]
    assert base.segment_cells(p,p)==vectorized.segment_cells(p,p)
    assert segment_cell_distances(p,p,np.array([key]))[0]==0.
    assert not nominal_connector(p,p,[key])['nominal_disk_connector_clear']
    finekey=tuple(np.floor(p/.01).astype(int))
    assert FineCellClearance([finekey]).minimum(p,p)==0.
    assert AxisGraphClearance(frozenset([finekey])).minimum(p,p)==0.
    pred=np.zeros((6,8,5));pred[:,:,3]=1.
    extension=footprint_extension(pred,p,np.eye(3),set())
    assert len(extension['candidates'])==6
    assert all(r['leaves_map_bounds']==(max(abs(p))+.48>8.) for r in extension['candidates'])
    d=np.zeros((480,640),np.float32);v=np.zeros_like(d,bool)
    cells=np.array([key]);position=np.array([*p,0.])
    assert not floor_coverage(d,v,np.eye(3),position,-.32,cells)['covered'].any()
    for constructor in (FloorFrameGeometry,BodyProjectedFloorGeometry,LocalCoverageGeometry,
            lambda:CurrentPlaneCoverageGeometry(True),lambda:ProjectedPolygonCoverageGeometry(True)):
        g=constructor()
        try:assert not g.floor_coverage(d,v,np.eye(3),position,-.32,cells)['covered'].any()
        finally:g.close()


@pytest.mark.parametrize('xy',list(itertools.product((-7.9,0.,7.9),repeat=2)))
def test_current_mapper_pose_to_grid_conversion_at_full_domain(xy):
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.navigation_capability_startup_recovery_development import RecoverableStartupMap
    from lewm.later_floor_evidence_development import LaterFloorEvidence
    from lewm.tests.test_later_floor_evidence_development import observations
    policy,depth,aux,raw,now=packets()
    mapper=RecoverableStartupMap()
    p=np.array([*xy,0.]);R=np.eye(3);pose=raw['current_pose']
    # The fixture supplies an already-qualified pose directly: this tests map
    # coordinates, not registration or tracker estimation.
    mapper._read_pose=lambda *a,**k:(p,R,pose)
    snap=mapper.update(policy,depth,raw,auxiliary_depth=aux,measured_ns=now)
    np.testing.assert_array_equal(snap.position_map,p)
    assert all(-COARSE_HALF_CELLS<=v<COARSE_HALF_CELLS for c in snap.floor|snap.occupied for v in c)
    key=tuple(np.floor(np.array(xy)/.05).astype(int))
    ledger=LaterFloorEvidence();ledger.record_pair(0,now,np.eye(3),-.32,observations(0,[key],[key]))
    assert key in ledger._cell_observations


@pytest.mark.parametrize('xy',list(itertools.product((-7.9,0.,7.9),repeat=2)))
def test_terminal_goal_can_use_its_boundary_cell_centre(xy):
    from lewm.exact_mission_target_development import terminal_target
    from lewm.exact_terminal_waypoint_development import exact_terminal_target
    from lewm.observed_floor_waypoint_development import segment_cells
    from lewm.tests.test_exact_terminal_waypoint_development import selection
    goal=np.array(xy);cell=tuple(map(int,np.floor(goal/.05)));waypoint=centre(cell)
    proposal=dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL',route_cells=[list(cell)])
    target,receipt=terminal_target(proposal,waypoint,goal,goal,segment_cells(goal,goal),set())
    np.testing.assert_array_equal(target,goal);assert receipt['selected']
    s=selection();s['proposal']=proposal;s['waypoint_map_xy_m']=waypoint.tolist()
    r=exact_terminal_target(s,np.r_[goal,0.],np.eye(3),goal)
    np.testing.assert_array_equal(r['waypoint_map_xy_m'],goal)


@pytest.mark.parametrize('xy',list(itertools.product((-7.9,0.,7.9),repeat=2)))
def test_camera_projection_and_view_cell_translation_at_domain_corners(xy):
    from lewm.tests.test_current_position_coverage_view_development import fixture
    from lewm.camera_frontier_viewpoint_development import floor_cell_projection, directed_rotation
    from lewm.current_position_coverage_view_development import CurrentPositionCoverageVisits
    from lewm.navigation_capability_map_domain_development import COARSE_CELL_M
    snapshot,route,p,R,target=fixture()
    if xy[0]>0:target=(-11,0)  # Aim inward so the queried patch remains in the map.
    R,_=directed_rotation(R,p[:2],target)
    shift=np.r_[xy,0.];delta=np.rint(np.asarray(xy)/COARSE_CELL_M).astype(int)
    expected=floor_cell_projection(target,p,R,snapshot.floor_height)
    shifted_target=tuple(np.asarray(target)+delta)
    actual=floor_cell_projection(shifted_target,p+shift,R,snapshot.floor_height)
    for a,b in zip(actual,expected):
        assert a['fully_projected']==b['fully_projected']
        np.testing.assert_allclose(a['lower_pixel'],b['lower_pixel'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(a['upper_pixel'],b['upper_pixel'],atol=1e-10,rtol=0)
    snapshot.floor=frozenset(tuple(np.asarray(c)+delta) for c in snapshot.floor)
    view=CurrentPositionCoverageVisits()._choose(snapshot,route,p+shift,R,shifted_target)
    assert view['current_measured_position_view']
    assert view['viewpoint_cell']==np.floor(np.asarray(xy)/COARSE_CELL_M).astype(int).tolist()


@pytest.mark.parametrize('xy',list(itertools.product((-7.9,0.,7.9),repeat=2)))
def test_inherited_primary_map_conversions_at_corners(xy):
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.joint_visual_floor_map_development import JointVisualFloorMap
    from lewm.frame_cached_floor_map_development import FrameCachedFloorMap
    from lewm.frame_cached_floor_geometry_development import FloorFrameGeometry
    policy,depth,aux,raw,now=packets()
    R=np.diag([-1.,-1.,1.]) if xy[0]>0 else np.eye(3)
    surface=SimpleNamespace(position=np.r_[xy,0.],rotation=R,
        observe=lambda *a,**kw:dict(frame=0,rgb_sha256=depth['rgb_sha256'],depth_sha256='synthetic'))
    for constructor,method in ((JointVisualFloorMap,'observe'),(FrameCachedFloorMap,'_observe_primary')):
        mapper=constructor();mapper.surface=surface;mapper.map_from_initial=np.eye(3);mapper.floor_height=-.32
        geometry=FloorFrameGeometry();mapper.frame_geometry=geometry
        try:
            getattr(mapper,method)(policy,depth,raw,now_ns=now)
            assert mapper.floor
            assert all(-COARSE_HALF_CELLS<=v<COARSE_HALF_CELLS for c in mapper.floor|mapper.occupied for v in c)
        finally:geometry.close()


@pytest.mark.parametrize('xy',list(itertools.product((-7.9,0.,7.9),repeat=2)))
def test_inherited_auxiliary_map_conversions_at_corners(xy):
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.tests.test_floor_pose_registration_development import render_plane
    from lewm.auxiliary_tilted_depth_geometry_development import body_from_optical
    from lewm.auxiliary_tilted_depth_observation_development import from_native_depth
    from lewm.auxiliary_depth_floor_map_development import AuxiliaryDepthSurfaceMemory
    from lewm.auxiliary_downward45_floor_map_development import AuxiliaryDownward45SurfaceMemory
    from lewm.frame_cached_floor_map_development import FrameCachedFloorMemory
    from lewm.frame_cached_floor_geometry_development import FloorFrameGeometry
    policy,depth,aux,raw,now=packets()
    tilted,_=render_plane(body_from_optical(),np.array([0.,0.,1.]),.32)
    tilted=from_native_depth(tilted,policy,measured_ns=now,available_ns=now,now_ns=now)
    R=np.diag([-1.,-1.,1.]) if xy[0]>0 else np.eye(3)
    for cls,method,packet in ((AuxiliaryDepthSurfaceMemory,'observe_auxiliary',tilted),
            (AuxiliaryDownward45SurfaceMemory,'observe_auxiliary',aux),
            (FrameCachedFloorMemory,'_observe_auxiliary_original',aux)):
        memory=cls(identity=(0,0,0));memory._current=lambda *a,**kw:None
        memory.position=np.r_[xy,0.];memory.rotation=R;memory.route=[dict(frame=0)]
        memory.classified_ns=now;geometry=FloorFrameGeometry();memory.frame_geometry=geometry
        floor={};occupied={}
        try:
            getattr(memory,method)(policy,packet,np.eye(3),-.32,floor,occupied,now_ns=now)
            assert floor
            assert all(-COARSE_HALF_CELLS<=v<COARSE_HALF_CELLS for c in floor|occupied for v in c)
        finally:geometry.close()


@pytest.mark.parametrize('xy',list(itertools.product((-7.9,0.,7.9),repeat=2)))
def test_current_global_obstacle_conversion_at_corners(xy,monkeypatch):
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm import fresh_obstacle_dispatch_development as dispatch
    policy,depth,aux,raw,now=packets();p=np.r_[xy,0.]
    monkeypatch.setattr(dispatch,'current_measured_floor_pose',lambda *a,**kw:(p,np.eye(3),dict(frame=0)))
    cloud=dict(points_body_m=np.array([[0.,0.,-.2]]),valid=np.array([True]))
    monkeypatch.setattr(dispatch,'body_points',lambda *a,**kw:cloud)
    monkeypatch.setattr(dispatch,'auxiliary_points',lambda *a,**kw:cloud)
    snapshot=SimpleNamespace(age_ns=lambda **kw:0,map_from_initial=np.eye(3),floor_height=-.32)
    r=dispatch.observe_obstacles(policy,depth,raw,snapshot,auxiliary_depth=aux,measured_ns=now)
    assert r.occupied==frozenset([tuple(map(int,np.floor(np.array(xy)/.05)))])
