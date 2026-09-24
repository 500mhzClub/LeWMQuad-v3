"""Current planning cells, unchanged temporal/contact state, and stale-view rejection."""
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.current_observation_planning_map_development import CurrentObservationPlanningView, occupied_cells
from lewm.current_observation_planning_controller_development import (
    CurrentObservationPlanningController, CurrentObservationPlanningSelector)
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController, MeasuredFloorTransportMap
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_measured_floor_transport_development import item
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from scripts.analyze_go2_ground_plane_development_v1 import URDF

NOW = 1_600_000_000


def view_fixture(*, empty=False):
    surface = SimpleNamespace(position=np.zeros(3), rotation=np.eye(3), route=[{},{}], failed=False)
    def current(now):
        if now != NOW: raise ValueError('stale fixture pose')
    surface._current = current
    floor = {(x,y):0 for x in range(-12,25) for y in range(-12,13)}
    owner = SimpleNamespace(surface=surface, failed=False, floor=floor, occupied={(40,40):0},
        map_from_initial=np.eye(3), floor_height=-.32)
    witnesses=[dict(camera=c,frame=1,measured_ns=NOW) for c in ('primary','auxiliary')]
    visible = {} if empty else {c:0 for c in floor if c[0] <= 9}
    view=CurrentObservationPlanningView(owner,visible,{},witnesses,frame=1,now_ns=NOW)
    return owner,view,witnesses


def test_current_view_changes_route_and_keeps_reobserved_old_cells_without_erasing_history():
    owner,view,_=view_fixture();before=deepcopy(owner.floor)
    old=MeasuredFloorTransportMap.waypoint(owner,[1.,0.],now_ns=NOW)
    new=view.waypoint([1.,0.],now_ns=NOW)
    assert old['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    assert new['status']=='OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    assert old['observed_floor_cells'] > new['observed_floor_cells'] == len(view.floor)
    assert (0,0) in view.floor and view.floor[(0,0)]==1 and owner.floor[(0,0)]==0
    assert not view.occupied and owner.occupied=={(40,40):0}
    assert owner.floor==before and view.surface is owner.surface
    with pytest.raises(TypeError):view.floor[(0,0)]=5
    with pytest.raises(ValueError):view.map_from_initial[0,0]=2


def test_empty_current_observation_never_recovers_historical_floor_as_free_space():
    owner,view,_=view_fixture(empty=True)
    assert owner.floor and view.waypoint([1.,0.],now_ns=NOW)['status']=='ADDITIONAL_VIEW_REQUIRED'
    assert view.receipt['current_floor_cells']==0 and not view.receipt['accumulated_planning_cells_queried']
    assert view.receipt['selector_scan_state_retained']
    assert view.receipt['persistent_contact_history_retained'] and not view.receipt['memoryless_controller']


@pytest.mark.parametrize('fault',['clock','frame','owner_failure','surface_failure','witness_order','witness_time','unmeasured'])
def test_stale_failed_or_unwitnessed_map_cannot_reach_planning(fault):
    owner,view,witnesses=view_fixture()
    if fault=='clock':
        with pytest.raises(ValueError):view.waypoint([1.,0.],now_ns=NOW+100_000_000)
    elif fault in ('frame','owner_failure','surface_failure'):
        if fault=='frame':owner.surface.route.append({})
        elif fault=='owner_failure':owner.failed=True
        else:owner.surface.failed=True
        with pytest.raises(ValueError):view.waypoint([1.,0.],now_ns=NOW)
    else:
        floor=view.floor
        if fault=='witness_order':witnesses.reverse()
        elif fault=='witness_time':witnesses[1]['measured_ns']+=100_000_000
        else:floor={(90,90):1}
        with pytest.raises(ValueError):CurrentObservationPlanningView(owner,floor,{},witnesses,frame=1,now_ns=NOW)


def test_obstacle_cells_keep_original_height_band_and_grid_limits():
    p=np.array([[.02,.02,-.2],[.02,.02,-.29],[.02,.02,.33],[5.,0.,0.],[-5.,0.,0.],[0.,0.,-.32]])
    assert occupied_cells(p,-.32)=={(0,0),(-100,0)}
    with pytest.raises(ValueError):occupied_cells(np.array([[np.nan,0.,0.]]),-.32)


def test_paired_public_pipeline_and_contact_queries_unchanged_through_floor_transport(monkeypatch):
    monkeypatch.setattr(fixture,'visual',partial(visual,origin=1_500_000_000))
    geometry=ArticulatedCollisionGeometry(URDF)
    options=dict(public_mission=dict(goal_initial_body_xy_m=[1.,0.],return_initial_body_xy_m=[0.,0.],
        require_return_after_goal=True),navigation_ticks=100,condition='jepa',variant='full',persistent=True)
    old=MeasuredFloorTransportController(None,geometry,**options)
    new=CurrentObservationPlanningController(None,geometry,**options)
    ids=[id(x) for x in (new.mission,new.registration,new.residual,new.history,new.memory)]
    previous=None;had_less_current_floor=False
    for frame in range(3):
        p,d,a,raw,now,image=item(frame,previous,narrow=frame==1);results=[]
        for controller in (old,new):
            controller.motion=SimpleNamespace(observe=lambda *args,**kwargs:raw)
            result=controller.observe(p,d,None,auxiliary_depth=a,auxiliary_rgb=image,now_ns=now)
            assert result['terminal'] is None,result['failure'];results.append(result)
        baseline,candidate=results;extra=set(candidate)-set(baseline)
        assert extra=={'planning_map_variant','accumulated_planning_cells_queried','selector_scan_state_retained','persistent_contact_history_retained',
            'tracking_and_floor_anchor_history_retained','learned_temporal_history_and_residual_retained',
            'mission_and_settling_state_retained','memoryless_controller'}
        normalized={k:v for k,v in candidate.items() if k not in extra}
        normalized['controller']=baseline['controller'];assert normalized==baseline
        assert new.mapper.floor==old.mapper.floor and new.mapper.occupied==old.mapper.occupied
        assert new.memory.route==old.memory.route and new.residual.snapshot()==old.residual.snapshot()
        assert ids==[id(x) for x in (new.mission,new.registration,new.residual,new.history,new.memory)]
        assert new.selector.residual is new.residual and new.memory is new.mapper.surface
        view=new.mapper.planning_view(now_ns=now)
        assert view.surface is new.memory and set(view.floor)<=set(new.mapper.floor)
        assert all(v==frame for v in view.floor.values())
        had_less_current_floor |= len(view.floor)<len(new.mapper.floor)
        for xy,yaw in (([0.,0.],0.),([.01,0.],.04)):
            assert new.memory.footprint(geometry,xy,yaw,now_ns=now,persistent=True)==old.memory.footprint(geometry,xy,yaw,now_ns=now,persistent=True)
        previous=raw
    assert had_less_current_floor
    assert CurrentObservationPlanningController.observe is MeasuredFloorTransportController.observe
    assert CurrentObservationPlanningController.advance is MeasuredFloorTransportController.advance


def test_entire_selector_chain_uses_current_cells_and_original_contact_memory(monkeypatch):
    from lewm import mission_target_waypoint_selection_development as base
    from lewm.observation_horizon_predictive_selection_development import score_candidates
    from lewm.online_executed_residual_development import OnlineExecutedResidual
    owner,view,_=view_fixture();contacts=[]
    def footprint(geometry,xy,yaw,*,now_ns,persistent):
        contacts.append((geometry,now_ns,persistent));return {'possible_intersection':False}
    owner.surface.footprint=footprint
    owner.planning_view=lambda *,now_ns:view
    forecast=np.zeros((6,8,5));forecast[:,:,3]=1.;forecast[:,:,4]=-10.
    forecast[1,:,0]=np.linspace(.01,.1,8)
    def select(model,history,*,head,input_variant,goal_body_xy_m,contact_penalty_m):
        return score_candidates(forecast,goal_body_xy_m=goal_body_xy_m,contact_penalty_m=contact_penalty_m)|dict(
            prediction=forecast.tolist(),first_prediction_horizon_ns=100_000_000,
            target_offsets_ns=list(range(100_000_000,800_000_001,100_000_000)),selection_wall_ms=0.)
    monkeypatch.setattr(base,'select',select)
    residual=OnlineExecutedResidual();residual.now_ns=NOW;residual.frame=1
    selector=CurrentObservationPlanningSelector(residual=residual,condition='jepa',variant='full',goal_initial_body_xy_m=[1.,0.])
    geometry=object();result=selector.choose(None,None,owner,geometry,now_ns=NOW)
    assert result['proposal']['observed_floor_cells']==len(view.floor)
    assert result['proposal']['occupied_cells']==0 and owner.occupied
    assert result['planning_map_receipt']==view.receipt
    assert all(c['all_predicted_segments_nominally_clear'] for c in result['nominal_path_checks'])
    assert all(s['minimum_observed_cell_distance_m'] is None for c in result['nominal_path_checks'] for s in c['segments'])
    assert len(contacts)==6 and all(c==(geometry,NOW,True) for c in contacts)
    result['planning_map_receipt']['camera_witnesses'].clear();assert len(view.receipt['camera_witnesses'])==2
