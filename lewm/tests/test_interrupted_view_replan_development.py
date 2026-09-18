from copy import deepcopy
from types import SimpleNamespace
import numpy as np

from lewm.camera_frontier_visits_development import CameraFrontierVisits
from lewm.current_position_coverage_view_development import CurrentPositionCoverageVisits
from lewm.interrupted_view_replan_development import retire_interrupted_view,InterruptedViewReplanMixin


def fixture(visits=None):
    visits=CameraFrontierVisits() if visits is None else visits
    visits.visit=dict(unknown_neighbour=[10,0],started_ns=1,
        camera_viewpoint=dict(viewpoint_cell=[0,0],viewpoint_map_xy_m=[.025,.025]))
    snap=SimpleNamespace(floor=frozenset({(0,0),(1,0),(8,0)}),occupied=frozenset(),frame=4)
    return visits,snap


def test_interruption_retires_only_viewpoint_without_resolving_or_removing_floor():
    visits,snap=fixture();before=snap.floor
    assert retire_interrupted_view(visits,snap,[.025,.025],100)
    assert visits.visit is None and visits.attempted[(10,0)]=={(0,0),(1,0)}
    assert not visits.events[-1]['unknown_cell_observed'] and snap.floor==before
    assert visits.events[-1]['completion_reason']=='WEAK_VISUAL_SUPPORT_INTERRUPTED_VIEW'


def test_distant_approach_does_not_disqualify_an_untried_viewpoint():
    visits,snap=fixture();before=deepcopy(visits.visit)
    assert not retire_interrupted_view(visits,snap,[.4,.025],100)
    assert visits.visit==before and not visits.attempted and not visits.events


def test_in_place_view_is_recorded_in_its_current_position_exclusion():
    visits,snap=fixture(CurrentPositionCoverageVisits())
    visits.visit['camera_viewpoint']['current_measured_position_view']=True
    assert retire_interrupted_view(visits,snap,[.025,.025],100)
    np.testing.assert_array_equal(visits.failed_current_positions[(10,0)][0],[.025,.025])


class Parent:
    def _route(self,*args,**kwargs):
        if not getattr(self,'recovering',True):
            return dict(status='FRONTIER_STANDOFF_REQUIRES_VIEW',route_cells=[],
                view_heading_rad=2.,frontier_visit=deepcopy(self.frontier_visits.visit))
        return dict(status='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW',route_cells=[],view_heading_rad=.9)
    def _pose(self,*args,**kwargs):return np.array([.025,.025,0]),np.eye(3),{}


class Runtime(InterruptedViewReplanMixin,Parent):pass


def test_measured_recovery_objective_stays_active_after_view_is_retired():
    runtime=Runtime();runtime.frontier_visits,snap=fixture()
    runtime.coverage_views=CurrentPositionCoverageVisits();snap.map_from_initial=np.eye(3)
    runtime.recovering=False;runtime.coverage_view_status=None
    runtime._route(snap,{},np.zeros(2),measured_ns=80)
    runtime.recovering=True
    evidence=dict(visual_support=dict(recovery_state_at_observation=dict(trigger_ns=90)))
    result=runtime._route(snap,evidence,np.zeros(2),measured_ns=100)
    assert result['status']=='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW'
    assert result['view_heading_rad']==.9 and result['route_cells']==[]
    assert runtime.frontier_visits.visit is None


def test_one_recovery_does_not_disqualify_new_unexecuted_alternate_views():
    runtime=Runtime();runtime.frontier_visits,snap=fixture()
    runtime.coverage_views=CurrentPositionCoverageVisits();snap.map_from_initial=np.eye(3)
    runtime.last_camera_view_requests={'frontier_visits':1}
    evidence=dict(visual_support=dict(recovery_state_at_observation=dict(trigger_ns=90)))
    runtime._route(snap,evidence,np.zeros(2),measured_ns=100)
    runtime.frontier_visits,_=fixture();runtime.frontier_visits.visit['started_ns']=101
    runtime._route(snap,evidence,np.zeros(2),measured_ns=104)
    assert runtime.frontier_visits.visit is not None and not runtime.frontier_visits.events


def test_view_without_a_prior_heading_request_is_not_treated_as_attempted():
    runtime=Runtime();runtime.frontier_visits,snap=fixture()
    runtime.coverage_views=CurrentPositionCoverageVisits();snap.map_from_initial=np.eye(3)
    evidence=dict(visual_support=dict(recovery_state_at_observation=dict(trigger_ns=90)))
    runtime._route(snap,evidence,np.zeros(2),measured_ns=100)
    assert runtime.frontier_visits.visit is not None and not runtime.frontier_visits.events
