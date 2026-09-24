"""Request measured coverage before extending a translating footprint into unknown."""
from copy import deepcopy
import itertools
import math

import numpy as np

from lewm.camera_frontier_visits_development import CameraFrontierVisits
from lewm.clearance_turn_recovery_development import choose
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.observed_geometry_refinement_development import segment_cell_distances
from lewm.observed_floor_waypoint_development import centre

TRANSLATIONS = ('forward', 'left_arc', 'right_arc')


def footprint_extension(prediction, position, rotation, floor):
    prediction = np.asarray(prediction, float)
    if prediction.shape != (6, 8, 5) or not np.isfinite(prediction).all():
        raise ValueError('complete finite candidate forecasts required')
    paths = np.asarray(position)[:2]+np.concatenate((np.zeros((6,1,2)),
        prediction[:,:,:2]),axis=1)@np.asarray(rotation)[:2,:2].T
    low = np.floor((paths.min(axis=(0,1))-.48)/.05).astype(int)
    high = np.floor((paths.max(axis=(0,1))+.48)/.05).astype(int)
    # The retained floor grid is bounded; extension outside it is unobserved.
    bounded_low = np.maximum(low, -100); bounded_high = np.minimum(high, 99)
    cells = np.array(list(itertools.product(range(bounded_low[0],bounded_high[0]+1),
        range(bounded_low[1],bounded_high[1]+1))),dtype=int).reshape(-1,2)
    unknown = np.array([tuple(c) not in floor for c in cells], dtype=bool)
    current = (segment_cell_distances(paths[0,0],paths[0,0],cells)<=.48)&unknown
    masks = [np.logical_or.reduce([segment_cell_distances(a,b,cells)<=.48
        for a,b in zip(path,path[1:])])&unknown for path in paths]
    baseline = current|masks[ACTIONS.index('hold')]
    return dict(radius_m=.48,current_unknown_cells=int(current.sum()),
        baseline_unknown_cells=int(baseline.sum()),
        current_and_hold_unknown_not_declared_free=True,
        candidates=[dict(action=action,new_unknown_cells=int((mask&~baseline).sum()),
            added_cells=cells[mask&~baseline].tolist(),
            leaves_map_bounds=bool((path.min(0)-.48 < -5).any() or (path.max(0)+.48 > 5).any()))
            for action,path,mask in zip(ACTIONS,paths,masks)])


def filter_translation(selection, prediction, snapshot, position, rotation):
    if selection['action'] not in TRANSLATIONS:
        return selection, None
    receipt = footprint_extension(prediction,position,rotation,snapshot.floor)
    selected = receipt['candidates'][ACTIONS.index(selection['action'])]
    if selected['new_unknown_cells']==0 and not selected['leaves_map_bounds']:
        return selection | dict(translation_footprint_coverage=receipt | dict(rejected=False)), None
    utilities = {r['action']:r['utility_m'] for r in selection.get('scan_utilities',selection['candidates'])}
    stopping = {r['action']:r['projection_clear'] for r in selection['planned_stopping_projection']['candidates']}
    eligible = [r['action'] for r in selection['memory_forecast_candidates'] if
        r['action'] in ('hold','left_turn','right_turn') and r['nominal_predicted_path_clear']
        and stopping[r['action']] and r['action'] in utilities]
    action = max(eligible,key=lambda a:utilities[a]) if eligible else 'hold'
    result = deepcopy(selection); choose(result,ACTIONS.index(action))
    result['translation_footprint_coverage'] = receipt | dict(rejected=True,
        previous_action=selection['action'],selected_action=action,
        obstacle_reserve_and_stopping_checks_unchanged=True,dispatch_guards_unchanged=True,
        footprint_safety_certified=False)
    observed = snapshot.floor | snapshot.occupied
    targets = [tuple(c) for c in selected['added_cells'] if tuple(c) not in observed]
    target = min(targets,key=lambda c:(float(np.linalg.norm(centre(c)-np.asarray(position)[:2])),c)) if targets else None
    return result, target


class CoverageTranslationViewMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.coverage_views = CameraFrontierVisits()
        self.coverage_target = None
        self.coverage_generation = self.mission_generation
        self.coverage_view_status = None

    def _route(self,snapshot,evidence,goal,*,measured_ns):
        route = super()._route(snapshot,evidence,goal,measured_ns=measured_ns)
        if self.coverage_generation != self.mission_generation:
            self.coverage_views.visit = None; self.coverage_target = None
            self.coverage_generation = self.mission_generation
        self.coverage_view_status = None
        target = self.coverage_target
        if target is None:return route
        if target in snapshot.floor | snapshot.occupied:
            if self.coverage_views.visit is not None:
                self.coverage_views._finish(snapshot,measured_ns,'REQUESTED_PATCH_OBSERVED',True)
            self.coverage_target = None
            self.coverage_view_status = 'REQUESTED_PATCH_OBSERVED'
            return route
        if route['status']=='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW':
            self.coverage_view_status = 'WEAK_VIEW_RECOVERY_HAS_PRIORITY'
            return route
        p,R,_ = self._pose(evidence,identity=(0,0,0),now_ns=measured_ns)
        B = np.asarray(snapshot.map_from_initial);q,Q = B@p,B@R
        heading = math.atan2(Q[1,0],Q[0,0])
        query = route | dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',
            target_map_xy_m=centre(target).tolist())
        self.coverage_views.context = (q,Q)
        if self.coverage_views.visit is None:
            view = self.coverage_views._choose(snapshot,query,q,Q,target)
            if view is None:
                self.coverage_view_status = 'NO_REACHABLE_UNTRIED_COVERAGE_VIEWPOINT'
                return route
            self.coverage_views.visit = dict(target_cell=list(target),target_xy_m=centre(target).tolist(),
                unknown_neighbour=list(target),started_ns=measured_ns,aligned_ns=None,
                heading_rad=view['view_heading_rad'],camera_viewpoint=view,
                view_start_map_xy_m=q[:2].tolist(),camera_visibility_is_hypothesis=True,
                physical_frontier_arrival_claimed=False)
        status,view = self.coverage_views.advance(snapshot,q[:2],heading,(B@np.r_[goal,0])[:2],measured_ns,query)
        self.coverage_view_status = status
        if status=='view':
            self.clearance_turn = None
            return query | dict(route_cells=[],view_heading_rad=view['heading_rad'],
                status='FOOTPRINT_EXTENSION_REQUIRES_OBSERVED_VIEW')
        if status=='route':
            # The viewpoint route, rather than an inherited scan heading, now
            # defines the objective. It still uses the original route machinery.
            query.pop('view_heading_rad',None)
            return query
        return route

    def _select_clear_prediction(self,selected,prediction,snapshot,position,rotation):
        result = super()._select_clear_prediction(selected,prediction,snapshot,position,rotation)
        result,target = filter_translation(result,prediction,snapshot,position,rotation)
        if result.get('translation_footprint_coverage',{}).get('rejected'):
            self.clearance_turn = None
            if self.coverage_target is None and target is not None:
                self.coverage_target = target
                self.coverage_view_status = 'TRANSLATION_REQUIRES_COVERAGE'
        if self.coverage_target is not None or self.coverage_view_status is not None:
            result = result | dict(coverage_view_request=dict(
                target_cell=None if self.coverage_target is None else list(self.coverage_target),
                status=self.coverage_view_status,
                visit=None if self.coverage_views.visit is None else deepcopy(self.coverage_views.visit),
                recent_events=deepcopy(self.coverage_views.events[-2:])))
        return result
