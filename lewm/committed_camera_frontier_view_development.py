"""Complete an initiated viewing turn before reassessing its camera evidence."""
import numpy as np

from lewm.camera_frontier_visits_development import CameraFrontierVisits, CameraFrontierRuntimeMixin
from lewm.camera_frontier_viewpoint_development import directed_rotation, floor_cell_projection
from lewm.frontier_visit_runtime_development import wrap
from lewm.observed_floor_waypoint_development import centre


class CommittedCameraFrontierVisits(CameraFrontierVisits):
    def advance(self, snapshot, position, heading, goal, now_ns, route):
        if self.floor_count!=(len(snapshot.floor),len(snapshot.occupied)):
            self.excluded.clear()
            self.floor_count=(len(snapshot.floor),len(snapshot.occupied))
        if self.visit is None or not self.visit.get('directed_view_committed'):
            action, receipt=super().advance(snapshot,position,heading,goal,now_ns,route)
            if action=='view':
                self.visit.update(directed_view_committed=True,view_commit_started_ns=now_ns,
                    approach_radius_rechecked_during_turn=False,
                    actual_aligned_camera_evidence_required=True)
                receipt=dict(self.visit)
            return action,receipt
        v=self.visit; unknown=tuple(v['unknown_neighbour'])
        if unknown in snapshot.floor|snapshot.occupied:
            self._finish(snapshot,now_ns,'REQUESTED_PATCH_OBSERVED',True)
            return 'replan',None
        if route['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':
            self._finish(snapshot,now_ns,'MISSION_GOAL_ROUTE_AVAILABLE',False)
            return 'route',None
        p,R=self.context
        _,desired=directed_rotation(R,p[:2],unknown)
        v['heading_rad']=desired
        if abs(wrap(desired-heading))>.10:
            v['aligned_ns']=None
            return 'view',dict(v)
        actual=floor_cell_projection(unknown,p,R,snapshot.floor_height)
        v['actual_aligned_projection']=actual
        visible=any(r['fully_projected'] for r in actual)
        if visible:
            if v['aligned_ns'] is None: v['aligned_ns']=now_ns
            if snapshot.measured_ns<v['aligned_ns']: return 'view',dict(v)
        # Alignment alone never observes a patch. An infeasible actual view or
        # a fresh unresolved map asks for a different known-floor viewpoint.
        tried=self.attempted.setdefault(unknown,set())
        tried.add(tuple(v['camera_viewpoint']['viewpoint_cell']))
        tried.update(c for c in snapshot.floor if np.linalg.norm(centre(c)-position)<=.10)
        reason='FRESH_VIEW_PATCH_STILL_UNKNOWN' if visible else 'ALIGNED_VIEW_PATCH_OUTSIDE_IMAGE'
        self._finish(snapshot,now_ns,reason,False)
        return 'replan',None


class CommittedCameraFrontierRuntimeMixin(CameraFrontierRuntimeMixin):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        old=self.frontier_visits
        if old.visit is not None or old.events or old.excluded:
            raise ValueError('install committed camera views before execution')
        self.frontier_visits=CommittedCameraFrontierVisits()
