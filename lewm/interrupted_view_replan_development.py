"""Remember when a camera viewpoint loses visual support before alignment."""
import numpy as np

from lewm.observed_floor_waypoint_development import centre


def retire_interrupted_view(visits,snapshot,position,now_ns):
    visit=visits.visit
    if visit is None:return False
    viewpoint=np.asarray(visit['camera_viewpoint']['viewpoint_map_xy_m'])
    position=np.asarray(position)[:2]
    if np.linalg.norm(position-viewpoint)>.10:return False
    target=tuple(visit['unknown_neighbour'])
    # Exclude this viewing position, not its traversable floor or the unknown
    # target. Use the same spatial extent as fresh-but-unobserved view attempts.
    tried=visits.attempted.setdefault(target,set())
    tried.update(c for c in snapshot.floor if np.linalg.norm(centre(c)-position)<=.10)
    tried.add(tuple(visit['camera_viewpoint']['viewpoint_cell']))
    if visit['camera_viewpoint'].get('current_measured_position_view'):
        visits.failed_current_positions.setdefault(target,[]).append(position.copy())
    visits._finish(snapshot,now_ns,'WEAK_VISUAL_SUPPORT_INTERRUPTED_VIEW',False)
    return True


class InterruptedViewReplanMixin:
    def __init__(self,*args,**kwargs):
        self.last_camera_view_requests={}
        self.last_interrupted_recovery_trigger=-1
        super().__init__(*args,**kwargs)

    def _route(self,snapshot,evidence,goal,*,measured_ns):
        route=super()._route(snapshot,evidence,goal,measured_ns=measured_ns)
        if route['status']!='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW':
            self.last_camera_view_requests={}
            if 'view_heading_rad' in route:
                name=('coverage_views' if self.coverage_view_status=='view' else
                    'frontier_visits' if 'frontier_visit' in route else None)
                visit=None if name is None else getattr(self,name).visit
                if visit is not None:self.last_camera_view_requests[name]=visit['started_ns']
            return route
        active=evidence['visual_support']['recovery_state_at_observation']
        trigger=active['trigger_ns']
        if trigger<=self.last_interrupted_recovery_trigger:return route
        self.last_interrupted_recovery_trigger=trigger
        p,_,_=self._pose(evidence,identity=(0,0,0),now_ns=measured_ns)
        position=np.asarray(snapshot.map_from_initial)@p
        changed={}
        for name,started in self.last_camera_view_requests.items():
            visits=getattr(self,name)
            if visits.visit is not None and visits.visit['started_ns']==started:
                changed[name]=retire_interrupted_view(visits,snapshot,position,measured_ns)
        self.last_camera_view_requests={}
        if any(changed.values()):
            route=route|dict(interrupted_camera_viewpoints=changed)
        # Keep the measured recovery heading and all movement checks. The
        # ordinary view planner selects an alternative after recovery releases.
        return route
