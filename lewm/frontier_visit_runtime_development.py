"""Finish viewing a reached frontier before selecting another exploration target."""
import math
import numpy as np
from lewm.fine_stored_obstacle_routing_development import FineStoredObstacleRuntime,proposer
from lewm.observed_floor_waypoint_development import centre,NEIGHBOURS,inflated_cells


def wrap(value):return math.atan2(math.sin(value),math.cos(value))


class FrontierVisits:
    def __init__(self,*,standoff_m=None,panorama=False):
        self.excluded=set();self.visit=None;self.events=[]
        self.standoff_m=standoff_m
        self.panorama=panorama

    def advance(self,snapshot,position,heading,goal,now_ns,route):
        observed=snapshot.floor|snapshot.occupied
        if self.visit is not None:
            v=self.visit
            if abs(wrap(v['heading_rad']-heading))<=.1:
                if v['aligned_ns'] is None:v['aligned_ns']=now_ns
                if snapshot.measured_ns>=v['aligned_ns']:
                    if self.panorama:
                        # Replace lists rather than mutating earlier planning receipts.
                        v['completed_view_stages']=v['completed_view_stages']+[dict(
                            heading_rad=v['heading_rad'],aligned_ns=v['aligned_ns'],
                            completed_ns=now_ns,map_frame=snapshot.frame)]
                        if v['view_index']+1<len(v['panorama_headings_rad']):
                            v['view_index']+=1
                            v['heading_rad']=v['panorama_headings_rad'][v['view_index']]
                            v['aligned_ns']=None
                            return 'view',dict(v)
                    target=np.asarray(v['target_xy_m'])
                    # Already viewed nearby targets remain traversable floor.
                    # They are excluded only from exploration target selection.
                    viewed={c for c in snapshot.floor if np.linalg.norm(centre(c)-target)<=.10}
                    self.excluded.update(viewed)
                    self.events.append(v|dict(completed_ns=now_ns,map_frame=snapshot.frame,excluded_cells=len(viewed)))
                    self.visit=None
                    return 'replan',None
            return 'view',dict(v)
        if route['status']!='OBSERVED_FLOOR_ROUTE_TO_FRONTIER':return 'route',None
        target=np.asarray(route['target_map_xy_m'])
        if self.standoff_m is None:
            if np.linalg.norm(target-position)>.10:return 'route',None
        else:
            # Measure remaining route length, so proximity across a wall/bend
            # does not trigger an early view of a still-distant frontier.
            points=[np.asarray(position)]+[centre(c) for c in route['route_cells']]
            remaining=sum(float(np.linalg.norm(b-a)) for a,b in zip(points[:-1],points[1:]))
            if remaining>self.standoff_m:return 'route',None
        cell=tuple(route['route_cells'][-1]);blocked=inflated_cells(snapshot.occupied)
        unknown=[(cell[0]+dx,cell[1]+dy) for dx,dy in NEIGHBOURS
            if (cell[0]+dx,cell[1]+dy) not in observed]
        if not unknown:return 'route',None
        # Prefer a direction outside the known inflated obstacles, then nearer
        # the mission goal. This is a view request, not an unknown-space move.
        neighbour=min(unknown,key=lambda c:(c in blocked,float(np.linalg.norm(centre(c)-goal)),c))
        direction=(np.asarray(neighbour)-np.asarray(cell) if self.standoff_m is None
            else centre(neighbour)-position)
        self.visit=dict(target_cell=list(cell),target_xy_m=target.tolist(),started_ns=now_ns,
            heading_rad=math.atan2(direction[1],direction[0]),aligned_ns=None,
            unknown_neighbour=list(neighbour))
        if self.standoff_m is not None:
            self.visit.update(standoff_view=True,maximum_remaining_route_m=self.standoff_m,
                remaining_route_at_view_start_m=remaining,
                view_start_map_xy_m=np.asarray(position).tolist(),
                physical_frontier_arrival_claimed=False)
        if self.panorama:
            base=self.visit['heading_rad']
            self.visit.update(panorama_headings_rad=[wrap(base+i*math.pi/4) for i in range(9)],
                view_index=0,completed_view_stages=[],panoramic_frontier_view=True,
                full_footprint_visibility_certified=False)
        return 'view',dict(self.visit)


class FrontierVisitRuntime(FineStoredObstacleRuntime):
    frontier_standoff_m=None
    frontier_panorama=False

    def __init__(self,*args,**kwargs):
        self.frontier_visits=FrontierVisits(standoff_m=self.frontier_standoff_m,panorama=self.frontier_panorama)
        super().__init__(*args,**kwargs)

    def _routing_proposer(self,snapshot):
        original=proposer(snapshot)
        def propose(*args,**kwargs):
            excluded=self.frontier_visits.excluded
            if self.mission_latest is not None and self.mission_latest['phase']!='OUTBOUND':excluded=()
            return original(*args,**kwargs,excluded_frontiers=excluded)
        return propose

    def _route(self,snapshot,evidence,goal,*,measured_ns):
        route=super()._route(snapshot,evidence,goal,measured_ns=measured_ns)
        # An actual translation veto owns its measured view until it completes.
        if route['status'] in ('TRANSLATION_VETO_REQUIRES_NEW_VIEW','WAITING_FOR_POST_VETO_VIEW'):return route
        p,R,_=self._pose(evidence,identity=(0,0,0),now_ns=measured_ns)
        B=np.asarray(snapshot.map_from_initial);Q,q=B@R,B@p
        heading=math.atan2(Q[1,0],Q[0,0]);mapped_goal=(B@np.r_[goal,0.])[:2]
        if self.mission_latest is not None and self.mission_latest['phase']!='OUTBOUND':
            self.frontier_visits.visit=None
            return route
        action,visit=self.frontier_visits.advance(snapshot,q[:2],heading,mapped_goal,measured_ns,route)
        if action=='replan':
            return super()._route(snapshot,evidence,goal,measured_ns=measured_ns)
        if action=='view':
            status=('REACHED_FRONTIER_REQUIRES_VIEW' if self.frontier_standoff_m is None
                else 'FRONTIER_STANDOFF_REQUIRES_VIEW')
            return route|dict(route_cells=[],status=status,
                view_heading_rad=visit['heading_rad'],frontier_visit=visit)
        return route
