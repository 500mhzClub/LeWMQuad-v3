"""Observe around the initial pose before proposing the first translating route."""
import math
import numpy as np
from lewm.frontier_visit_runtime_development import wrap
from lewm.closed_loop_motion_residual_development import PanoramicMotionResidualRuntime


class InitialPanorama:
    def __init__(self):self.state=None

    @property
    def complete(self):return self.state is not None and self.state['complete']

    def advance(self,snapshot,position,heading,now_ns):
        if self.state is None:
            headings=[wrap(heading+i*math.pi/4) for i in range(9)]
            self.state=dict(started_ns=now_ns,start_map_xy_m=np.asarray(position).tolist(),
                headings_rad=headings,view_index=0,heading_rad=headings[0],aligned_ns=None,
                completed_view_stages=[],complete=False,full_footprint_visibility_certified=False)
        v=self.state
        if not v['complete'] and abs(wrap(v['heading_rad']-heading))<=.1:
            if v['aligned_ns'] is None:v['aligned_ns']=now_ns
            if snapshot.measured_ns>=v['aligned_ns']:
                v['completed_view_stages']=v['completed_view_stages']+[dict(
                    heading_rad=v['heading_rad'],aligned_ns=v['aligned_ns'],completed_ns=now_ns,
                    map_frame=snapshot.frame,map_xy_m=np.asarray(position).tolist())]
                if v['view_index']==8:
                    v.update(complete=True,completed_ns=now_ns)
                else:
                    v['view_index']+=1;v['heading_rad']=v['headings_rad'][v['view_index']];v['aligned_ns']=None
        return dict(v)


class InitialSurveyMixin:
    def __init__(self,*args,**kwargs):
        self.initial_panorama=InitialPanorama()
        super().__init__(*args,**kwargs)

    def _route(self,snapshot,evidence,goal,*,measured_ns):
        if self.initial_panorama.complete:
            return super()._route(snapshot,evidence,goal,measured_ns=measured_ns)
        p,R,_=self._pose(evidence,identity=(0,0,0),now_ns=measured_ns)
        B=np.asarray(snapshot.map_from_initial);q=B@p;Q=B@R
        survey=self.initial_panorama.advance(snapshot,q[:2],math.atan2(Q[1,0],Q[0,0]),measured_ns)
        if survey['complete']:
            return super()._route(snapshot,evidence,goal,measured_ns=measured_ns)|dict(initial_survey=survey)
        return dict(status='INITIAL_PANORAMA_REQUIRES_VIEW',route_cells=[],
            view_heading_rad=survey['heading_rad'],initial_survey=survey)


class InitialSurveyRuntime(InitialSurveyMixin,PanoramicMotionResidualRuntime):
    pass
