"""Fixed development controller interventions; no simulator or model imports."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from lewm.physical_execution_development import build_case, rotation_xyzw, evaluate_execution

ARMS = ('baseline','prealign','arrival_feedback','combined')


def wrapped_error(target, actual):
    return math.atan2(math.sin(target-actual), math.cos(target-actual))


def trial_spec(kind, width, arm):
    if arm not in ARMS:
        raise ValueError('unknown controller arm')
    spec=build_case(kind,width)
    spec['scene_id']=f'go2-local-control-factorial-dev-v1-{kind}-width-{int(width*100):03d}-{arm}'
    spec['procedural_seed']=2026090600+spec['case_index']
    spec['arm']=arm
    return spec


def continuation_geometry(geometry):
    """Next observed corridor region; scene collision geometry is unchanged."""
    import copy
    result=copy.deepcopy(geometry)
    edge=result['selected_directed_edge']
    opening=np.asarray(edge['opening_segment_world'],dtype=float)
    normal=np.asarray(edge['opening_normal_world'],dtype=float)
    next_opening=opening+.6*normal
    def region(start,end):
        return [*(opening+start*normal).tolist(), *(opening+end*normal)[::-1].tolist()]
    result['source_node']={'node_id':'corridor_first','centre_world':(opening.mean(axis=0)+.3*normal).tolist(),
                           'boundary_polygon_world':region(0,.6)}
    result['target_node']={'node_id':'corridor_second','centre_world':(opening.mean(axis=0)+.95*normal).tolist(),
                           'boundary_polygon_world':region(.8,1.2)}
    edge.update(edge_id='continuation-edge',source_node_id='corridor_first',target_node_id='corridor_second',
                opening_segment_world=next_opening.tolist(),edge_region_polygon_world=region(.6,1.2))
    result['teacher_route_polyline_world']=[opening.mean(axis=0).tolist(),next_opening.mean(axis=0).tolist(),
                                           (opening.mean(axis=0)+1.1*normal).tolist()]
    return result


def motion_window_ok(poses,twists,geometry,width):
    """All of the last 100 physics samples meet the unchanged arrival targets."""
    poses=np.asarray(poses,dtype=float)
    twists=np.asarray(twists,dtype=float)
    if len(poses)<100:
        return False
    if poses.shape[1:]!=(7,) or twists.shape!=(len(poses),6):
        raise ValueError('motion window shape mismatch')
    edge=geometry['selected_directed_edge']
    opening=np.asarray(edge['opening_segment_world'])
    normal=np.asarray(edge['opening_normal_world'])
    tangent=(opening[1]-opening[0])/np.linalg.norm(opening[1]-opening[0])
    desired=math.atan2(normal[1],normal[0])
    for pose,twist in zip(poses[-100:],twists[-100:],strict=True):
        if not np.isfinite(pose).all() or not np.isfinite(twist).all():
            raise ValueError('nonfinite feedback measurement')
        rotation=rotation_xyzw(pose[3:])
        yaw=math.atan2(rotation[1,0],rotation[0,0])
        roll=math.atan2(rotation[2,1],rotation[2,2])
        pitch=math.asin(float(np.clip(-rotation[2,0],-1,1)))
        delta=pose[:2]-opening.mean(axis=0)
        if (delta@normal<.02 or abs(delta@tangent)>width/2-.10 or abs(wrapped_error(desired,yaw))>.35
                or np.linalg.norm(twist[:2])>.10 or abs(twist[5])>.25
                or pose[2]<.20 or max(abs(roll),abs(pitch))>.50):
            return False
    return True


def evaluate_edge(spec,arrays,*,stop_reason,crossing):
    result=evaluate_execution(spec,arrays,stop_reason=stop_reason,crossing=crossing)
    arrival=arrays['phase']==2
    result['checks']['braking_phase_completed']=bool(np.count_nonzero(arrival)>=250)
    result['checks']['edge_time_budget']=bool(np.count_nonzero(arrays['phase']!=0)<=4250)
    result['status']='SUCCESS' if all(result['checks'].values()) else 'PHYSICAL_FAILURE'
    result['sustained_arrival_window']=motion_window_ok(arrays['base_pose_world'][arrival],
        arrays['base_twist_world'][arrival],spec['geometry'],spec['width_m'])
    result['active_time_s']=float(np.count_nonzero(arrays['phase']!=0)*.002)
    result['arrival_time_s']=float(np.count_nonzero(arrival)*.002)
    return result


@dataclass
class LocalController:
    arm: str

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError('unknown controller arm')
        self.stage='ALIGN' if self.arm in ('prealign','combined') else 'APPROACH'
        self.aligned_boundaries=0
        self.arrival_start_tick=None
        self.terminal_reason=None

    def decide(self, *, tick, alignment_error, pursuit_error, arrival_error,
               body_forward_velocity, angular_velocity, crossed, stable_arrival):
        if not isinstance(tick,int) or isinstance(tick,bool) or not 0<=tick<=85:
            raise ValueError('invalid command tick')
        if not all(math.isfinite(value) for value in (alignment_error,pursuit_error,arrival_error,body_forward_velocity,angular_velocity)):
            raise ValueError('nonfinite controller input')
        if self.stage=='DONE':
            return None
        if self.stage=='ALIGN':
            self.aligned_boundaries=self.aligned_boundaries+1 if abs(alignment_error)<=.10 and abs(angular_velocity)<=.25 else 0
            if self.aligned_boundaries>=2:
                self.stage='APPROACH'
        if self.stage in ('ALIGN','APPROACH') and (tick>=80 or (self.stage=='APPROACH' and crossed)):
            self.stage='ARRIVE'
            self.arrival_start_tick=tick
        feedback=self.arm in ('arrival_feedback','combined')
        if self.stage=='ARRIVE' and tick-self.arrival_start_tick>=5 and (not feedback or stable_arrival):
            self.stage='DONE'
            self.terminal_reason='ARRIVAL_POLICY_FINISHED'
            return None
        if tick==85:
            self.stage='DONE'
            self.terminal_reason='EDGE_BUDGET_EXHAUSTED'
            return None
        if self.stage=='ALIGN':
            return [0.,0.,float(np.clip(1.5*alignment_error,-.45,.45))]
        if self.stage=='APPROACH':
            forward=0. if abs(pursuit_error)>=1.20 else float(np.clip(.25*math.cos(pursuit_error),.08,.25))
            return [forward,0.,float(np.clip(1.5*pursuit_error,-.45,.45))]
        if not feedback:
            return [0.,0.,0.]
        return [float(np.clip(-.5*body_forward_velocity,-.08,.08)),0.,
                float(np.clip(1.5*arrival_error-.3*angular_velocity,-.45,.45))]
