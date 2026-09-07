"""Declared continuous motion/return assay, not autonomous maze exploration.

Corner and home coordinates are stored from actual visual observations. The
fixed stage order is experimental instruction, not a sensed branch detector.
"""
from copy import deepcopy
import math
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.raw_pulse_runtime_development import RawPulseExecution

STAGES=('FORWARD_ONE','TURN_OUT','FORWARD_TWO','TURN_BACK','RETURN_CORNER','TURN_HOME','RETURN_HOME')


def local_target(pose, target_xy, heading):
    p=np.asarray(pose['position_initial_body_m']);R=np.asarray(pose['rotation_initial_body_from_current_body'])
    A=R[:2,:2]
    if np.linalg.svd(A,compute_uv=False)[-1]<.2:raise SensorContractError('planar goal transform ill-conditioned')
    delta=np.linalg.solve(A,np.asarray(target_xy)-p[:2]);length=float(np.linalg.norm(delta))
    final=length<=.4
    if not final:delta*=.4/length
    direction=np.linalg.solve(A,[math.cos(heading),math.sin(heading)])
    return delta.tolist(),math.atan2(direction[1],direction[0]),final


class RoomReturnPulse:
    def __init__(self,sign):
        if type(sign) is not int or sign not in (-1,1):raise ValueError('declared signed assay direction required')
        self.sign=sign;self.runtime=RawPulseExecution();self.index=0;self.pending=None
        self.home=self.corner=None;self.completed=[];self.terminal=None;self.reason=None

    def observe(self,policy,depth,fast,*,now_ns):
        raw=self.runtime.observe(policy,depth,fast,now_ns=now_ns)
        return self.advance(raw,now_ns=now_ns)

    def advance(self,raw,*,now_ns):
        """Factor scheduling from acquisition for explicit synthetic tests."""
        ex=raw['execution'];e=raw['evidence'];dispatch=None
        if self.terminal is None:
            try:
                if ex['status']=='FAILED':raise SensorContractError(ex['reason'])
                pose=e['current_pose']
                if self.home is None:self.home=deepcopy(pose)
                if ex['status']=='IDLE':
                    if self.pending is not None:
                        if self.pending['final']:
                            self.completed.append(dict(stage=STAGES[self.index],decision_ns=now_ns,
                                                       observation=deepcopy(pose),leg_index=self.pending['leg_index']))
                            if self.index==0:self.corner=deepcopy(pose)
                            self.index+=1
                        self.pending=None
                    if self.index==len(STAGES):
                        self.terminal='ROOM_RETURN_CANDIDATE';self.reason='all declared stages and visual final holds complete'
                    else:
                        stage=STAGES[self.index];final=True
                        if stage.startswith('FORWARD'):d,w=[.4,0.],0.
                        elif stage=='TURN_OUT':d,w=[0.,0.],self.sign*math.pi/2
                        elif stage=='TURN_BACK':d,w=[0.,0.],self.sign*math.pi
                        else:
                            destination=self.corner if stage=='RETURN_CORNER' else self.home
                            target=destination['position_initial_body_m'][:2]
                            p=pose['position_initial_body_m'];bearing=math.atan2(target[1]-p[1],target[0]-p[0])
                            heading=0. if stage=='RETURN_HOME' else bearing
                            d,w,final=local_target(pose,target,heading)
                            if stage=='TURN_HOME':d=[0.,0.];final=True
                        leg=self.runtime.executor.begin(d,w,now_ns=now_ns)
                        self.pending=dict(stage=stage,final=final,leg_index=leg['leg_index'])
                        dispatch=deepcopy(self.pending)|dict(goal=leg['goal'])
            except (ValueError,TypeError,KeyError,IndexError) as error:
                self.terminal='ROOM_RETURN_FAILED';self.reason=str(error)
                self.runtime.executor.fail(self.reason,now_ns=now_ns)
        return dict(decision_ns=now_ns,stage=STAGES[self.index] if self.index<len(STAGES) else 'DONE',
                    terminal=self.terminal,reason=self.reason,evidence=e,execution=ex,dispatch=dispatch,
                    requested_command=[0.,0.,0.] if self.terminal else raw['requested_command'],
                    completed_stages=len(self.completed),mission_complete=False,home_verified=False,
                    scripted_motion_assay=True,navigation_qualified=False)

    def snapshot(self):
        return dict(home=deepcopy(self.home),corner=deepcopy(self.corner),completed=deepcopy(self.completed),
                    pending=deepcopy(self.pending),executor=self.runtime.executor.snapshot(),
                    terminal=self.terminal,reason=self.reason,mission_complete=False,home_verified=False)

    def finish_physical_stop(self, reason, *, now_ns):
        self.terminal='ROOM_RETURN_PHYSICAL_STOP';self.reason=str(reason)
        self.runtime.executor.fail('PHYSICAL_STOP: '+str(reason),now_ns=now_ns)
