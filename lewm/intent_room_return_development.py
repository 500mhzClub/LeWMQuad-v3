"""Declared continuous motion/return assay, not autonomous maze exploration.

Corner and home coordinates are stored from actual visual observations. The
fixed stage order is experimental instruction, not a sensed branch detector.
"""
from copy import deepcopy
from lewm.causal_sensor_state import SensorContractError
from lewm.coupled_room_return_development import RawCoupledPulseExecution
from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion
from lewm.metric_return_intent_development import ReturnIntent
from lewm.room_return_pulse_development import STAGES
import math


class IntentRoomReturn:
    def __init__(self,sign,table,*,multi_reference=True):
        if type(sign) is not int or sign not in (-1,1):raise ValueError('declared signed assay direction required')
        if type(multi_reference) is not bool:raise ValueError('explicit observer choice required')
        self.sign=sign;self.runtime=RawCoupledPulseExecution(table);self.index=0;self.pending=None
        if multi_reference:self.runtime.motion=MultiReferenceVisualLedMotion()
        self.return_intent=None
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
                            self.index+=1;self.return_intent=None
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
                            if self.return_intent is None:self.return_intent=ReturnIntent.create(stage,pose,target)
                            if self.return_intent.stage!=stage:raise SensorContractError('return intent stage mismatch')
                            d,w,final=self.return_intent.subgoal(pose)
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
                    pending=deepcopy(self.pending),return_intent=None if self.return_intent is None else self.return_intent.snapshot(),
                    executor=self.runtime.executor.snapshot(),
                    terminal=self.terminal,reason=self.reason,mission_complete=False,home_verified=False)

    def finish_physical_stop(self, reason, *, now_ns):
        self.terminal='ROOM_RETURN_PHYSICAL_STOP';self.reason=str(reason)
        self.runtime.executor.fail('PHYSICAL_STOP: '+str(reason),now_ns=now_ns)
