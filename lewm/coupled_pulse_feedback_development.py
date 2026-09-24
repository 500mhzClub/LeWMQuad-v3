"""Sensor-feedback empirical dynamics baseline, not JEPA or clearance proof."""
import math
import numpy as np
from lewm.anchored_pulse_servo_development import AnchoredPulseServo
from lewm.sensor_anchored_goal_development import current_pose
from lewm.causal_sensor_state import SensorContractError,_ns
from lewm.bounded_visual_servo_development import wrapped
from lewm.coupled_pulse_rollout_development import PulseTable,compose,plan


class CoupledPulseServo(AnchoredPulseServo):
    """Reuse only the fixed goal/excursion/result contract, not old selection.

    Every planner invocation consumes current sensor pose and a fixed empirical
    table. Only its first pulse can be dispatched. No online table adaptation;
    full pulse/brake residuals are recorded for subsequent scientific analysis.
    """
    def __init__(self,goal,table):
        super().__init__(goal)
        if not isinstance(table,PulseTable):raise SensorContractError('fixed typed pulse model required')
        self.table=table
        R=np.asarray(goal.anchor_rotation);yaw=math.atan2(R[1,0],R[0,0])
        self.anchor_yaw=self.unwrapped_yaw=yaw
        self.target_unwrapped_yaw=goal.target_yaw_rad+2*math.pi*round(
            (yaw+goal.requested_yaw_delta_rad-goal.target_yaw_rad)/(2*math.pi))
        self.previous=(np.asarray(goal.anchor_position),yaw);self.last_ns=goal.anchor_ns
        self.previous_in_goal=False

    def step(self,evidence,*,now_ns):
        now=self.last_ns;command=[0.,0.,0.];diagnostic={}
        if self.terminal is not None:return self._result(now,command,diagnostic)
        try:
            now=_ns(now_ns,'coupled feedback decision')
            if now-self.last_ns!=100_000_000:raise SensorContractError('exact10Hz required')
            p,R,_=current_pose(evidence,identity=self.identity,now_ns=now)
            yaw=math.atan2(R[1,0],R[0,0]);increment=wrapped(yaw-self.previous[1])
            if abs(increment)>.20:raise SensorContractError('gyro-supported incremental yaw bound exceeded')
            self.unwrapped_yaw+=increment
            distance=float(np.linalg.norm(np.array(self.goal.target_xy)-p[:2]))
            error=self.target_unwrapped_yaw-self.unwrapped_yaw
            speed=float(np.linalg.norm(p-self.previous[0])/.1);rate=abs(increment)/.1
            in_goal=distance<=.06 and abs(error)<=.05;prior_in_goal=self.previous_in_goal
            quiet=speed<=.02 and rate<=.05
            self.previous=(p.copy(),yaw);self.previous_in_goal=in_goal;self.last_ns=now
            diagnostic=dict(position_initial_body_m=p.tolist(),yaw_rad=yaw,
                unwrapped_yaw_rad=self.unwrapped_yaw,target_unwrapped_yaw_rad=self.target_unwrapped_yaw,
                position_error_m=distance,yaw_error_rad=error,observed_speed_m_s=speed,observed_yaw_rate_rad_s=rate)
            if (now-self.start_ns)//100_000_000>=1000:self._fail('TIME_LIMIT')
            elif self._excursion(p):self._fail('OBSERVED_EXCURSION_LIMIT')
            elif self.stage=='pulse':
                if self.remaining:
                    command=list(self.action['command']);self.remaining-=1
                else:self.stage='brake';self.brake_ticks=self.quiet=0
            elif self.stage=='brake':
                self.brake_ticks+=1;self.quiet=self.quiet+1 if quiet else 0
                if self.brake_ticks>=20 and self.quiet>=10:
                    if self.action is not None:
                        body=self.action['R'].T@(p-self.action['p'])
                        observed=np.r_[body[:2],self.unwrapped_yaw-self.action['unwrapped_yaw']]
                        predicted=np.asarray(self.table.effects[self.action['index']].delta_xy_yaw)
                        diagnostic['completed_action_response']=dict(kind=self.action['kind'],pulse_index=self.pulses,
                            action_index=self.action['index'],displacement_start_body_m=body.tolist(),
                            yaw_change_rad=float(observed[2]),measured_ns=now,brake_ticks=self.brake_ticks,
                            predicted_delta_xy_yaw=predicted.tolist(),observed_minus_predicted= (observed-predicted).tolist())
                        self.action=None
                    self.stage='choose'
                elif self.brake_ticks>=40:self._fail('BRAKE_NOT_SETTLED')
            elif self.stage=='final_hold':
                self.quiet=self.quiet+1 if quiet and in_goal and prior_in_goal else 0
                self.brake_ticks+=1
                if not in_goal:self.stage='choose';self.quiet=0
                elif self.quiet>=10:
                    self.terminal='VISUAL_TARGET_SEQUENCE_COMPLETE';self.reason='actual final pose, signed yaw and ten quiet zero intervals'
                elif self.brake_ticks>=40:self._fail('FINAL_HOLD_NOT_SETTLED')
            if self.terminal is None and self.stage=='choose':
                self.task_stage='joint_pose'
                if in_goal:self.stage='final_hold';self.quiet=self.brake_ticks=0
                elif self.pulses>=35:self._fail('PULSE_LIMIT')
                else:
                    start=[p[0],p[1],self.unwrapped_yaw]
                    proposal=plan(self.table,start,[*self.goal.target_xy,self.target_unwrapped_yaw],
                                  yaw_mode='winding',horizon=min(24,35-self.pulses))
                    diagnostic['rollout']=proposal
                    if not proposal['action_indices']:
                        self._fail('PLANNER_NO_PROGRESS')
                    else:
                        index=proposal['action_indices'][0];effect=self.table.effects[index]
                        endpoint=compose(start,effect.delta_xy_yaw)
                        if self._excursion(np.r_[endpoint[:2],p[2]]):self._fail('PLANNED_ENDPOINT_EXCURSION_LIMIT')
                        else:
                            # SEARCH_EXHAUSTED with a nonempty best partial path is
                            # explicitly exploratory progress, not a reachability claim.
                            command=list(effect.command);kind='forward' if command[0] else 'turn'
                            self.action=dict(kind=kind,index=index,command=command.copy(),p=p.copy(),R=R.copy(),
                                             yaw=yaw,unwrapped_yaw=self.unwrapped_yaw)
                            self.remaining=effect.ticks-1;self.pulses+=1;self.stage='pulse';self.quiet=0
                            diagnostic['new_pulse']=dict(kind=kind,ticks=effect.ticks,pulse_index=self.pulses,
                                action_index=index,predicted_endpoint=endpoint.tolist(),
                                predicted_full_goal=proposal['status']=='PREDICTED_GOAL_CANDIDATE')
        except (ValueError,TypeError,KeyError,IndexError) as error:self._fail('COUPLED_INPUT_OR_PLANNER_REJECTED: '+str(error))
        return self._result(now,command if self.terminal is None else [0.,0.,0.],diagnostic)

    def _result(self,now,command,diagnostic):
        return super()._result(now,command,diagnostic)|dict(
            controller='coupled_empirical_pulse_feedback_v1',fitting_audit_sha256=self.table.fitting_audit_sha256,
            anchor_yaw_rad=self.anchor_yaw,target_unwrapped_yaw_rad=self.target_unwrapped_yaw,
            yaw_semantics='net_signed_winding',online_replanning=True,online_model_adaptation=False)
