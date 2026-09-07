"""Finite bank-command pulses with causal visual feedback; not JEPA."""
import math
import numpy as np

from lewm.bounded_visual_servo_development import wrapped
from lewm.causal_sensor_state import SensorContractError,_identity,_ns
from lewm.joint_rgbd_rigid_pose_development import proper

RULES=dict(target_xy_m=[.4,0.],target_yaw_rad=.3,approach_tolerance_m=.025,
    final_position_tolerance_m=.06,final_yaw_tolerance_rad=.05,
    maximum_ticks=1000,maximum_pulses=35,forward_m_s=.20,yaw_rad_s=.45,
    short_pulse_ticks=2,long_pulse_ticks=5,long_forward_distance_m=.12,
    approach_steering_tolerance_rad=.15,long_turn_error_rad=.30,
    minimum_brake_ticks=20,maximum_brake_ticks=40,settled_intervals=10,
    settled_speed_m_s=.02,settled_yaw_rate_rad_s=.05,
    maximum_observed_translation_m=1.,maximum_forward_x_m=.48,
    minimum_forward_response_m=.005)


class PulseFeedbackServo:
    def __init__(self,*,identity=(0,0,0)):
        self.identity=_identity(identity);self.stage='brake';self.task_stage='approach'
        self.last_ns=None;self.start_ns=None;self.previous=None;self.previous_in_goal=False
        self.quiet=0;self.brake_ticks=0;self.remaining=0;self.pulses=0
        self.action=None;self.forward_offset=None;self.terminal=None;self.reason=None

    def step(self,evidence,*,now_ns):
        now=_ns(now_ns,'pulse feedback decision');command=[0.,0.,0.];diagnostic={}
        if self.terminal is not None:return self._result(now,command,diagnostic)
        try:
            if self.last_ns is not None and now-self.last_ns!=100_000_000:raise SensorContractError('exact10Hz required')
            if (evidence['schema']!='visual_led_motion_evidence_development.v1' or tuple(evidence['identity'])!=self.identity
                    or evidence['decision_ns']!=now or evidence['status']!='CURRENT_VISUAL_POSE'
                    or evidence['terminal_failure'] is not None or evidence['current_pose'] is None):
                raise SensorContractError('same-episode current visual evidence required')
            pose=evidence['current_pose'];p=np.asarray(pose['position_initial_body_m'],float)
            R=proper(pose['rotation_initial_body_from_current_body'])
            if (pose['mode']!='gyro' or pose['measured_ns']!=now or pose['available_ns']>now
                    or p.shape!=(3,) or not np.isfinite(p).all()):raise SensorContractError('finite current gyro pose required')
            yaw=math.atan2(R[1,0],R[0,0]);delta=np.array([.4,0.])-p[:2]
            distance=float(np.linalg.norm(delta));yaw_error=wrapped(.3-yaw)
            speed=rate=None
            if self.previous is not None:
                speed=float(np.linalg.norm(p-self.previous[0])/.1);rate=abs(wrapped(yaw-self.previous[1]))/.1
            in_goal=distance<=.06 and abs(yaw_error)<=.05;prior_in_goal=self.previous_in_goal
            quiet=speed is not None and speed<=.02 and rate<=.05
            self.previous=(p.copy(),yaw);self.previous_in_goal=in_goal;self.last_ns=now
            if self.start_ns is None:self.start_ns=now
            diagnostic=dict(position_initial_body_m=p.tolist(),yaw_rad=yaw,position_error_m=distance,
                yaw_error_rad=yaw_error,observed_speed_m_s=speed,observed_yaw_rate_rad_s=rate)
            if (now-self.start_ns)//100_000_000>=1000:self._fail('TIME_LIMIT')
            elif np.linalg.norm(p)>1. or p[0]>.48:self._fail('OBSERVED_EXCURSION_LIMIT')
            elif self.stage=='pulse':
                if self.remaining:
                    command=list(self.action['command']);self.remaining-=1
                else:
                    self.stage='brake';self.brake_ticks=self.quiet=0
            elif self.stage=='brake':
                # Count only intervals following a zero-command decision.
                if speed is not None:self.brake_ticks+=1
                self.quiet=self.quiet+1 if quiet else 0
                if self.brake_ticks>=20 and self.quiet>=10:
                    if self.action is not None:
                        body=self.action['R'].T@(p-self.action['p'])
                        response=dict(kind=self.action['kind'],pulse_index=self.pulses,
                            displacement_start_body_m=body.tolist(),measured_ns=now,
                            yaw_change_rad=wrapped(yaw-self.action['yaw']))
                        diagnostic['completed_action_response']=response
                        if self.action['kind']=='forward' and np.linalg.norm(body[:2])>=.005:
                            self.forward_offset=math.atan2(body[1],body[0])
                        self.action=None
                    self.stage='choose'
                elif self.brake_ticks>=40:self._fail('BRAKE_NOT_SETTLED')
            elif self.stage=='final_hold':
                self.quiet=self.quiet+1 if quiet and in_goal and prior_in_goal else 0
                self.brake_ticks+=1
                if not in_goal:
                    self.stage='choose';self.quiet=0
                elif self.quiet>=10:
                    self.terminal='VISUAL_TARGET_SEQUENCE_COMPLETE';self.reason='actual final pose and ten quiet zero intervals'
                elif self.brake_ticks>=40:self._fail('FINAL_HOLD_NOT_SETTLED')
            if self.terminal is None and self.stage=='choose':
                if self.task_stage=='approach' and distance<=.025:self.task_stage='orient'
                if self.task_stage=='orient' and distance>.06:self.task_stage='approach'
                if self.task_stage=='orient' and in_goal:
                    self.stage='final_hold';self.quiet=self.brake_ticks=0
                else:
                    if self.task_stage=='approach':
                        offset=0. if self.forward_offset is None else self.forward_offset
                        error=wrapped(math.atan2(delta[1],delta[0])-yaw-offset)
                        diagnostic|=dict(forward_steering_error_rad=error,
                            forward_response_offset_rad=self.forward_offset,
                            forward_response_transport_assumption='latest forward-pulse effect persists across intervening turns')
                        kind='turn' if abs(error)>.15 else 'forward'
                    else:kind='turn';error=yaw_error
                    if self.pulses>=35:self._fail('PULSE_LIMIT')
                    else:
                        count=(5 if distance>.12 else 2) if kind=='forward' else (5 if abs(error)>.30 else 2)
                        command=[.20,0.,0.] if kind=='forward' else [0.,0.,math.copysign(.45,error)]
                        self.action=dict(kind=kind,command=command.copy(),p=p.copy(),R=R.copy(),yaw=yaw)
                        self.remaining=count-1;self.pulses+=1;self.stage='pulse';self.quiet=0
                        diagnostic['new_pulse']=dict(kind=kind,ticks=count,pulse_index=self.pulses)
        except (ValueError,TypeError,KeyError,IndexError) as error:self._fail('VISUAL_INPUT_REJECTED: '+str(error))
        return self._result(now,command if self.terminal is None else [0.,0.,0.],diagnostic)

    def _fail(self,reason):self.terminal='VISUAL_SERVO_FAILED';self.reason=reason

    def _result(self,now,command,diagnostic):
        phase=4 if self.stage=='final_hold' else 2
        if self.stage=='pulse':phase=1 if self.action['kind']=='forward' else 3
        return dict(decision_ns=now,stage=self.stage,task_stage=self.task_stage,phase=phase,
            requested_command=command,terminal=self.terminal,reason=self.reason,diagnostic=diagnostic,
            quiet_intervals=self.quiet,brake_ticks=self.brake_ticks,pulse_count=self.pulses,
            remaining_pulse_ticks=self.remaining,forward_response_offset_rad=self.forward_offset,
            native_pose_used=False,learned_predictor_used=False,terrain_safety_qualified=False,navigation_qualified=False)
