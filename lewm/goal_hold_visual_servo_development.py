"""Bounded goal-hold correction; sensor-sampled verification, not a safety proof."""
import math
import numpy as np

from lewm.bounded_visual_servo_development import wrapped
from lewm.causal_sensor_state import SensorContractError,_identity,_ns
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.visual_course_response_development import VisualCourseWindow

RULES=dict(target_xy_m=[.4,0.],target_yaw_rad=.3,position_tolerance_m=.025,
    final_position_tolerance_m=.06,yaw_tolerance_rad=.03,final_yaw_tolerance_rad=.05,
    forward_gain=1.8,maximum_forward_m_s=.12,yaw_gain=1.5,maximum_yaw_rad_s=.25,
    course_window_samples=6,minimum_course_speed_m_s=.012,maximum_ticks=350,
    settled_intervals=10,maximum_brake_ticks=40,maximum_observed_translation_m=1.,
    maximum_forward_x_m=.48,settled_speed_m_s=.02,settled_yaw_rate_rad_s=.05,
    maximum_corrections=3,correction_trigger_yaw_rad=.04,correction_arrival_yaw_rad=.015)


class GoalHoldVisualServo:
    def __init__(self, *, identity=(0,0,0)):
        self.identity=_identity(identity); self.course=VisualCourseWindow(); self.stage='forward'
        self.previous=None; self.last_ns=None; self.start_ns=None; self.quiet=0; self.brake_ticks=0
        self.terminal=None; self.reason=None; self.corrections=0; self.previous_in_goal=False

    def step(self,evidence,*,now_ns):
        now=_ns(now_ns,'course servo decision'); command=[0.,0.,0.]; diagnostic={}
        if self.terminal is not None: return self._result(now,command,diagnostic)
        try:
            if self.last_ns is not None and now-self.last_ns!=100_000_000: raise SensorContractError('exact10Hz updates required')
            if (evidence['schema']!='visual_led_motion_evidence_development.v1' or tuple(evidence['identity'])!=self.identity
                    or evidence['decision_ns']!=now or evidence['status']!='CURRENT_VISUAL_POSE'
                    or evidence['terminal_failure'] is not None or evidence['current_pose'] is None):
                raise SensorContractError('same-episode current visual evidence required')
            pose=evidence['current_pose']
            if pose['mode']!='gyro' or pose['measured_ns']!=now or pose['available_ns']>now: raise SensorContractError('current gyro mode required')
            p=np.asarray(pose['position_initial_body_m'],float); R=proper(pose['rotation_initial_body_from_current_body'])
            if p.shape!=(3,) or not np.isfinite(p).all(): raise SensorContractError('finite visual position required')
            yaw=math.atan2(R[1,0],R[0,0]); course=self.course.observe(measured_ns=now,position_initial_xy_m=p[:2],yaw_rad=yaw)
            delta=np.array([.4,0.])-p[:2]; distance=float(np.linalg.norm(delta)); yaw_error=wrapped(.3-yaw)
            speed=rate=None
            if self.previous is not None:
                speed=float(np.linalg.norm(p-self.previous[0])/.1); rate=abs(wrapped(yaw-self.previous[1]))/.1
            previous_in_goal=self.previous_in_goal
            in_goal=distance<=.06 and abs(yaw_error)<=.05
            self.previous_in_goal=in_goal
            self.previous=(p.copy(),yaw); self.last_ns=now
            if self.start_ns is None: self.start_ns=now
            diagnostic=dict(position_initial_body_m=p.tolist(),yaw_rad=yaw,position_error_m=distance,yaw_error_rad=yaw_error,
                observed_speed_m_s=speed,observed_yaw_rate_rad_s=rate,course=course)
            if (now-self.start_ns)//100_000_000>=350: self._fail('TIME_LIMIT')
            elif np.linalg.norm(p)>1. or p[0]>.48: self._fail('OBSERVED_EXCURSION_LIMIT')
            elif self.stage=='forward':
                if distance<=.025: self.stage='forward_brake'; self.brake_ticks=self.quiet=0
                else:
                    measured=course['current_course_hypothesis_rad']
                    effective=yaw if measured is None else measured
                    error=wrapped(math.atan2(delta[1],delta[0])-effective)
                    command=[min(.12,1.8*distance)*max(0.,math.cos(error)),0.,float(np.clip(1.5*error,-.25,.25))]
                    diagnostic|=dict(steering_error_rad=error,course_used=measured is not None,
                        low_course_fallback='body-heading engineering assumption' if measured is None else None)
            elif self.stage in ('turn','corrective_turn'):
                tolerance=.015 if self.stage=='corrective_turn' else .03
                if distance>.06: self._fail('TURN_POSITION_DRIFT')
                elif abs(yaw_error)<=tolerance: self.stage='final_brake'; self.brake_ticks=self.quiet=0
                else: command=[0.,0.,float(np.clip(1.5*yaw_error,-.25,.25))]
            else:
                self.brake_ticks+=1
                quiet=speed is not None and speed<=.02 and rate<=.05
                if self.stage=='final_brake':
                    quiet=quiet and in_goal and previous_in_goal
                    if distance>.06: self._fail('FINAL_POSITION_DRIFT')
                    elif abs(yaw_error)>.04:
                        if self.corrections>=3: self._fail('CORRECTION_LIMIT')
                        else:
                            self.corrections+=1; self.stage='corrective_turn'
                            self.brake_ticks=0
                            command=[0.,0.,float(np.clip(1.5*yaw_error,-.25,.25))]
                        quiet=False
                self.quiet=self.quiet+1 if quiet else 0
                if self.terminal is None and self.stage in ('forward_brake','final_brake') and self.quiet>=10:
                    if self.stage=='forward_brake':
                        if distance>.06: self._fail('FORWARD_BRAKE_TARGET_DRIFT')
                        else: self.stage='turn'
                    elif distance<=.06 and abs(yaw_error)<=.05:
                        self.terminal='VISUAL_TARGET_SEQUENCE_COMPLETE'; self.reason='ten zero-command intervals with visual endpoint pose and settling checks'
                    else: self._fail('FINAL_TARGET_OR_BRAKE_DRIFT')
                elif self.terminal is None and self.brake_ticks>=40: self._fail('BRAKE_NOT_SETTLED')
        except (ValueError,TypeError,KeyError,IndexError) as error: self._fail('VISUAL_INPUT_REJECTED: '+str(error))
        return self._result(now,command if self.terminal is None else [0.,0.,0.],diagnostic)

    def _fail(self,reason): self.terminal='VISUAL_SERVO_FAILED'; self.reason=reason

    def _result(self,now,command,diagnostic):
        return dict(decision_ns=now,stage=self.stage,phase={'forward':1,'forward_brake':2,'turn':3,'final_brake':4,'corrective_turn':5}[self.stage],
            requested_command=command,terminal=self.terminal,reason=self.reason,diagnostic=diagnostic,
            quiet_intervals=self.quiet,brake_ticks=self.brake_ticks,corrections=self.corrections,
            pose_verification_scope='10Hz visual endpoints; no intersample bound',native_pose_used=False,
            supervised_response_fit_used_for_commands=False,terrain_safety_qualified=False,navigation_qualified=False)
