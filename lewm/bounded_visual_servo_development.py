"""Fixed engineered visual feedback experiment, not a terrain-safe policy."""
import math
import numpy as np

from lewm.causal_sensor_state import SensorContractError, _identity, _ns
from lewm.joint_rgbd_rigid_pose_development import proper

RULES=dict(target_xy_m=[.4,0.],target_yaw_rad=.3,position_tolerance_m=.025,
    final_position_tolerance_m=.06,yaw_tolerance_rad=.03,final_yaw_tolerance_rad=.05,
    forward_gain=.8,yaw_gain=1.5,maximum_forward_m_s=.1,minimum_forward_m_s=.04,
    maximum_yaw_rad_s=.25,settled_speed_m_s=.02,settled_yaw_rate_rad_s=.05,
    settled_intervals=10,maximum_brake_ticks=40,maximum_ticks=200,
    maximum_observed_translation_m=1.,maximum_forward_overshoot_m=.08)


def wrapped(x): return math.atan2(math.sin(x),math.cos(x))


class BoundedVisualServo:
    def __init__(self, *, identity=(0,0,0)):
        self.identity=_identity(identity); self.stage='forward'; self.previous=None
        self.start_ns=None; self.last_ns=None; self.brake_ticks=0; self.quiet=0
        self.terminal=None; self.reason=None

    def step(self,evidence,*,now_ns):
        now=_ns(now_ns,'servo decision'); diagnostic={}; command=[0.,0.,0.]
        if self.terminal is not None: return self._result(now,command,diagnostic)
        try:
            if self.last_ns is not None and now-self.last_ns!=100_000_000:
                raise SensorContractError('exact 10Hz servo updates required')
            if (evidence['schema']!='visual_led_motion_evidence_development.v1'
                    or _identity(tuple(evidence['identity']))!=self.identity
                    or evidence['decision_ns']!=now or evidence['status']!='CURRENT_VISUAL_POSE'
                    or evidence['terminal_failure'] is not None or evidence['current_pose'] is None):
                raise SensorContractError('current same-episode visual pose required')
            pose=evidence['current_pose']
            if pose['mode']!='gyro' or pose['measured_ns']!=now or pose['available_ns']>now:
                raise SensorContractError('predeclared current gyro-conditioned pose required')
            p=np.asarray(pose['position_initial_body_m'],float); R=proper(pose['rotation_initial_body_from_current_body'])
            if p.shape!=(3,) or not np.isfinite(p).all(): raise SensorContractError('finite visual position required')
            yaw=math.atan2(R[1,0],R[0,0]); delta=np.asarray(RULES['target_xy_m'])-p[:2]
            distance=float(np.linalg.norm(delta)); yaw_error=wrapped(RULES['target_yaw_rad']-yaw)
            speed=rate=None
            if self.previous is not None:
                speed=float(np.linalg.norm(p-self.previous[0])/.1)
                rate=abs(wrapped(yaw-self.previous[1]))/.1
            diagnostic=dict(position_initial_body_m=p.tolist(),yaw_rad=yaw,position_error_m=distance,
                yaw_error_rad=yaw_error,observed_speed_m_s=speed,observed_yaw_rate_rad_s=rate)
            self.previous=(p.copy(),yaw)
            if self.start_ns is None: self.start_ns=now
            self.last_ns=now
            if (now-self.start_ns)//100_000_000>=RULES['maximum_ticks']:
                self._fail('TIME_LIMIT')
            elif np.linalg.norm(p)>RULES['maximum_observed_translation_m'] or p[0]>.4+RULES['maximum_forward_overshoot_m']:
                self._fail('OBSERVED_EXCURSION_LIMIT')
            elif self.stage=='forward':
                if distance<=RULES['position_tolerance_m']:
                    self.stage='forward_brake'; self.brake_ticks=self.quiet=0
                else:
                    heading=wrapped(math.atan2(delta[1],delta[0])-yaw)
                    v=min(.1,max(.04,RULES['forward_gain']*distance))*max(0.,math.cos(heading))
                    command=[v,0.,float(np.clip(RULES['yaw_gain']*heading,-.25,.25))]
            elif self.stage=='turn':
                if abs(yaw_error)<=RULES['yaw_tolerance_rad']:
                    self.stage='final_brake'; self.brake_ticks=self.quiet=0
                else: command=[0.,0.,float(np.clip(RULES['yaw_gain']*yaw_error,-.25,.25))]
            else:
                self.brake_ticks+=1
                quiet=speed is not None and speed<=.02 and rate<=.05
                self.quiet=self.quiet+1 if quiet else 0
                if self.quiet>=RULES['settled_intervals']:
                    if self.stage=='forward_brake':
                        if distance>.06: self._fail('FORWARD_BRAKE_TARGET_DRIFT')
                        else: self.stage='turn'
                    elif distance<=.06 and abs(yaw_error)<=.05:
                        self.terminal='VISUAL_TARGET_SEQUENCE_COMPLETE'; self.reason='visual targets and settling met'
                    else: self._fail('FINAL_TARGET_OR_BRAKE_DRIFT')
                elif self.brake_ticks>=RULES['maximum_brake_ticks']: self._fail('BRAKE_NOT_SETTLED')
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self._fail('VISUAL_INPUT_REJECTED: '+str(error))
        return self._result(now,command if self.terminal is None else [0.,0.,0.],diagnostic)

    def _fail(self,reason): self.terminal='VISUAL_SERVO_FAILED'; self.reason=reason

    def _result(self,now,command,diagnostic):
        return dict(decision_ns=now,stage=self.stage,phase={'forward':1,'forward_brake':2,'turn':3,'final_brake':4}[self.stage],
            requested_command=command,terminal=self.terminal,reason=self.reason,diagnostic=diagnostic,
            quiet_intervals=self.quiet,brake_ticks=self.brake_ticks,native_pose_used=False,
            stage_assumption='controlled continuous level floor and ideal hidden-robot RGB-D',
            terrain_safety_qualified=False,navigation_qualified=False)
