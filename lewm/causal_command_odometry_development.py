"""Measured-command/gyro dead-reckoning BASELINE, not measured translation.

Applied gait velocity commands are not body velocity observations. Integration
can drift even with ideal gyro sensing; no covariance or calibrated accuracy is
claimed. Do not promote this comparison to localization on the strength of its
input contract or synthetic tests. Runtime accesses only deployment-valid packet
fields, never actual pose/twist, map geometry or future realized commands.
"""
import copy

import numpy as np

from lewm.causal_relative_orientation_development import CausalRelativeOrientation
from lewm.causal_sensor_state import SensorContractError,_ns
from lewm.simulated_body_observation_development import validate_policy_packet


class CausalCommandOdometry:
    def __init__(self):
        self.status='NEW'; self.orientation=CausalRelativeOrientation()
        self.position=np.zeros(3); self.rotation=np.eye(3); self._command=None; self.last_ns=None

    def _commands(self,packet,now_ns):
        validate_policy_packet(packet); now_ns=_ns(now_ns,'odometry clock')
        source=packet['sensor_state']['control']['applied_command']
        value={k:np.asarray(source[k]) for k in ('values','valid','measured_ns','available_ns')}
        if (packet['sensor_state']['decision_ns']!=now_ns or packet['image']['measured_ns']!=now_ns
                or now_ns%100_000_000 or value['measured_ns'][-1]!=now_ns or not value['valid'].all()
                or not np.all(np.diff(value['measured_ns'])==100_000_000)):
            raise SensorContractError('complete current regular applied-command history required')
        if np.any(np.abs(value['values'])>np.array([.3,0.,.5])+1e-7):
            raise SensorContractError('command outside baseline bank limits')
        return copy.deepcopy(value)

    def begin(self,packet,*,now_ns):
        if self.status!='NEW': raise SensorContractError('fresh odometry required')
        try:
            command=self._commands(packet,now_ns); state=self.orientation.begin(packet,now_ns=now_ns)
            self.rotation=np.asarray(state['rotation_initial_body_from_current_body'])
            self._command=command; self.last_ns=now_ns; self.status='ACTIVE'
            return self.snapshot(now_ns=now_ns)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.status='FAILED_SENSOR'; raise SensorContractError('odometry initialization failed') from error

    def step(self,packet,*,now_ns):
        if self.status!='ACTIVE': raise SensorContractError('active odometry required')
        try:
            command=self._commands(packet,now_ns)
            if now_ns-self.last_ns!=100_000_000: raise SensorContractError('odometry clock discontinuity')
            old={int(t):i for i,t in enumerate(self._command['measured_ns'])}
            for i,t in enumerate(command['measured_ns']):
                if int(t) in old:
                    for field in ('values','valid','available_ns'):
                        if not np.array_equal(command[field][i],self._command[field][old[int(t)]]):
                            raise SensorContractError('applied command history rewritten')
            state=self.orientation.step(packet,now_ns=now_ns)
            rotation=np.asarray(state['rotation_initial_body_from_current_body'])
            # The sample at t records the applied command of the interval ending
            # at t. Using the previous sample would introduce a one-tick lag at
            # braking/reverse switches. R endpoint trapezoid is an approximation.
            velocity=np.array([*command['values'][-1,:2],0.])
            position=self.position+.05*(self.rotation+rotation)@velocity
            self.position=position; self.rotation=rotation; self._command=command; self.last_ns=now_ns
            return self.snapshot(now_ns=now_ns)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.status='FAILED_SENSOR'; raise SensorContractError('odometry update failed; estimate unavailable') from error

    def snapshot(self,*,now_ns):
        now_ns=_ns(now_ns,'odometry query clock')
        if self.status!='ACTIVE' or now_ns!=self.last_ns: raise SensorContractError('fresh active odometry required')
        return {'status':self.status,'decision_ns':now_ns,'start_ns':self.orientation.start_ns,
            'command_integrated_position_initial_body_m':self.position.tolist(),
            'rotation_initial_body_from_current_body':self.rotation.tolist(),
            'metric_translation_qualified':False,
            'scope':'applied-command/ideal-gyro dead-reckoning baseline; command is not measured velocity'}

    def relative_point(self,point_initial_body_m,*,now_ns):
        self.snapshot(now_ns=now_ns); point=np.asarray(point_initial_body_m,dtype=float)
        if point.shape!=(3,) or not np.isfinite(point).all(): raise SensorContractError('finite initial-frame point required')
        return self.rotation.T@(point-self.position)
