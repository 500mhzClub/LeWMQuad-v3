"""Causal body-rate relative turning; simulation-preparation, not calibrated IMU odometry."""
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError,_identity
from lewm.simulated_body_observation_development import validate_policy_packet


def rotation_increment(rotation_vector):
    vector=np.asarray(rotation_vector,dtype=float)
    if vector.shape!=(3,) or not np.isfinite(vector).all(): raise SensorContractError('finite body rotation vector required')
    angle=float(np.linalg.norm(vector))
    x,y,z=vector
    cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    if angle<1e-6:
        a=1-angle*angle/6; b=.5-angle*angle/24
    else: a=math.sin(angle)/angle; b=(1-math.cos(angle))/(angle*angle)
    return np.eye(3)+a*cross+b*(cross@cross)


def wrap(angle): return math.atan2(math.sin(angle),math.cos(angle))


class RelativeGyroTurn:
    """One bounded relative turn from fresh body gyros, with zero-motion release.

    Integrated orientation is relative to the INITIAL BODY frame, not world yaw.
    No absolute heading, base pose/velocity, map or future sample is consumed.
    Translation drift, contact and bias are not estimated or guaranteed safe.
    """
    def __init__(self):
        self.status='UNINITIALIZED'; self.identity=None; self.rotation=np.eye(3)
        self.last_ns=None; self.last_gyro=None; self.start_ns=None; self.stable_since=None

    def _packet(self,packet):
        validate_policy_packet(packet)
        state=packet['sensor_state']; ns=int(state['decision_ns'])
        if packet['image']['measured_ns']!=ns or ns%100_000_000:
            raise SensorContractError('turn controller requires current command-clock packet')
        gyro=state['sensed']['gyro']
        if gyro['measured_ns'][-1]!=ns or not gyro['valid'][-1].all():
            raise SensorContractError('current valid gyro required')
        return state,ns,gyro

    def begin(self,packet,target_yaw_rad):
        if self.status!='UNINITIALIZED': raise SensorContractError('fresh turn controller required for a new maneuver')
        if isinstance(target_yaw_rad,bool) or not math.isfinite(target_yaw_rad) or abs(target_yaw_rad)>math.pi:
            raise ValueError('relative target must be finite and within plus/minus pi')
        state,ns,gyro=self._packet(packet)
        self.identity=_identity(state['identity']); self.start_ns=self.last_ns=ns
        self.last_gyro=np.asarray(gyro['values'][-1],dtype=float).copy()
        self.target=float(target_yaw_rad); self.status='TURNING'
        return self._command(ns)

    def _command(self,ns):
        forward=self.rotation[:,0]
        if math.hypot(forward[0],forward[1])<.2:
            self.status='FAILED_ORIENTATION'; return self._result([0.,0.,0.],None,None)
        yaw=math.atan2(forward[1],forward[0]); error=wrap(self.target-yaw)
        # Angular velocity transformed into the initial-body frame provides the
        # instantaneous projected heading rate, not merely current body omega_z.
        omega=self.rotation@self.last_gyro
        derivative=np.cross(omega,forward)
        yaw_rate=float((forward[0]*derivative[1]-forward[1]*derivative[0])/(forward[0]**2+forward[1]**2))
        in_tolerance=abs(error)<=.08 and abs(yaw_rate)<=.1
        dwell_complete=in_tolerance and self.stable_since is not None and ns-self.stable_since>=300_000_000
        if ns-self.start_ns>=12_000_000_000 and not dwell_complete:
            self.status='FAILED_TIMEOUT'; return self._result([0.,0.,0.],yaw,error)
        if in_tolerance:
            if self.stable_since is None: self.stable_since=ns
            self.status='COMPLETE' if ns-self.stable_since>=300_000_000 else 'SETTLING'
            return self._result([0.,0.,0.],yaw,error)
        self.stable_since=None
        self.status='TURNING'
        command=[0.,0.,float(np.clip(1.5*error,-.35,.35))]
        return self._result(command,yaw,error)

    def _result(self,command,yaw,error):
        return {'status':self.status,'requested_command':command,'relative_heading_rad':yaw,
            'heading_error_rad':error,'rotation_initial_body_from_current_body':self.rotation.tolist(),
            'last_gyro_ns':self.last_ns,'scope':'relative gyro integration only; no translational/contact/hardware qualification'}

    def step(self,packet):
        if self.status=='UNINITIALIZED': raise SensorContractError('begin the turn before stepping')
        if self.status=='COMPLETE' or self.status.startswith('FAILED_'):
            return self._result([0.,0.,0.],None,None)
        try:
            state,ns,gyro=self._packet(packet)
            if _identity(state['identity'])!=self.identity or ns-self.last_ns!=100_000_000:
                raise SensorContractError('turn episode or command-clock discontinuity')
            times=np.asarray(gyro['measured_ns']); present=np.flatnonzero(times==self.last_ns)
            if len(present)!=1 or not gyro['valid'][present[0]].all() or not np.array_equal(gyro['values'][present[0]],self.last_gyro):
                raise SensorContractError('gyro overlap rewritten or missing')
            rotation=self.rotation.copy(); previous=self.last_gyro.copy(); last=self.last_ns
            for index in np.flatnonzero(times>self.last_ns):
                if times[index]-last!=20_000_000 or not gyro['valid'][index].all():
                    raise SensorContractError('invalid or missing 50-Hz gyro sample')
                current=np.asarray(gyro['values'][index],dtype=float)
                rotation=rotation@rotation_increment((previous+current)*.01)
                previous=current; last=int(times[index])
            if last!=ns: raise SensorContractError('gyro history does not reach current decision')
            # Commit only after validating the entire chunk: no partially
            # integrated state can be reused after a dropped/invalid sample.
            self.rotation=rotation; self.last_gyro=previous.copy(); self.last_ns=last
            return self._command(ns)
        except (SensorContractError,ValueError,TypeError,KeyError) as error:
            self.status='FAILED_SENSOR'
            return self._result([0.,0.,0.],None,None) | {'failure_reason':str(error)}
